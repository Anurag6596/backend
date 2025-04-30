import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model():
    # Load the InceptionV3 model pre-trained on ImageNet
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(4, activation='softmax')(x)  # 4 soil types: Red, Black, Clay, Alluvial
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze the base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def train_model(train_dir, validation_dir, epochs=10, batch_size=32):
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Only rescaling for validation
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    # Flow from directory
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Create model
    model = create_model()
    
    # Train model
    logger.info("Starting model training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Save model
    model.save('soil_model.h5')
    logger.info("Model trained and saved as soil_model.h5")
    
    return history, model

def fine_tune_model(model, train_dir, validation_dir, epochs=5, batch_size=32):
    # Unfreeze some layers for fine-tuning
    for layer in model.layers[:249]:
        layer.trainable = False
    for layer in model.layers[249:]:
        layer.trainable = True
    
    # Recompile the model with a lower learning rate
    model.compile(optimizer=Adam(learning_rate=0.00001), 
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    # Fine-tune the model
    logger.info("Starting fine-tuning...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size
    )
    
    # Save the fine-tuned model
    model.save('soil_model.h5')
    logger.info("Fine-tuned model saved as soil_model.h5")
    
    return history, model

def plot_training_history(history):
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

if __name__ == "__main__":
    # Set paths for training and validation data
    # These should be directories with subdirectories for each class
    # Example: train_dir/red_soil/, train_dir/black_soil/, etc.
    train_dir = "./Dataset/Train"
    validation_dir = "./Dataset/test"
    
    # Check if directories exist
    if not os.path.exists(train_dir) or not os.path.exists(validation_dir):
        logger.error(f"Training or validation directory does not exist: {train_dir}, {validation_dir}")
        logger.info("Please prepare your dataset with the following structure:")
        logger.info("./Dataset/Train/red_soil/")
        logger.info("./Dataset/Train/black_soil/")
        logger.info("./Dataset/Train/clay_soil/")
        logger.info("./Dataset/Train/alluvial_soil/")
        logger.info("./Dataset/test/red_soil/")
        logger.info("./Dataset/test/black_soil/")
        logger.info("./Dataset/test/clay_soil/")
        logger.info("./Dataset/test/alluvial_soil/")
    else:
        # Train the model
        history, model = train_model(train_dir, validation_dir, epochs=10)
        
        # Fine-tune the model
        fine_tune_history, model = fine_tune_model(model, train_dir, validation_dir, epochs=5)
        
        # Plot training history
        plot_training_history(fine_tune_history)
        
        logger.info("Model training complete. The model is saved as 'soil_model.h5'")
