import os
import logging
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from werkzeug.utils import secure_filename
from data_processor import CropRecommender

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "soil_analysis_secret_key")

# Enable CORS
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Soil types that can be predicted
SOIL_TYPES = ['Red', 'Black', 'Clay', 'Alluvial']

# Initialize crop recommender
crop_recommender = CropRecommender('./Crop_recommendation.csv')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_soil_image(img_path):
    """
    Analyze soil image and predict soil type using color features
    
    Args:
        img_path: Path to the soil image
        
    Returns:
        str: Predicted soil type
    """
    try:
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize for consistency
        img_array = np.array(img)
        
        # Extract color features
        r_mean = np.mean(img_array[:,:,0])
        g_mean = np.mean(img_array[:,:,1])
        b_mean = np.mean(img_array[:,:,2])
        
        # Simple rule-based classifier based on RGB values
        # These thresholds are approximations and can be adjusted
        if r_mean > 150 and g_mean < 100 and b_mean < 100:
            return 'Red'  # High red component
        elif r_mean < 80 and g_mean < 80 and b_mean < 80:
            return 'Black'  # Low values across all channels
        elif r_mean > 120 and g_mean > 120 and b_mean < 100:
            return 'Clay'  # Higher red and green, lower blue
        else:
            return 'Alluvial'  # Default to alluvial for other combinations
            
    except Exception as e:
        logger.error(f"Error analyzing soil image: {str(e)}")
        return None

def preprocess_image(img_path):
    """
    Preprocess image for analysis
    
    Args:
        img_path: Path to the image
        
    Returns:
        PIL.Image: Processed image
    """
    try:
        img = Image.open(img_path)
        img = img.resize((224, 224))  # Resize for consistency
        return img
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if file is in request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    # Check if file is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Please upload a JPG, JPEG or PNG image.'}), 400
    
    try:
        # Get parameters from request
        n = float(request.form.get('nitrogen', 0))
        p = float(request.form.get('phosphorus', 0))
        k = float(request.form.get('potassium', 0))
        ph = float(request.form.get('ph', 0))
        temperature = float(request.form.get('temperature', 0))
        humidity = float(request.form.get('humidity', 0))
        rainfall = float(request.form.get('rainfall', 0))
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Predict soil type using simplified color analysis
        predicted_soil_type = analyze_soil_image(filepath)
        if predicted_soil_type is None:
            return jsonify({'error': 'Error analyzing soil image'}), 500
        
        # Get recommended crops using our data processor
        recommended_crops = crop_recommender.recommend_crops(
            predicted_soil_type, n, p, k, ph, temperature, humidity, rainfall
        )
        
        # Get the main recommended crop (first one in the list or a default)
        recommended_crop = recommended_crops[0] if recommended_crops else "No specific recommendation"
        
        # Analyze parameters
        analysis = {
            'N': crop_recommender.analyze_parameter('N', n),
            'P': crop_recommender.analyze_parameter('P', p),
            'K': crop_recommender.analyze_parameter('K', k),
            'pH': crop_recommender.analyze_parameter('pH', ph),
            'temperature': crop_recommender.analyze_parameter('temperature', temperature),
            'humidity': crop_recommender.analyze_parameter('humidity', humidity),
            'rainfall': crop_recommender.analyze_parameter('rainfall', rainfall)
        }
        
        # Return results
        return jsonify({
            'soil_type': predicted_soil_type,
            'recommended_crop': recommended_crop,
            'recommended_crops': recommended_crops,
            'analysis': analysis
        })
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'OK'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
