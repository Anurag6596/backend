import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def get_crop_recommendation(soil_type, n, p, k, ph, temperature, humidity, rainfall):
    """
    Get crop recommendation based on soil type and environmental parameters
    
    Args:
        soil_type (str): The predicted soil type
        n (float): Nitrogen value
        p (float): Phosphorus value
        k (float): Potassium value
        ph (float): pH value
        temperature (float): Temperature value
        humidity (float): Humidity value
        rainfall (float): Rainfall value
        
    Returns:
        str: Recommended crop
    """
    try:
        # Load crop recommendation dataset
        csv_path = './Crop_recommendation.csv'
        if not os.path.exists(csv_path):
            logger.error(f"Crop recommendation CSV not found at {csv_path}")
            return "Data not available"
        
        crop_data = pd.read_csv(csv_path)
        
        # Filter suitable crops based on parameters
        filtered_crops = crop_data[
            (crop_data['N'] <= n * 1.2) & (crop_data['N'] >= n * 0.8) &
            (crop_data['P'] <= p * 1.2) & (crop_data['P'] >= p * 0.8) &
            (crop_data['K'] <= k * 1.2) & (crop_data['K'] >= k * 0.8) &
            (crop_data['ph'] <= ph * 1.2) & (crop_data['ph'] >= ph * 0.8) &
            (crop_data['temperature'] <= temperature * 1.2) & (crop_data['temperature'] >= temperature * 0.8) &
            (crop_data['humidity'] <= humidity * 1.2) & (crop_data['humidity'] >= humidity * 0.8) &
            (crop_data['rainfall'] <= rainfall * 1.2) & (crop_data['rainfall'] >= rainfall * 0.8)
        ]
        
        # If no exact match, widen the search
        if len(filtered_crops) == 0:
            filtered_crops = crop_data[
                (crop_data['N'] <= n * 1.5) & (crop_data['N'] >= n * 0.5) &
                (crop_data['P'] <= p * 1.5) & (crop_data['P'] >= p * 0.5) &
                (crop_data['K'] <= k * 1.5) & (crop_data['K'] >= k * 0.5) &
                (crop_data['ph'] <= ph * 1.5) & (crop_data['ph'] >= ph * 0.5) &
                (crop_data['temperature'] <= temperature * 1.5) & (crop_data['temperature'] >= temperature * 0.5) &
                (crop_data['humidity'] <= humidity * 1.5) & (crop_data['humidity'] >= humidity * 0.5) &
                (crop_data['rainfall'] <= rainfall * 1.5) & (crop_data['rainfall'] >= rainfall * 0.5)
            ]
        
        # Soil-specific crops
        soil_specific_crops = {
            'Red': ['cotton', 'wheat', 'rice', 'pulses', 'millets'],
            'Black': ['cotton', 'sugarcane', 'vegetables', 'citrus fruits'],
            'Clay': ['rice', 'lettuce', 'chard', 'broccoli', 'cabbage'],
            'Alluvial': ['rice', 'wheat', 'sugarcane', 'maize', 'pulses']
        }
        
        # If still no match, recommend based on soil type
        if len(filtered_crops) == 0:
            if soil_type in soil_specific_crops:
                return soil_specific_crops[soil_type][0]
            return "No specific crop recommendation available"
            
        # Get most suitable crop from filtered list
        # Preference to crops that match soil type if available
        soil_matching_crops = [crop for crop in filtered_crops['label'] if crop.lower() in soil_specific_crops.get(soil_type, [])]
        
        if soil_matching_crops:
            return soil_matching_crops[0]
        
        # Return the first match if no soil-specific match
        return filtered_crops.iloc[0]['label']
        
    except Exception as e:
        logger.error(f"Error in crop recommendation: {str(e)}")
        return "Error in processing crop recommendation"

def analyze_parameters(n, p, k, ph, temperature, humidity, rainfall):
    """
    Analyze soil and environmental parameters and provide assessment
    
    Args:
        n (float): Nitrogen value
        p (float): Phosphorus value
        k (float): Potassium value
        ph (float): pH value
        temperature (float): Temperature value
        humidity (float): Humidity value
        rainfall (float): Rainfall value
        
    Returns:
        dict: Analysis of each parameter
    """
    analysis = {}
    
    # Nitrogen analysis
    if n < 40:
        analysis['N'] = "Low (Consider nitrogen fertilizers)"
    elif n < 80:
        analysis['N'] = "Medium"
    else:
        analysis['N'] = "Optimal"
    
    # Phosphorus analysis
    if p < 30:
        analysis['P'] = "Low (Consider phosphate fertilizers)"
    elif p < 60:
        analysis['P'] = "Medium"
    else:
        analysis['P'] = "Optimal"
    
    # Potassium analysis
    if k < 30:
        analysis['K'] = "Low (Consider potash fertilizers)"
    elif k < 60:
        analysis['K'] = "Medium"
    else:
        analysis['K'] = "Optimal"
    
    # pH analysis
    if ph < 5.5:
        analysis['pH'] = "Acidic (Consider lime to raise pH)"
    elif ph < 6.5:
        analysis['pH'] = "Slightly Acidic"
    elif ph < 7.5:
        analysis['pH'] = "Neutral (Optimal)"
    elif ph < 8.5:
        analysis['pH'] = "Slightly Alkaline"
    else:
        analysis['pH'] = "Alkaline (Consider sulfur to lower pH)"
    
    # Temperature analysis
    if temperature < 15:
        analysis['temperature'] = "Cold (Suitable for cold-season crops)"
    elif temperature < 25:
        analysis['temperature'] = "Moderate (Optimal for most crops)"
    else:
        analysis['temperature'] = "Hot (Suitable for heat-tolerant crops)"
    
    # Humidity analysis
    if humidity < 40:
        analysis['humidity'] = "Low (Consider irrigation or humidity management)"
    elif humidity < 70:
        analysis['humidity'] = "Moderate (Optimal for most crops)"
    else:
        analysis['humidity'] = "High (Watch for fungal diseases)"
    
    # Rainfall analysis
    if rainfall < 80:
        analysis['rainfall'] = "Low (Irrigation necessary)"
    elif rainfall < 200:
        analysis['rainfall'] = "Moderate (Optimal for many crops)"
    else:
        analysis['rainfall'] = "High (Good drainage required)"
    
    return analysis
