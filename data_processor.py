import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

class CropRecommender:
    """Class for processing crop recommendation data"""
    
    def __init__(self, csv_path='../backend/Crop_recommendation.csv'):
        self.csv_path = csv_path
        self.crop_data = None
        self.load_data()
        
    def load_data(self):
        """Load the crop recommendation dataset"""
        try:
            if os.path.exists(self.csv_path):
                self.crop_data = pd.read_csv(self.csv_path)
                logging.info(f"Loaded crop data with {len(self.crop_data)} entries")
                logging.info(f"Unique crops: {self.crop_data['label'].unique()}")
            else:
                logging.error(f"Crop recommendation file not found at {self.csv_path}")
        except Exception as e:
            logging.error(f"Error loading crop data: {e}")
    
    def get_soil_compatible_crops(self, soil_type):
        """Get crops that are compatible with a specific soil type"""
        # This is a simplified mapping based on general agricultural knowledge
        # In a real system, this would be based on more detailed data
        soil_crop_compatibility = {
            'Red': ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas', 'mothbeans', 'blackgram', 'mungbean'],
            'Black': ['cotton', 'maize', 'wheat', 'jute', 'coffee'],
            'Clay': ['rice', 'watermelon', 'muskmelon', 'papaya', 'coconut'],
            'Alluvial': ['rice', 'wheat', 'maize', 'sugarcane', 'banana', 'mango', 'grapes']
        }
        
        return soil_crop_compatibility.get(soil_type, self.crop_data['label'].unique())
    
    def recommend_crops(self, soil_type, N, P, K, pH, temperature, humidity, rainfall):
        """
        Recommend suitable crops based on soil type and parameters
        
        Args:
            soil_type: Type of soil (Red, Black, Clay, Alluvial)
            N: Nitrogen content (in kg/ha)
            P: Phosphorus content (in kg/ha)
            K: Potassium content (in kg/ha)
            pH: pH value of soil
            temperature: Temperature in Celsius
            humidity: Relative humidity in percentage
            rainfall: Annual rainfall in mm
            
        Returns:
            List of recommended crops
        """
        if self.crop_data is None:
            logging.error("Crop data not loaded")
            return []
        
        # Get crops compatible with the soil type
        compatible_crops = self.get_soil_compatible_crops(soil_type)
        
        # Calculate suitability scores for each crop
        scores = {}
        for crop_name in compatible_crops:
            crop_rows = self.crop_data[self.crop_data['label'] == crop_name.lower()]
            
            if len(crop_rows) == 0:
                continue
                
            # Calculate average ideal values for this crop
            ideal_N = crop_rows['N'].mean()
            ideal_P = crop_rows['P'].mean()
            ideal_K = crop_rows['K'].mean()
            ideal_pH = crop_rows['ph'].mean()
            ideal_temp = crop_rows['temperature'].mean()
            ideal_humidity = crop_rows['humidity'].mean()
            ideal_rainfall = crop_rows['rainfall'].mean()
            
            # Calculate weighted Euclidean distance (simplified approach)
            distance = (
                0.15 * ((N - ideal_N) / max(1, ideal_N)) ** 2 +
                0.15 * ((P - ideal_P) / max(1, ideal_P)) ** 2 +
                0.15 * ((K - ideal_K) / max(1, ideal_K)) ** 2 +
                0.15 * ((pH - ideal_pH) / max(1, ideal_pH)) ** 2 +
                0.15 * ((temperature - ideal_temp) / max(1, ideal_temp)) ** 2 +
                0.15 * ((humidity - ideal_humidity) / max(1, ideal_humidity)) ** 2 +
                0.10 * ((rainfall - ideal_rainfall) / max(1, ideal_rainfall)) ** 2
            )
            
            scores[crop_name] = distance
        
        # Sort by score (lower is better)
        if not scores:
            return []
        
        # Get top 3 recommendations
        sorted_crops = sorted(scores.items(), key=lambda x: x[1])
        return [crop[0].capitalize() for crop in sorted_crops[:3]]
    
    def analyze_parameter(self, param_name, value):
        """
        Analyze a specific parameter and provide recommendations
        
        Args:
            param_name: Name of the parameter (N, P, K, pH, etc.)
            value: Value of the parameter
            
        Returns:
            Dictionary with status and recommendation
        """
        analysis = {}
        
        if param_name == "N":
            if value < 20:
                analysis["status"] = "Low"
                analysis["recommendation"] = "Increase nitrogen fertilization"
            elif value < 40:
                analysis["status"] = "Medium"
                analysis["recommendation"] = "Moderate nitrogen fertilization recommended"
            else:
                analysis["status"] = "Optimal"
                analysis["recommendation"] = "Maintain current nitrogen levels"
        
        elif param_name == "P":
            if value < 10:
                analysis["status"] = "Low"
                analysis["recommendation"] = "Increase phosphorus application"
            elif value < 30:
                analysis["status"] = "Medium"
                analysis["recommendation"] = "Moderate phosphorus application recommended"
            else:
                analysis["status"] = "Optimal"
                analysis["recommendation"] = "Maintain current phosphorus levels"
        
        elif param_name == "K":
            if value < 20:
                analysis["status"] = "Low"
                analysis["recommendation"] = "Increase potassium application"
            elif value < 40:
                analysis["status"] = "Medium"
                analysis["recommendation"] = "Moderate potassium application recommended"
            else:
                analysis["status"] = "Optimal"
                analysis["recommendation"] = "Maintain current potassium levels"
        
        elif param_name == "pH":
            if value < 5.5:
                analysis["status"] = "Acidic"
                analysis["recommendation"] = "Consider adding lime to raise pH"
            elif value < 7.5:
                analysis["status"] = "Neutral"
                analysis["recommendation"] = "Optimal pH for most crops"
            else:
                analysis["status"] = "Alkaline"
                analysis["recommendation"] = "Consider adding sulfur to lower pH"
        
        elif param_name == "temperature":
            if value < 15:
                analysis["status"] = "Cool"
                analysis["recommendation"] = "Consider cold-weather crops"
            elif value < 30:
                analysis["status"] = "Moderate"
                analysis["recommendation"] = "Optimal temperature for most crops"
            else:
                analysis["status"] = "Hot"
                analysis["recommendation"] = "Ensure adequate irrigation, consider heat-tolerant crops"
        
        elif param_name == "humidity":
            if value < 30:
                analysis["status"] = "Dry"
                analysis["recommendation"] = "Increase irrigation frequency"
            elif value < 70:
                analysis["status"] = "Moderate"
                analysis["recommendation"] = "Optimal humidity for most crops"
            else:
                analysis["status"] = "Humid"
                analysis["recommendation"] = "Monitor for fungal diseases"
        
        elif param_name == "rainfall":
            if value < 50:
                analysis["status"] = "Low"
                analysis["recommendation"] = "Implement irrigation system"
            elif value < 150:
                analysis["status"] = "Moderate"
                analysis["recommendation"] = "Supplemental irrigation may be needed"
            else:
                analysis["status"] = "High"
                analysis["recommendation"] = "Ensure proper drainage"
        
        return analysis
