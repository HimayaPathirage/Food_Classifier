import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --- ADD THIS FIX HERE ---
from tensorflow.keras.layers import Dense

# This class "fixes" the loading error by removing the argument Keras doesn't like
class FixedDense(Dense):
    @classmethod
    def from_config(cls, config):
        config.pop('quantization_config', None)
        return super().from_config(config)

# Register the fixed layer so the model uses it during load_model()
tf.keras.utils.get_custom_objects()['Dense'] = FixedDense
# --------------------------

app = Flask(__name__)

# ---------------- CONFIGURATION ----------------
# Ensure 'final_food_model_v2.h5' and 'custom_nutrition_data.csv' are in this folder
MODEL_PATH = 'best_food_model_v2.h5'
CSV_PATH = 'custom_nutrition_data.csv'
UPLOAD_FOLDER = 'static/uploads'

# Load model once when app starts
model = load_model(MODEL_PATH)

# Load nutrition data once when app starts
nutrition_df = pd.read_csv(CSV_PATH)
nutrition_df['Food_Item'] = nutrition_df['Food_Item'].str.strip().str.lower()

# EXACT class order from your training folders (Alphabetical)
CLASS_NAMES = ['egg', 'meat', 'noodles-pasta', 'rice', 'vegetable-fruit']

# Normalization values (Standard for CNNs like yours)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---------------- ROUTES ----------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    if file:
        # Save uploaded image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 1. PREPROCESSING (Matches Final_code.py)
        # Target size is 160 based on your training script
        img = image.load_img(filepath, target_size=(160, 160)) 
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array.astype('float32') / 255.0 # Rescale to [0,1]
        
        # 2. STANDARDIZATION
        img_array = (img_array - np.array(MEAN)) / (np.array(STD) + 1e-7)

        # 3. AI PREDICTION
        predictions = model.predict(img_array)
        score_index = np.argmax(predictions[0])
        detected_food = CLASS_NAMES[score_index]

        # 4. NUTRITION LOOKUP FROM CSV
        match = nutrition_df[nutrition_df['Food_Item'] == detected_food]
        if not match.empty:
            info = match.iloc[0][['Calories', 'Carbs', 'Protein', 'Fat']].to_dict()
        else:
            info = {"Calories": 0, "Carbs": 0, "Protein": 0, "Fat": 0}

        return render_template('index.html', 
                               prediction=detected_food, 
                               nutrition=info, 
                               user_image=filename)

if __name__ == '__main__':
    app.run(debug=True)