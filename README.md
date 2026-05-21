🍽️ Food Intelligence System
An AI-powered web application that classifies food from images and provides nutritional information using a trained deep learning model.

📸 Overview
Upload a photo of your meal and the system will:

Identify the food category using a CNN model
Display nutritional info (Calories, Protein, Carbs, Fat) from a local CSV database

⚙️ Requirements

Python 3.8+
TensorFlow 2.x
Flask
NumPy
Pandas
Werkzeug

🧠 Model Details
Property        Value
Architecture    CNN (custom, trained on food)
Input Size      160 × 160 px
Normalization   ImageNet mean/std
Output          Classes5

🚀 Running the App

1.Clone or download this repository.
2.Place the required files in the project root:
  best_food_model_v2.h5 — the trained model
  custom_nutrition_data.csv — nutrition data
3.Start the Flask server:
  bash
  python app.py

4. Open your browser and go to: http://127.0.0.1:5000
