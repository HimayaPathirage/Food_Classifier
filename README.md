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

Property            Value

Architecture        CNN (custom, trained on food)

Input Size          160 × 160 px

Normalization       ImageNet mean/std

Output              Classes5


🚀 Running the Application Locally

This application currently runs locally and has not been deployed online. The implementation instructions are below.

Please follow the steps below to run the application on a local machine.

### Step 1: Download the Project

Clone the repository or download the ZIP file and extract it.

```bash
git clone https://github.com/HimayaPathirage/Food_Classifier.git
```

### Step 2: Open the Project Folder

Navigate to the project directory.

```bash
cd Food_Classifier
```

### Step 3: Install Required Libraries

Install all required Python dependencies.

```bash
pip install -r requirements.txt
```

### Step 4: Verify Required Files

Ensure the following files are present in the project root directory:

- `app.py`
- `best_food_model_v2.h5`
- `custom_nutrition_data.csv`

Also ensure the folders below exist:

- `templates/`
- `static/`

### Step 5: Run the Application

```bash
python app.py
```

### Step 6: Open the Application

After running the command, the terminal should display:

```text
Running on http://127.0.0.1:5000
```

Open a web browser and go to:

```text
http://127.0.0.1:5000
```

The Food Intelligence application interface should now be available for use.


