from flask import Flask, render_template, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
import os
import datetime
from werkzeug.utils import secure_filename
import cv2  # For image processing (optional)
import numpy as np
import tensorflow as tf  # Assuming TensorFlow for model inference

# Initialize Flask app and database
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to save uploaded images
db = SQLAlchemy(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Database model for patient details
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    dob = db.Column(db.Date, nullable=False)
    gender = db.Column(db.String(10), nullable=False)
    symptoms = db.Column(db.Text, nullable=True)
    medical_history = db.Column(db.Text, nullable=True)
    ct_scan_path = db.Column(db.String(200), nullable=True)  # Path to uploaded image
    report = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.datetime.utcnow)

# Function to check allowed file types
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_ct_scan', methods=['POST'])
def upload_ct_scan():
    # Get form data
    name = request.form['name']
    dob = datetime.datetime.strptime(request.form['dob'], '%Y-%m-%d')
    gender = request.form['gender']
    symptoms = request.form['symptoms']
    medical_history = request.form['medical_history']
    file = request.files['ct_scan']

    # Check if file is valid
    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Analyze the image (Placeholder for model inference)
        analysis_result = analyze_ct_scan(file_path)

        # Save patient details in the database
        patient = Patient(
            name=name,
            dob=dob,
            gender=gender,
            symptoms=symptoms,
            medical_history=medical_history,
            ct_scan_path=file_path,
            report=analysis_result
        )
        db.session.add(patient)
        db.session.commit()

        # Return the result to the user
        return render_template('result.html', name=name, message="Analysis Complete", report=analysis_result)

    else:
        return "Invalid file type. Please upload a valid image file.", 400

# Function to analyze CT scan image
def analyze_ct_scan(file_path):
    # Load your pre-trained model (e.g., TensorFlow or PyTorch model)
    # Replace 'model.h5' with the path to your model file
    model = tf.keras.models.load_model('efficientnetb7_model.h5')

    # Preprocess the image for the model
    image = cv2.imread(file_path)  # Load image using OpenCV
    image = cv2.resize(image, (224, 224))  # Resize to model's input size
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image / 255.0  # Normalize pixel values

    # Predict using the model
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)  # Get the class with highest probability

    # Map class indices to labels (adjust this to match your model)
    labels = {0: "Normal", 1: "Aneurysm Stage 1", 2: "Aneurysm Stage 2", 3: "Aneurysm Stage 3", 4: "Aneurysm Stage 4"}
    result = labels.get(predicted_class, "Unknown")

    return result

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.mkdir('uploads')
    with app.app_context():
        db.create_all()
    app.run(debug=True)
