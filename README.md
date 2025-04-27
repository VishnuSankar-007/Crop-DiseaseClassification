🌾 Crop Disease Classification
This project focuses on detecting crop diseases using deep learning. The goal is to build a reliable image classification model that can identify different crop diseases, helping farmers take timely action and improve crop yields.

🚀 Project Overview
Model Framework: PyTorch

Deployment: Flask Web Application

Frontend: HTML/CSS

Problem: Multi-class classification of crop diseases from leaf images.

📂 Project Structure
graphql
Copy
Edit
Crop-Disease-Classification/
│
├── static/               # Static files (CSS, images)
├── templates/            # HTML templates
├── model/                # Trained PyTorch model files
├── app.py                # Flask backend
├── model_training.ipynb  # Jupyter Notebook for model development
├── requirements.txt      # Required Python packages
└── README.md             # Project documentation
🛠️ Features
Image Upload: Upload an image of a crop leaf.

Disease Prediction: The model predicts the type of disease.

Confidence Score: Displays the prediction probability.

User-friendly Interface: Simple and clean design.

🧠 Model Details
Architecture: Transfer Learning using CNN

Dataset: PlantVillage 



Training Steps
Data Augmentation applied to prevent overfitting.

Model trained using Cross-Entropy Loss and Adam optimizer.

Early Stopping used to prevent overtraining.




📚 Future Improvements
Add more crop types and diseases.

Deploy on cloud (AWS, Heroku, etc.)

Add mobile app support.


