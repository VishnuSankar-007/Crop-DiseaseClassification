from flask import Flask, request, render_template, jsonify
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (256 // 8) * (256 // 8), 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class_names = ['Apple Apple scab', 'Apple Black rot',
               'Apple Cedar apple rust', 'Apple healthy',
               'Blueberry healthy', 'Cherry (including sour)Powdery mildew',
               'Cherry (including sour) healthy', 'Corn (maize) Cercospora leaf spot Gray leaf spot',
               'Corn_(maize) Common rust ', 'Corn (maize) Northern Leaf Blight',
               'Corn (maize) healthy', 'Grape Black rot', 'Grape Esca (Black Measles)',
               'Grape Leaf blight (Isariopsis Leaf Spot)', 'Grape healthy',
               'Orange Haunglongbing (Citrus greening)', 'Peach Bacterial spot',
               'Peach healthy', 'Pepper bell Bacterial spot', 'Pepper bell healthy',
               'Potato Early blight', 'Potato Late blight', 'Potato healthy', 'Raspberry healthy',
               'Soybean healthy', 'Squash Powdery mildew', 'Strawberry Leaf scorch',
               'Strawberry healthy', 'Tomato Bacterial spot', 'Tomato Early blight',
               'Tomato Late blight', 'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
               'Tomato Spider mites Two-spotted spider mite', 'Tomato Target Spot',
               'Tomato Tomato Yellow Leaf Curl Virus', 'Tomato Tomato mosaic virus',
               'Tomato healthy']
num_classes = len(class_names)

model = PlantDiseaseModel(num_classes=num_classes).to(device)
model.load_state_dict(torch.load("models/bigtorchLatestModel.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            class_name = class_names[predicted.item()]

        return jsonify({'prediction': class_name})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
