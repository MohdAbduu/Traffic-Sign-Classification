import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model import TrafficSignCNN, TinyVGG3
from flask import Flask, request, render_template, jsonify
import base64
from io import BytesIO

# Create Flask app
app = Flask(__name__)

# Classes for traffic signs
CLASSES = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road',
    13: 'Yield', 14: 'Stop', 15: 'No vehicles',
    16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry', 18: 'General caution',
    19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road',
    24: 'Road narrows on the right', 25: 'Road work', 26: 'Traffic signals',
    27: 'Pedestrians', 28: 'Children crossing', 29: 'Bicycles crossing',
    30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead',
    34: 'Turn left ahead', 35: 'Ahead only', 36: 'Go straight or right',
    37: 'Go straight or left', 38: 'Keep right', 39: 'Keep left',
    40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load models
def load_models():
    num_classes = 43  # For German Traffic Sign Dataset

    # Load TrafficSignCNN
    cnn_model = TrafficSignCNN(num_classes)
    cnn_model.load_state_dict(torch.load('trafficsigncnn.pth', map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    # Load TinyVGG3
    vgg_model = TinyVGG3(num_classes)
    vgg_model.load_state_dict(torch.load('tinyvgg3.pth', map_location=device))
    vgg_model.to(device)
    vgg_model.eval()

    return cnn_model, vgg_model


# Transformations for input images
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# Load models
cnn_model, vgg_model = load_models()

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Write HTML template
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Traffic Sign Prediction Demo</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            text-align: center;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .model-selection {
            margin-bottom: 20px;
        }
        .upload-section {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-bottom: 20px;
        }
        .image-preview {
            margin: 20px 0;
            text-align: center;
        }
        #imagePreview {
            max-width: 300px;
            max-height: 300px;
            border: 1px solid #ddd;
            border-radius: 4px;
            display: none;
        }
        .result {
            margin-top: 20px;
            padding: 10px;
            border-radius: 4px;
            text-align: center;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            padding: 10px 15px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background-color: #45a049;
        }
        select {
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .prediction {
            font-weight: bold;
            font-size: 18px;
            color: #333;
        }
        .confidence {
            color: #666;
            font-size: 16px;
        }
    </style>
</head>
<body>
    <h1>Traffic Sign Prediction Demo</h1>
    <div class="container">
        <div class="model-selection">
            <label for="model">Select Model:</label>
            <select id="model" name="model">
                <option value="cnn">TrafficSignCNN</option>
                <option value="vgg">TinyVGG3</option>
            </select>
        </div>

        <div class="upload-section">
            <input type="file" id="imageUpload" accept="image/*" style="display:none">
            <button class="btn" id="uploadBtn">Upload Traffic Sign Image</button>
        </div>

        <div class="image-preview">
            <img id="imagePreview" alt="Image Preview">
        </div>

        <div class="result" id="result">
            <p>Upload an image to see the prediction</p>
        </div>
    </div>

    <script>
        // Handle file upload
        const uploadBtn = document.getElementById('uploadBtn');
        const imageUpload = document.getElementById('imageUpload');
        const imagePreview = document.getElementById('imagePreview');
        const resultDiv = document.getElementById('result');
        const modelSelect = document.getElementById('model');

        uploadBtn.addEventListener('click', () => {
            imageUpload.click();
        });

        imageUpload.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const reader = new FileReader();

                reader.onload = function(e) {
                    // Display image preview
                    imagePreview.src = e.target.result;
                    imagePreview.style.display = 'block';

                    // Get selected model
                    const selectedModel = modelSelect.value;

                    // Make prediction request
                    fetch('/predict', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            image: e.target.result,
                            model: selectedModel
                        })
                    })
                    .then(response => response.json())
                    .then(data => {
                        resultDiv.innerHTML = `
                            <p class="prediction">Prediction: ${data.class_name}</p>
                            <p class="confidence">Confidence: ${data.confidence.toFixed(2)}%</p>
                        `;
                    })
                    .catch(error => {
                        console.error('Error:', error);
                        resultDiv.innerHTML = '<p>Error making prediction. Please try again.</p>';
                    });
                };

                reader.readAsDataURL(file);
            }
        });
    </script>
</body>
</html>
    ''')


# Prediction function
def predict_image(image, model_type='cnn'):
    # Convert to RGB if not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Apply transformations
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Select model
    model = cnn_model if model_type == 'cnn' else vgg_model

    # Make prediction
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        _, predicted = torch.max(outputs, 1)
        predicted_class = predicted.item()
        confidence = probabilities[predicted_class].item() * 100

    return {
        'class_id': predicted_class,
        'class_name': CLASSES.get(predicted_class, f"Class {predicted_class}"),
        'confidence': confidence
    }


# Routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.get_json()
        image_data = data['image']
        model_type = data['model']

        # Decode base64 image
        image_data = image_data.split(',')[1]
        image = Image.open(BytesIO(base64.b64decode(image_data)))

        # Make prediction
        result = predict_image(image, model_type)

        return jsonify(result)


if __name__ == '__main__':
    print("Starting Traffic Sign Prediction Demo server...")
    print("Open your web browser and navigate to http://127.0.0.1:5000")
    app.run(debug=True)