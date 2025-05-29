# Traffic Sign Recognition with Deep Learning

This project implements a deep learning pipeline for classifying German traffic signs using PyTorch. It features two custom convolutional neural network architectures (TrafficSignCNN and TinyVGG3), comprehensive data visualization, model evaluation, and a web-based demo for real-time predictions.

## Features
- **Custom Dataset Loader**: Handles image cropping, normalization, and label extraction from CSV files.
- **Two Deep Learning Models**: 
  - `TrafficSignCNN`: A custom CNN architecture.
  - `TinyVGG3`: A VGG-inspired compact model.
- **Training & Evaluation**: Includes training loops, accuracy/loss tracking, confusion matrix, and classification report plotting.
- **Visualization Tools**: Scripts to visualize class distribution and sample images from each class.
- **Web Demo**: Flask-based web app for uploading images and getting real-time predictions from both models.

## Project Structure
```
main.py                  # Main training and evaluation script
model.py                 # Model architectures
visualization.py         # Plotting and evaluation utilities
plot new.py              # Class distribution visualization
plot2.py, plot3.py       # Example image visualizations
prediction_demo_web.py.py# Flask web demo for predictions
requirements.txt         # Python dependencies
Data/                    # Dataset (CSV files and images)
plots/                   # Output plots
```

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Train models**:
   ```bash
   python main.py
   ```
   - This will train both models, save their weights, and generate evaluation plots in the `plots/` folder.
3. **Run the web demo**:
   ```bash
   python prediction_demo_web.py.py
   ```
   - Open your browser at [http://127.0.0.1:5000](http://127.0.0.1:5000) to test predictions interactively.

## Results
- Training and evaluation metrics are saved as plots in the `plots/` directory.
- The web demo allows you to upload any traffic sign image and see predictions from both models.

## Contributors
- Abdullah
- Wahaj

---
Feel free to fork, use, and improve this project!

