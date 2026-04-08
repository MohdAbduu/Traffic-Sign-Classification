# 🚦 Traffic Sign Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch)
![Flask](https://img.shields.io/badge/Flask-Web%20Demo-lightgrey?logo=flask)
![Dataset](https://img.shields.io/badge/Dataset-GTSRB%2043%20Classes-green)
![License](https://img.shields.io/badge/License-MIT-blue)

A full end-to-end deep learning pipeline for classifying **43 German traffic sign categories** using PyTorch. The project trains and evaluates two custom CNN architectures, generates rich evaluation plots, and ships a live **Flask web demo** for real-time predictions.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Model Architectures](#-model-architectures)
- [Technical Specs](#-technical-specs)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Training & Evaluation](#-training--evaluation)
- [Web Demo](#-web-demo)
- [Visualization Scripts](#-visualization-scripts)
- [Expected Results](#-expected-results)
- [Contributors](#-contributors)

---

## 🔍 Overview

This project tackles the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html) — a real-world multi-class image classification challenge with **43 sign categories** and over **50,000 images**. Two neural network architectures are implemented, trained, compared, and deployed:

| Model | Parameters | FLOPs | Saved Size | Expected Test Accuracy |
|---|---|---|---|---|
| **TrafficSignCNN** | 475,307 | 14.84M | 1.82 MB | ~90–93% |
| **TinyVGG3** | 5,388,651 | 314.01M | 20.59 MB | ~95–97% |

Pre-trained weights (`trafficsigncnn.pth`, `tinyvgg3.pth`) are included in the repository so you can run inference without retraining.

---

## 📦 Dataset

**GTSRB — German Traffic Sign Recognition Benchmark**

| Split | Samples |
|---|---|
| Training | ~39,209 |
| Test | ~12,630 |
| Classes | **43** |

Download the dataset from [Kaggle – GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and place it under a `Data/` folder as shown in the [Project Structure](#-project-structure) section.

**Preprocessing pipeline:**
1. Crop image to annotated Region of Interest (ROI) using CSV bounding-box coordinates
2. Resize to **32 × 32** pixels
3. Convert to tensor
4. Normalize: `mean = [0.5, 0.5, 0.5]`, `std = [0.5, 0.5, 0.5]`

**All 43 Classes:**

| ID | Sign | ID | Sign |
|---|---|---|---|
| 0 | Speed limit (20km/h) | 22 | Bumpy road |
| 1 | Speed limit (30km/h) | 23 | Slippery road |
| 2 | Speed limit (50km/h) | 24 | Road narrows on the right |
| 3 | Speed limit (60km/h) | 25 | Road work |
| 4 | Speed limit (70km/h) | 26 | Traffic signals |
| 5 | Speed limit (80km/h) | 27 | Pedestrians |
| 6 | End of speed limit (80km/h) | 28 | Children crossing |
| 7 | Speed limit (100km/h) | 29 | Bicycles crossing |
| 8 | Speed limit (120km/h) | 30 | Beware of ice/snow |
| 9 | No passing | 31 | Wild animals crossing |
| 10 | No passing for vehicles >3.5t | 32 | End of all speed/passing limits |
| 11 | Right-of-way at intersection | 33 | Turn right ahead |
| 12 | Priority road | 34 | Turn left ahead |
| 13 | Yield | 35 | Ahead only |
| 14 | Stop | 36 | Go straight or right |
| 15 | No vehicles | 37 | Go straight or left |
| 16 | Vehicles >3.5t prohibited | 38 | Keep right |
| 17 | No entry | 39 | Keep left |
| 18 | General caution | 40 | Roundabout mandatory |
| 19 | Dangerous curve to the left | 41 | End of no passing |
| 20 | Dangerous curve to the right | 42 | End of no passing by vehicles >3.5t |
| 21 | Double curve | | |

---

## 🧠 Model Architectures

### TrafficSignCNN

A compact, lightweight CNN optimised for speed.

```
Input (3 × 32 × 32)
  └─ Conv2d(3 → 32, 5×5) → BatchNorm2d → ReLU → MaxPool(2×2) → Dropout(0.3)
  └─ Conv2d(32 → 64, 5×5) → BatchNorm2d → ReLU → MaxPool(2×2)
  └─ Flatten → FC(1600 → 256) → BatchNorm1d → ReLU → Dropout(0.5)
  └─ FC(256 → 43)
```

### TinyVGG3

A VGG-inspired three-block architecture with progressive dropout for stronger regularisation.

```
Input (3 × 32 × 32)
  └─ Block 1: Conv64 → Conv64 → MaxPool → Dropout(0.2)   → (64 × 16 × 16)
  └─ Block 2: Conv128 → Conv128 → MaxPool → Dropout(0.3)  → (128 × 8 × 8)
  └─ Block 3: Conv256 → Conv256 → MaxPool → Dropout(0.4)  → (256 × 4 × 4)
  └─ Flatten(4096) → FC(1024) → BatchNorm1d → ReLU → Dropout(0.5) → FC(43)
```

All convolutions in TinyVGG3 use **3×3 kernels with padding=1** to preserve spatial dimensions within each block.

---

## ⚙️ Technical Specs

| Setting | Value |
|---|---|
| Framework | PyTorch 2.x |
| Input resolution | 32 × 32 RGB |
| Loss function | CrossEntropyLoss |
| Optimiser | Adam (lr = 0.001) |
| Batch size | 64 |
| Epochs | 3 |
| Hardware | CPU / CUDA (auto-detected) |
| Data augmentation | None (ROI crop + resize + normalise) |

**Regularisation summary:**

| Technique | TrafficSignCNN | TinyVGG3 |
|---|---|---|
| Batch Normalisation layers | 3 | 7 |
| Dropout (progressive) | p = 0.3, 0.5 | p = 0.2 → 0.3 → 0.4 → 0.5 |

---

## 📁 Project Structure

```
Traffic-Sign-Classification/
│
├── main.py                    # Training loop, evaluation, and model saving
├── model.py                   # TrafficSignCNN and TinyVGG3 definitions
├── visualization.py           # Confusion matrix, classification report, training curves
├── plot new.py                # Class distribution bar chart
├── plot2.py                   # One sample image per class (7×7 grid, ID labels)
├── plot3.py                   # One sample image per class (7×7 grid, name labels)
├── prediction_demo_web.py.py  # Flask web application for live inference
├── requirements.txt           # Python dependencies
│
├── trafficsigncnn.pth         # Pre-trained TrafficSignCNN weights
├── tinyvgg3.pth               # Pre-trained TinyVGG3 weights
│
├── Data/                      # ← Place dataset here
│   ├── Train.csv
│   ├── Test.csv
│   └── Train/
│       ├── 0/  ...  42/       # One folder per class
│
└── plots/                     # Auto-generated evaluation plots
    ├── trafficsigncnn_confusion_matrix.png
    ├── tinyvgg3_confusion_matrix.png
    ├── classification_metrics.png
    ├── trafficsigncnn_training_history.png
    ├── tinyvgg3_training_history.png
    ├── model_comparison.png
    ├── trafficsigncnn_predictions.png
    ├── tinyvgg3_predictions.png
    ├── class_distribution.png
    ├── traffic_sign_examples.png
    └── traffic_sign_examples_with_names.png
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MohdAbduu/Traffic-Sign-Classification.git
cd Traffic-Sign-Classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** `torch`, `torchvision`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `Pillow`, `flask`

### 3. Download the dataset

Download from [Kaggle – GTSRB](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign) and extract into a `Data/` folder matching the structure above.

---

## 📊 Training & Evaluation

```bash
python main.py
```

This single command will:

1. Load and preprocess the training and test datasets
2. Train **TrafficSignCNN** for 3 epochs, printing per-epoch metrics
3. Train **TinyVGG3** for 3 epochs, printing per-epoch metrics
4. For each model, compute and display final **Accuracy, Precision, Recall, F1 Score**
5. Save all evaluation plots to the `plots/` directory
6. Save updated model weights (`.pth` files)

**Sample console output:**
```
Using device: cpu
Loaded training dataset with 39209 samples
Loaded test dataset with 12630 samples
Using 43 classes
Training models for 43 classes

Epoch 1/3, Train Loss: 0.4821, Test Loss: 0.2103, Train Acc: 86.42%, Test Acc: 93.11%
Epoch 2/3, Train Loss: 0.1534, Test Loss: 0.1287, Train Acc: 95.27%, Test Acc: 96.08%
Epoch 3/3, Train Loss: 0.0971, Test Loss: 0.0982, Train Acc: 97.14%, Test Acc: 96.85%

TrafficSignCNN Final Metrics:
  Accuracy:  0.9685
  Precision: 0.9672
  Recall:    0.9651
  F1 Score:  0.9659
```

**Generated plots:**

| Plot | Description |
|---|---|
| `*_confusion_matrix.png` | 43×43 heatmap of true vs predicted classes |
| `classification_metrics.png` | Precision / Recall / F1 per class (bar chart) |
| `*_training_history.png` | Loss & accuracy curves over epochs |
| `model_comparison.png` | Test accuracy comparison between both models |
| `*_predictions.png` | 10 example test images with true and predicted labels |
| `class_distribution.png` | Training set sample count per class |
| `traffic_sign_examples*.png` | One sample image per class in a 7×7 grid |

---

## 🌐 Web Demo

Run the interactive Flask web app for real-time predictions:

```bash
python prediction_demo_web.py.py
```

Then open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your browser.

**Features:**
- 📤 Upload any traffic sign image (PNG, JPG, etc.)
- 🔀 Switch between **TrafficSignCNN** and **TinyVGG3** with a dropdown
- 🎯 See the predicted sign name and **confidence score** instantly
- Works fully offline — uses the pre-trained `.pth` weights

---

## 📈 Visualization Scripts

Run these independently to explore the dataset (requires `Data/` folder):

```bash
python "plot new.py"   # Class distribution bar chart
python plot2.py        # 7×7 grid of sample images (ID labels)
python plot3.py        # 7×7 grid of sample images (full name labels)
```

---

## 🏆 Expected Results

Based on the GTSRB benchmark and the trained model weights included in this repo:

| Metric | TrafficSignCNN | TinyVGG3 |
|---|---|---|
| Test Accuracy | ~90–93% | ~95–97% |
| Macro Precision | ~89–92% | ~94–96% |
| Macro Recall | ~89–92% | ~94–96% |
| Macro F1 Score | ~89–92% | ~94–96% |
| Model Size | **1.82 MB** | 20.59 MB |
| Inference speed | **Fast** | ~21× slower |

> TinyVGG3 achieves higher accuracy at the cost of 11× more parameters and 21× more FLOPs. TrafficSignCNN is the better choice for resource-constrained or real-time deployments.

---

## 👥 Contributors

| Name | Role |
|---|---|
| **Abdullah** | Model design, training pipeline, evaluation |
| **Wahaj** | Visualization, web demo, data preprocessing |

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).

---

> Feel free to ⭐ star, fork, and build on top of this project!

