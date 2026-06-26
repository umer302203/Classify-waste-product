# 🌱 Waste Classification using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19.0-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.0+-red.svg)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Spaces-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Classify waste products as Organic (O) or Recyclable (R) using Deep Learning & Transfer Learning**

[![Open in Hugging Face](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Umer78786/Waste-Classify)
[![Open in GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/umer302203/Classify-waste-product)

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Live Demo](#-live-demo)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

This project implements a waste classification system using **Transfer Learning** with the **VGG16** architecture. The model is designed to automatically classify waste images into two categories:

- **♻️ Organic (O)**: Food waste, leaves, biodegradable materials
- **🗑️ Recyclable (R)**: Plastic, glass, paper, metal, and other recyclable materials

### Why This Matters
- Improves waste sorting efficiency
- Reduces contamination in recycling streams
- Supports environmental sustainability efforts
- Demonstrates the power of AI in solving real-world problems

---

## 📊 Dataset

### Source
The dataset used is the "Organic vs Recyclable" dataset from Kaggle, containing **1,200+ images** of waste items.

### Structure
```
o-vs-r-split-reduced-1200/
├── train/
│   ├── O/          # Organic waste images
│   └── R/          # Recyclable waste images
└── test/
    ├── O/
    └── R/
```

### Data Split
- **Training**: 800 images (400 per class)
- **Validation**: 200 images (100 per class)
- **Test**: 200 images (100 per class)

---

## 🔬 Methodology

### 1️⃣ Transfer Learning Approach

Transfer learning leverages pre-trained models to solve new tasks with limited data. We use **VGG16** pre-trained on ImageNet:

```
                ┌─────────────────┐
   Input Image  │                 │
   (150x150x3)  │    VGG16 Base   │
                │   (Frozen Weights)│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Feature Vector │
                │   (Flatten)     │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Dense (512)   │
                │   ReLU + Dropout│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │   Dense (512)   │
                │   ReLU + Dropout│
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │  Dense (1)      │
                │  Sigmoid        │
                └─────────────────┘
```

### 2️⃣ Phase 1: Feature Extraction

The base VGG16 model (without top layers) extracts features from images. We freeze all base layers and train only the custom top layers:

```python
# Load pre-trained VGG16 (without top layers)
vgg = vgg16.VGG16(include_top=False, weights='imagenet', 
                  input_shape=(150, 150, 3))

# Freeze base layers
for layer in basemodel.layers:
    layer.trainable = False

# Add custom classification head
model = Sequential()
model.add(basemodel)
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
```

### 3️⃣ Phase 2: Fine-Tuning

After feature extraction, we unfreeze the top convolutional layers for fine-tuning to adapt the pre-trained features to our specific task:

```python
# Unfreeze last convolutional block
for layer in basemodel.layers:
    if layer.name in ['block5_conv3']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
```

### 4️⃣ Data Augmentation

To prevent overfitting and improve generalization:

- **Width Shift Range**: 10%
- **Height Shift Range**: 10%
- **Horizontal Flip**: Enabled
- **Rescaling**: 1/255 normalization

### 5️⃣ Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 10 |
| Learning Rate | 1e-4 (exponential decay) |
| Optimizer | RMSprop |
| Loss Function | Binary Cross-Entropy |
| Validation Split | 20% |

### 6️⃣ Regularization Techniques

- **Dropout**: 0.3 to prevent overfitting
- **Early Stopping**: Patience = 4 epochs
- **Learning Rate Scheduling**: Exponential decay (`lr = lr_initial * exp(-0.1 * epoch)`)
- **Model Checkpointing**: Save best model based on validation loss

---

## 📈 Results

### Feature Extraction Model

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 80% |
| **Precision (O/R)** | 0.80 / 0.80 |
| **Recall (O/R)** | 0.80 / 0.80 |
| **F1-Score (O/R)** | 0.80 / 0.80 |

**Classification Report:**
```
              precision    recall  f1-score   support
           O       0.80      0.80      0.80        50
           R       0.80      0.80      0.80        50
    accuracy                           0.80       100
```

### Fine-Tuned Model

| Metric | Score |
|--------|-------|
| **Test Accuracy** | 81% |
| **Precision (O/R)** | 0.82 / 0.80 |
| **Recall (O/R)** | 0.80 / 0.82 |
| **F1-Score (O/R)** | 0.81 / 0.81 |

**Classification Report:**
```
              precision    recall  f1-score   support
           O       0.82      0.80      0.81        50
           R       0.80      0.82      0.81        50
    accuracy                           0.81       100
```

### Key Observations
- Fine-tuning improved accuracy by ~1%
- Models show good precision-recall balance
- No significant overfitting observed
- Fine-tuning helped adapt pre-trained features to waste classification

---

## 🚀 Live Demo

Try the model live on Hugging Face Spaces:

[![Open in Hugging Face](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/Umer78786/Waste-Classify)

Upload an image of waste and get instant classification results!

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.x
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/umer302203/Classify-waste-product.git
cd Classify-waste-product
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Requirements
```
tensorflow>=2.19.0
numpy>=1.24.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
requests>=2.31.0
tqdm>=4.66.0
gradio>=4.0.0
```

---

## 📖 Usage

### Run the Jupyter Notebook
```bash
jupyter notebook Classify_waste_product_using_Transfer_Learning.ipynb
```

### Run the Gradio Web App
```bash
python app.py
```

### Quick Inference

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('O_R_tlearn_fine_tune_vgg16.keras')

# Load and preprocess image
def predict_waste(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    label = 'Organic' if prediction < 0.5 else 'Recyclable'
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    
    return label, confidence

# Example usage
label, confidence = predict_waste('path/to/your/image.jpg')
print(f"Classification: {label} (Confidence: {confidence:.2f})")
```

---

## 📁 Project Structure

```
Classify-waste-product/
│
├── Classify_waste_product_using_Transfer_Learning.ipynb   # Main notebook
├── app.py                                                  # Gradio web app
├── requirements.txt                                        # Dependencies
├── README.md
├── LICENSE
└── .gitignore
```

---

## 🔮 Future Improvements

- [ ] **Additional Categories**: Extend to multi-class classification (metal, glass, hazardous waste)
- [ ] **Better Architectures**: Experiment with ResNet50, EfficientNet, MobileNet
- [ ] **Real-time Deployment**: Optimize for mobile/edge deployment
- [ ] **Model Compression**: Implement quantization and pruning
- [ ] **Video Processing**: Classify waste from video streams
- [ ] **Data Augmentation**: Implement more sophisticated augmentation techniques
- [ ] **Explainability**: Add Grad-CAM visualization for interpretability

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Write clear commit messages
- Update documentation as needed
- Add tests for new features

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **VGG16** researchers for the pre-trained model
- **Kaggle** for providing the waste dataset
- **TensorFlow** and **Keras** teams for deep learning frameworks
- **Hugging Face** for Spaces hosting
- Open-source community for tools and inspiration

---

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub and share it with others!

---

<div align="center">

**Made with ❤️ for a cleaner planet**

</div>
