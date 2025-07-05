# 🔬 Cervical Cell Classification - Flask Application

A Flask-based web application for classifying cervical cell images using an advanced AI pipeline combining EfficientNetB0, Particle Swarm Optimization (PSO), and Support Vector Machine (SVM).

## 🎯 Overview

This application provides a complete solution for cervical cell classification with:

- **Feature Extraction**: EfficientNetB0 pre-trained model
- **Feature Selection**: Particle Swarm Optimization (PSO)
- **Classification**: Support Vector Machine (SVM) with linear kernel
- **Web Interface**: Modern, responsive Flask web application
- **API Endpoints**: RESTful API for integration

## 📋 Cell Categories

The model can classify cervical cells into 5 categories:

1. **im_Dyskeratotic** - Abnormal - Dyskeratotic cells
2. **im_Koilocytotic** - Abnormal - Koilocytotic cells  
3. **im_Metaplastic** - Benign - Metaplastic cells
4. **im_Parabasal** - Normal - Parabasal cells
5. **im_Superficial-Intermediate** - Normal - Superficial-Intermediate cells

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- 8GB+ RAM (for model training)
- GPU recommended (for faster training)

### Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model** (if you have the dataset):
   ```bash
   python train_model.py --dataset_path raw_dataset
   ```

4. **Run the Flask application**:
   ```bash
   python app.py
   ```

5. **Open your browser** and go to: `http://localhost:5000`

## 📁 Project Structure

```
├── app.py                          # Main Flask application
├── train_model.py                  # Model training script
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Web interface template
├── raw_dataset/                   # Dataset directory (if available)
│   ├── im_Dyskeratotic/
│   ├── im_Koilocytotic/
│   ├── im_Metaplastic/
│   ├── im_Parabasal/
│   └── im_Superficial-Intermediate/
└── README.md                      # This file
```

## 🎓 Model Training

### Dataset Requirements

The training script expects the following dataset structure:
```
raw_dataset/
├── im_Dyskeratotic/im_Dyskeratotic/CROPPED/*.bmp
├── im_Koilocytotic/im_Koilocytotic/CROPPED/*.bmp
├── im_Metaplastic/im_Metaplastic/CROPPED/*.bmp
├── im_Parabasal/im_Parabasal/CROPPED/*.bmp
└── im_Superficial-Intermediate/im_Superficial-Intermediate/CROPPED/*.bmp
```

### Training Process

The training pipeline consists of:

1. **Data Loading**: Load and preprocess images from all categories
2. **Feature Extraction**: Use EfficientNetB0 to extract 1280-dimensional features
3. **Feature Selection**: Apply PSO to select optimal feature subset
4. **Model Training**: Train SVM classifier on selected features
5. **Evaluation**: Generate performance metrics and visualizations

### Training Command

```bash
python train_model.py --dataset_path raw_dataset
```

**Expected Output Files**:
- `svm_linear_Eff_model.pkl` - Trained SVM model
- `pso_efficientnet.npy` - Feature selection mask
- `features_EfficientNet.npy` - Extracted features
- `confusion_matrix.png` - Confusion matrix visualization
- `pso_convergence.png` - PSO convergence curve
- `training_summary.txt` - Training summary report

## 🌐 Web Application

### Features

- **Drag & Drop Interface**: Easy image upload
- **Real-time Analysis**: Instant classification results
- **Confidence Scores**: Probability distribution for all classes
- **Responsive Design**: Works on desktop and mobile devices
- **Visual Feedback**: Loading indicators and progress bars

### Usage

1. **Upload Image**: Drag and drop or click to select a cervical cell image
2. **Preview**: Review the uploaded image
3. **Analyze**: Click "Analyze Image" to get classification results
4. **Results**: View predicted cell type, confidence score, and probability distribution

## 🔌 API Endpoints

### Health Check
```bash
GET /health
```
Returns the status of loaded models.

### Model Information
```bash
GET /model_info
```
Returns detailed information about the model architecture.

### Image Prediction (File Upload)
```bash
POST /predict
Content-Type: multipart/form-data

Form data:
- image: Image file (.bmp, .jpg, .png)
```

### Image Prediction (Base64)
```bash
POST /predict_base64
Content-Type: application/json

{
    "image": "base64_encoded_image_data"
}
```

### Example API Response
```json
{
    "predicted_class": "im_Superficial-Intermediate",
    "confidence": 0.85,
    "description": "Normal - Superficial-Intermediate cells",
    "probabilities": {
        "im_Dyskeratotic": 0.05,
        "im_Koilocytotic": 0.03,
        "im_Metaplastic": 0.02,
        "im_Parabasal": 0.05,
        "im_Superficial-Intermediate": 0.85
    }
}
```

## 🛠️ Technical Details

### Model Architecture

1. **EfficientNetB0**: Pre-trained on ImageNet for feature extraction
   - Input: 224×224×3 RGB images
   - Output: 1280-dimensional feature vectors
   - Global average pooling for dimensionality reduction

2. **Particle Swarm Optimization (PSO)**:
   - 30 particles
   - 50 iterations
   - Binary feature selection (0/1 mask)
   - Objective: Minimize (1 - validation accuracy)

3. **Support Vector Machine (SVM)**:
   - Linear kernel
   - Probability estimation enabled
   - Trained on PSO-selected features

### Preprocessing Pipeline

- **Resize**: 224×224 pixels
- **Augmentation**: Random rotation, flip, brightness/contrast, crop
- **Normalization**: ImageNet mean/std values
- **Color Conversion**: BGR to RGB

## 📊 Performance

Based on the notebook results:
- **Feature Reduction**: ~48% (664/1280 features selected)
- **Training Time**: ~2 hours (with PSO optimization)
- **Model Size**: EfficientNetB0 + SVM classifier

## 🔧 Configuration

### Model Parameters

Edit `app.py` to modify:
- `INPUT_SHAPE`: Image input size (default: 224)
- `CATEGORIES`: Cell type categories
- `CATEGORY_DESCRIPTIONS`: Human-readable descriptions

### Training Parameters

Edit `train_model.py` to modify:
- PSO parameters (particles, iterations)
- SVM kernel type
- Augmentation pipeline
- Train/test split ratio

## 🚨 Troubleshooting

### Common Issues

1. **"SVM model not loaded"**
   - Ensure `svm_linear_Eff_model.pkl` exists
   - Train the model first using `train_model.py`

2. **"Feature mask file not found"**
   - Ensure `pso_efficientnet.npy` exists
   - This file is generated during training

3. **Memory errors during training**
   - Reduce batch size or use GPU
   - Consider using fewer PSO particles/iterations

4. **Import errors**
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Optimization

- **GPU Usage**: Install TensorFlow-GPU for faster training
- **Memory**: Use data generators for large datasets
- **PSO**: Reduce particles/iterations for faster training

## 📝 License

This project is for educational and research purposes. Please ensure compliance with relevant medical data regulations when using with real patient data.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and enhancement requests.

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the training logs
3. Ensure all dependencies are installed correctly

---

**Note**: This application is designed for research and educational purposes. For clinical use, additional validation and regulatory compliance may be required. 