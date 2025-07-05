from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
import albumentations as A
from werkzeug.utils import secure_filename
import base64
from PIL import Image
import io
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model components
base_model = None
svm_model = None
feature_mask = None
augmentation_pipeline = None
clinical_model = None
clinical_scaler = None
clinical_features = None
clinical_encoders = None
INPUT_SHAPE = 224

# Category mapping
CATEGORIES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]
CATEGORY_DESCRIPTIONS = {
    "im_Dyskeratotic": "Abnormal - Dyskeratotic cells",
    "im_Koilocytotic": "Abnormal - Koilocytotic cells", 
    "im_Metaplastic": "Benign - Metaplastic cells",
    "im_Parabasal": "Normal - Parabasal cells",
    "im_Superficial-Intermediate": "Normal - Superficial-Intermediate cells"
}

def clean_data_for_json(data):
    """Clean data to make it JSON serializable by replacing NaN values"""
    if isinstance(data, dict):
        return {key: clean_data_for_json(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float) and math.isnan(data):
        return None
    elif isinstance(data, (int, float, str, bool)) or data is None:
        return data
    else:
        return str(data)

# Clinical feature to cell type mapping based on medical knowledge
CLINICAL_TO_CELL_MAPPING = {
    'Age': ['im_Metaplastic'],
    'Number of sexual partners': ['im_Koilocytotic', 'im_Parabasal'],
    'First sexual intercourse': ['im_Koilocytotic'],
    'Num of pregnancies': ['im_Metaplastic'],
    'Smokes': ['im_Dyskeratotic'],
    'Smokes (years)': ['im_Dyskeratotic'],
    'Smokes (packs/year)': ['im_Dyskeratotic'],
    'Hormonal Contraceptives': ['im_Metaplastic'],
    'Hormonal Contraceptives (years)': ['im_Metaplastic'],
    'IUD': ['im_Parabasal'],
    'IUD (years)': ['im_Parabasal'],
    'STDs': ['im_Koilocytotic', 'im_Parabasal'],
    'STDs (number)': ['im_Koilocytotic', 'im_Parabasal'],
    'STDs:condylomatosis': ['im_Koilocytotic'],
    'STDs:cervical condylomatosis': ['im_Koilocytotic'],
    'STDs:vaginal condylomatosis': ['im_Koilocytotic'],
    'STDs:vulvo-perineal condylomatosis': ['im_Koilocytotic'],
    'STDs:syphilis': ['im_Parabasal'],
    'STDs:pelvic inflammatory disease': ['im_Parabasal'],
    'STDs:genital herpes': ['im_Parabasal'],
    'STDs:molluscum contagiosum': ['im_Parabasal'],
    'STDs:AIDS': ['im_Koilocytotic', 'im_Parabasal'],
    'STDs:HIV': ['im_Koilocytotic'],
    'STDs:Hepatitis B': [],
    'STDs:HPV': ['im_Koilocytotic'],
    'STDs: Number of diagnosis': ['im_Koilocytotic', 'im_Parabasal'],
    'STDs: Time since first diagnosis': ['im_Koilocytotic'],
    'STDs: Time since last diagnosis': ['im_Koilocytotic'],
    'Dx:Cancer': ['im_Dyskeratotic'],
    'Dx:CIN': ['im_Dyskeratotic'],
    'Dx:HPV': ['im_Koilocytotic'],
    'Dx': ['im_Dyskeratotic', 'im_Koilocytotic'],
    'Hinselmann': ['im_Dyskeratotic'],
    'Schiller': ['im_Dyskeratotic'],
    'Citology': ['im_Superficial-Intermediate', 'im_Dyskeratotic'],
    'Biopsy': ['im_Dyskeratotic']
}

# Detailed cell class information for medical context
CELL_CLASS_INFO = {
    "im_Dyskeratotic": {
        "title": "Dyskeratotic Cells",
        "description": "Abnormal squamous epithelial cells with keratinization, irregular shape, and dense cytoplasm.",
        "clinical_significance": [
            "Strongly associated with high-grade cervical intraepithelial neoplasia (CIN II/III) or invasive cervical cancer.",
            "Seen in dysplastic lesions and squamous cell carcinoma.",
            "Presence indicates definitive pathology requiring immediate biopsy and clinical follow-up."
        ],
        "clinical_action": "High-risk → requires colposcopy and possible surgical intervention.",
        "risk_level": "High Risk",
        "risk_color": "#dc3545",
        "risk_emoji": "🔴"
    },
    "im_Koilocytotic": {
        "title": "Koilocytotic Cells",
        "description": "Squamous epithelial cells with perinuclear halos, nuclear enlargement, and irregularities.",
        "clinical_significance": [
            "Hallmark of HPV infection (especially high-risk HPV types 16 & 18).",
            "May be present in low-grade squamous intraepithelial lesions (LSIL).",
            "Can be reversible or progress to high-grade lesions if not monitored."
        ],
        "clinical_action": "Regular HPV testing, cytology follow-up, or HPV vaccination depending on age and guidelines.",
        "risk_level": "High Risk",
        "risk_color": "#dc3545",
        "risk_emoji": "🔴"
    },
    "im_Metaplastic": {
        "title": "Metaplastic Cells",
        "description": "Cells undergoing transformation (usually from columnar to squamous type) during normal repair or hormonal response.",
        "clinical_significance": [
            "Often found in the transformation zone of the cervix.",
            "Generally benign; may be present due to hormonal changes, inflammation, or cervical irritation.",
            "Can resemble abnormal cells; careful interpretation is needed."
        ],
        "clinical_action": "Usually no treatment needed unless associated with atypical changes or abnormal Pap smear.",
        "risk_level": "Low Risk",
        "risk_color": "#ffc107",
        "risk_emoji": "🟡"
    },
    "im_Parabasal": {
        "title": "Parabasal Cells",
        "description": "Immature basal epithelial cells, round with large nucleus and little cytoplasm.",
        "clinical_significance": [
            "Common in postmenopausal women or atrophic vaginitis.",
            "May be seen in inflammatory conditions or infections (e.g., PID, syphilis).",
            "Not typically cancerous but may obscure cytologic interpretation."
        ],
        "clinical_action": "Treat underlying inflammation or hormonal imbalance; repeat cytology if needed.",
        "risk_level": "Medium Risk",
        "risk_color": "#fd7e14",
        "risk_emoji": "🟠"
    },
    "im_Superficial-Intermediate": {
        "title": "Superficial-Intermediate Cells",
        "description": "Mature squamous epithelial cells found in healthy cervix.",
        "clinical_significance": [
            "Indicates a normal cytological finding.",
            "Found in women with no signs of dysplasia or infection.",
            "May vary in appearance with menstrual cycle or hormonal state."
        ],
        "clinical_action": "No action needed. Reassuring finding in routine Pap smear.",
        "risk_level": "No Risk",
        "risk_color": "#28a745",
        "risk_emoji": "🟢"
    }
}

def load_models():
    """Load the trained models and feature mask"""
    global base_model, svm_model, feature_mask, augmentation_pipeline, clinical_model, clinical_scaler, clinical_features, clinical_encoders
    
    try:
        # Load EfficientNet base model
        base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                                   pooling='avg', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3))
        
        # Load SVM model
        if os.path.exists('svm_linear_Eff_model.pkl'):
            svm_model = joblib.load('svm_linear_Eff_model.pkl')
        else:
            print("Warning: SVM model file not found. Please train the model first.")
            svm_model = None
        
        # Load feature mask
        if os.path.exists('pso_efficientnet.npy'):
            feature_mask = np.load('pso_efficientnet.npy')
        else:
            print("Warning: Feature mask file not found. Using all features.")
            feature_mask = None
        
        # Load clinical model components
        if os.path.exists('clinical_model.pkl'):
            clinical_model = joblib.load('clinical_model.pkl')
            clinical_features = joblib.load('clinical_features.pkl')
            clinical_scaler = joblib.load('clinical_scaler.pkl')
            clinical_encoders = joblib.load('clinical_encoders.pkl')
            print("Clinical model loaded successfully!")
        else:
            print("Warning: Clinical model files not found. Clinical prediction will not be available.")
            clinical_model = None
            clinical_features = None
            clinical_scaler = None
            clinical_encoders = None
        
        # Initialize augmentation pipeline
        augmentation_pipeline = A.Compose([
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.RandomCrop(height=100, width=100, p=0.5),
            A.Resize(height=224, width=224, p=1.0),
            A.HueSaturationValue(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
        ])
        
        print("Models loaded successfully!")
        
    except Exception as e:
        print(f"Error loading models: {e}")

def preprocess_image(image):
    """Preprocess a single image for prediction"""
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image_resized = cv2.resize(image, (INPUT_SHAPE, INPUT_SHAPE))
        image_array = img_to_array(image_resized)
        
        # Apply augmentation
        if augmentation_pipeline:
            augmented = augmentation_pipeline(image=image_array)
            image_array = augmented["image"]
        
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        
        return image_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def extract_features(image_array):
    """Extract features using EfficientNet"""
    try:
        if base_model is None:
            raise ValueError("Base model not loaded")
        
        # Preprocess for EfficientNet
        preproc_fn = tf.keras.applications.efficientnet.preprocess_input
        image_preprocessed = preproc_fn(image_array)
        
        # Extract features
        features = base_model.predict(image_preprocessed, verbose=0)
        
        return features
        
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

def predict_cell_type(image):
    """Predict cell type from image"""
    try:
        if svm_model is None:
            return {"error": "SVM model not loaded. Please train the model first."}
        
        # Preprocess image
        processed_image = preprocess_image(image)
        if processed_image is None:
            return {"error": "Failed to preprocess image"}
        
        # Extract features
        features = extract_features(processed_image)
        if features is None:
            return {"error": "Failed to extract features"}
        
        # Apply feature selection if mask exists
        if feature_mask is not None:
            features = features[:, feature_mask.astype(bool)]
        
        # Make prediction
        prediction = svm_model.predict(features)[0]
        probabilities = svm_model.predict_proba(features)[0]
        
        # Get confidence score
        confidence = np.max(probabilities)
        
        # Get detailed cell information
        cell_info = CELL_CLASS_INFO.get(prediction, {})
        
        # Create result
        result = {
            "predicted_class": prediction,
            "confidence": float(confidence),
            "description": CATEGORY_DESCRIPTIONS.get(prediction, "Unknown"),
            "probabilities": {
                cat: float(prob) for cat, prob in zip(CATEGORIES, probabilities)
            },
            "cell_info": cell_info
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

def predict_cell_type_from_clinical(patient_data):
    """Predict cell type from clinical data using the mapping"""
    try:
        cell_scores = {cell_type: 0.0 for cell_type in CATEGORIES}
        
        # Calculate scores based on clinical features
        for feature, value in patient_data.items():
            if feature in CLINICAL_TO_CELL_MAPPING:
                mapped_cells = CLINICAL_TO_CELL_MAPPING[feature]
                
                # Convert value to float and normalize
                try:
                    feature_value = float(value)
                    # Normalize value (assuming most features are 0-1 or small ranges)
                    if feature_value > 0:
                        normalized_value = min(feature_value / 10.0, 1.0)  # Cap at 1.0
                        
                        # Distribute score among mapped cell types
                        for cell_type in mapped_cells:
                            cell_scores[cell_type] += normalized_value / len(mapped_cells)
                except (ValueError, TypeError):
                    # If value is not numeric, assume presence = 1
                    for cell_type in mapped_cells:
                        cell_scores[cell_type] += 1.0 / len(mapped_cells)
        
        # Normalize scores to probabilities
        total_score = sum(cell_scores.values())
        if total_score > 0:
            probabilities = {cell_type: score / total_score for cell_type, score in cell_scores.items()}
        else:
            # Default to Superficial-Intermediate if no clinical indicators
            probabilities = {cell_type: 0.0 for cell_type in CATEGORIES}
            probabilities['im_Superficial-Intermediate'] = 1.0
        
        # Get predicted cell type
        predicted_cell = max(probabilities, key=probabilities.get)
        confidence = probabilities[predicted_cell]
        
        # Get detailed cell information
        cell_info = CELL_CLASS_INFO.get(predicted_cell, {})
        
        return {
            "predicted_cell_type": predicted_cell,
            "confidence": float(confidence),
            "description": CATEGORY_DESCRIPTIONS.get(predicted_cell, "Unknown"),
            "probabilities": {cell_type: float(prob) for cell_type, prob in probabilities.items()},
            "method": "clinical_mapping",
            "cell_info": cell_info
        }
        
    except Exception as e:
        return {"error": f"Clinical prediction failed: {str(e)}"}

def predict_cancer_risk(patient_data):
    """Predict cancer risk using clinical model"""
    try:
        if clinical_model is None:
            return {"error": "Clinical model not loaded"}
        
        # Prepare feature vector
        feature_vector = []
        
        for feature in clinical_features:
            if feature in patient_data:
                value = patient_data[feature]
                
                # Handle encoding for categorical variables
                if feature in clinical_encoders:
                    try:
                        value = clinical_encoders[feature].transform([str(value)])[0]
                    except:
                        value = 0  # Default value if encoding fails
                
                feature_vector.append(float(value))
            else:
                feature_vector.append(0.0)  # Default value for missing features
        
        # Scale features
        features_scaled = clinical_scaler.transform([feature_vector])
        
        # Make prediction
        prediction = clinical_model.predict(features_scaled)[0]
        probability = clinical_model.predict_proba(features_scaled)[0][1]
        
        # Get the predicted cell type from clinical data to align risk assessment
        cell_prediction = predict_cell_type_from_clinical(patient_data)
        predicted_cell_type = cell_prediction.get("predicted_cell_type", "im_Superficial-Intermediate")
        
        # Get cell info to use its risk level
        cell_info = CELL_CLASS_INFO.get(predicted_cell_type, {})
        
        # Use the cell type's risk level instead of probability-based risk
        risk_level = cell_info.get("risk_level", "Low")
        risk_color = cell_info.get("risk_color", "#28a745")
        
        # Keep the probability for reference but use cell-based risk assessment
        if risk_level == "High Risk":
            risk_level = "High"
        elif risk_level == "Medium Risk":
            risk_level = "Medium"
        elif risk_level == "Low Risk":
            risk_level = "Low"
        else:  # No Risk
            risk_level = "Low"
        
        return {
            'cancer_detected': bool(prediction),
            'cancer_probability': float(probability),
            'risk_level': risk_level,
            'risk_color': risk_color
        }
        
    except Exception as e:
        return {"error": f"Cancer risk prediction failed: {str(e)}"}

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template('index.html', 
                         categories=CATEGORIES, 
                         descriptions=CATEGORY_DESCRIPTIONS,
                         accuracy=None,
                         clinical_accuracy=None)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for unified prediction (image + clinical data)"""
    try:
        # Check for image file
        image_result = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                # Read and process image
                image_bytes = file.read()
                image = Image.open(io.BytesIO(image_bytes))
                
                # Make image prediction
                image_result = predict_cell_type(image)
        
        # Check for CSV file
        clinical_result = None
        if 'csv' in request.files:
            csv_file = request.files['csv']
            if csv_file.filename != '':
                # Read CSV data
                csv_content = csv_file.read().decode('utf-8')
                csv_data = pd.read_csv(io.StringIO(csv_content))
                
                # Convert to patient data format
                patient_data = csv_data.iloc[0].to_dict() if len(csv_data) > 0 else {}
                
                # Make clinical predictions
                cell_from_clinical = predict_cell_type_from_clinical(patient_data)
                cancer_risk = predict_cancer_risk(patient_data)
                
                clinical_result = {
                    "cell_prediction": cell_from_clinical,
                    "cancer_prediction": cancer_risk,
                    "patient_data": clean_data_for_json(patient_data)
                }
        
        # Combine results
        if image_result and clinical_result:
            # Both image and clinical data provided
            combined_result = {
                "image_prediction": image_result,
                "clinical_prediction": clinical_result,
                "fusion_prediction": {
                    "prediction": "Combined Analysis",
                    "cell_class": image_result.get("predicted_class", "Unknown"),
                    "cell_class_significance": f"Image confidence: {image_result.get('confidence', 0):.2f}, Clinical confidence: {clinical_result['cell_prediction'].get('confidence', 0):.2f}",
                    "cell_class_clinical_hint": f"Image suggests {image_result.get('predicted_class', 'Unknown')} cells, Clinical factors suggest {clinical_result['cell_prediction'].get('predicted_cell_type', 'Unknown')} cells"
                }
            }
        elif image_result:
            # Only image provided
            combined_result = {
                "prediction": "Image Analysis",
                "cell_class": image_result.get("predicted_class", "Unknown"),
                "cell_class_significance": f"Confidence: {image_result.get('confidence', 0):.2f}",
                "cell_class_clinical_hint": image_result.get("description", "No clinical data provided")
            }
        elif clinical_result:
            # Only clinical data provided
            combined_result = {
                "prediction": "Clinical Analysis",
                "cell_class": clinical_result["cell_prediction"].get("predicted_cell_type", "Unknown"),
                "cell_class_significance": f"Confidence: {clinical_result['cell_prediction'].get('confidence', 0):.2f}",
                "cell_class_clinical_hint": clinical_result["cell_prediction"].get("description", "Based on clinical factors")
            }
        else:
            return jsonify({"error": "No valid files provided"}), 400
        
        return jsonify(combined_result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/predict_image', methods=['POST'])
def predict_image():
    """API endpoint for image-only prediction (backward compatibility)"""
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image file selected"}), 400
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_cell_type(image)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """API endpoint for base64 encoded image prediction"""
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_cell_type(image)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models_loaded": {
            "efficientnet": base_model is not None,
            "svm": svm_model is not None,
            "feature_mask": feature_mask is not None,
            "clinical_model": clinical_model is not None,
            "clinical_features": clinical_features is not None
        }
    })

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    """API endpoint for clinical prediction (both cell type and cancer risk)"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Predict cell type from clinical data
        cell_prediction = predict_cell_type_from_clinical(data)
        
        # Predict cancer risk
        cancer_prediction = predict_cancer_risk(data)
        
        # Combine results
        result = {
            "cell_prediction": cell_prediction,
            "cancer_prediction": cancer_prediction,
            "patient_data": clean_data_for_json(data)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Clinical prediction failed: {str(e)}"}), 500

@app.route('/predict_cell_from_clinical', methods=['POST'])
def predict_cell_from_clinical():
    """API endpoint for cell type prediction from clinical data only"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = predict_cell_type_from_clinical(data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Cell prediction failed: {str(e)}"}), 500

@app.route('/predict_cancer_risk', methods=['POST'])
def predict_cancer_risk_route():
    """API endpoint for cancer risk prediction only"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        result = predict_cancer_risk(data)
        
        if "error" in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": f"Cancer risk prediction failed: {str(e)}"}), 500

@app.route('/model_info')
def model_info():
    """Get model information"""
    return jsonify({
        "model_type": "EfficientNetB0 + PSO + SVM + Clinical Mapping",
        "input_shape": f"{INPUT_SHAPE}x{INPUT_SHAPE}",
        "categories": CATEGORIES,
        "category_descriptions": CATEGORY_DESCRIPTIONS,
        "feature_extraction": "EfficientNetB0 with global average pooling",
        "feature_selection": "Particle Swarm Optimization (PSO)",
        "classifier": "Support Vector Machine (SVM) with linear kernel",
        "clinical_mapping": "Feature-based cell type prediction",
        "clinical_model_loaded": clinical_model is not None,
        "available_endpoints": [
            "/predict - Image-based cell classification",
            "/predict_clinical - Combined clinical prediction",
            "/predict_cell_from_clinical - Cell type from clinical data",
            "/predict_cancer_risk - Cancer risk prediction"
        ]
    })

if __name__ == '__main__':
    # Load models on startup
    load_models()
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000) 