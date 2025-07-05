#!/usr/bin/env python3
"""
Training script for EfficientNet + PSO + SVM cervical cell classification model
This script implements the complete pipeline from the notebook
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pyswarms as ps
import joblib
import time
import albumentations as A
import pandas as pd

# Configuration
INPUT_SHAPE = 224
CATEGORIES = ["im_Dyskeratotic", "im_Koilocytotic", "im_Metaplastic", "im_Parabasal", "im_Superficial-Intermediate"]

def load_dataset(dataset_path):
    """
    Load the dataset from the specified path
    Expected structure: dataset_path/category/category/CROPPED/*.bmp
    """
    images = []
    labels = []
    
    for cls in CATEGORIES:
        folder_path = os.path.join(dataset_path, cls, cls, "CROPPED")
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist")
            continue
            
        file_list = glob.glob(os.path.join(folder_path, "*.bmp"))
        print(f"Found {len(file_list)} images in {folder_path}")
        
        for file in file_list:
            img = cv2.imread(file)
            if img is None:
                continue
            # Convert from BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(cls)
    
    return images, labels

def get_augmentation_pipeline():
    """Get the augmentation pipeline"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.RandomCrop(height=100, width=100, p=0.5),
        A.Resize(height=INPUT_SHAPE, width=INPUT_SHAPE, p=1.0),
        A.HueSaturationValue(p=0.3),
        A.GaussianBlur(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
    ])

def preprocess_images(images, target_size=(INPUT_SHAPE, INPUT_SHAPE), augmentation_pipeline=None):
    """Preprocess images for feature extraction"""
    proc_imgs = []
    for img in images:
        # Resize image
        img_resized = cv2.resize(img, target_size)
        img_array = img_to_array(img_resized)
        
        # Apply augmentation
        if augmentation_pipeline:
            augmented = augmentation_pipeline(image=img_array)
            img_array = augmented["image"]
        
        proc_imgs.append(img_array)
    
    return np.array(proc_imgs)

def extract_features(images):
    """Extract features using EfficientNetB0"""
    print("Loading EfficientNetB0 model...")
    base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                               pooling='avg', input_shape=(INPUT_SHAPE, INPUT_SHAPE, 3))
    
    preproc_fn = tf.keras.applications.efficientnet.preprocess_input
    
    # Preprocess images
    print("Preprocessing images...")
    images_preprocessed = preproc_fn(images)
    
    # Extract features
    print("Extracting features...")
    features = base_model.predict(images_preprocessed, verbose=1)
    
    return features

def objective_function(mask, X, y):
    """
    PSO objective function for feature selection
    """
    n_particles = mask.shape[0]
    scores = np.zeros(n_particles)
    
    for i in range(n_particles):
        binary_mask = mask[i].astype(bool)
        
        # If no feature is selected, assign worst cost
        if np.sum(binary_mask) == 0:
            scores[i] = 1.0
        else:
            # Select features based on mask
            X_selected = X[:, binary_mask]
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.3, random_state=42)
            
            # Train SVM
            clf = SVC(kernel='linear', random_state=42)
            clf.fit(X_train, y_train)
            
            # Calculate accuracy
            acc = clf.score(X_val, y_val)
            scores[i] = 1 - acc  # lower is better
    
    return scores

def train_model(dataset_path):
    """Main training function"""
    print("=" * 60)
    print("CERVICAL CELL CLASSIFICATION MODEL TRAINING")
    print("=" * 60)
    
    # Step 1: Load dataset
    print("\n1. Loading dataset...")
    images, labels = load_dataset(dataset_path)
    print(f"Total images loaded: {len(images)}")
    
    # Step 2: Preprocess images
    print("\n2. Preprocessing images...")
    augmentation_pipeline = get_augmentation_pipeline()
    images_proc = preprocess_images(images, target_size=(INPUT_SHAPE, INPUT_SHAPE), 
                                   augmentation_pipeline=augmentation_pipeline)
    
    # Step 3: Extract features
    print("\n3. Extracting features using EfficientNetB0...")
    features = extract_features(images_proc)
    print(f"Extracted features shape: {features.shape}")
    
    # Save features
    np.save('features_EfficientNet.npy', features)
    print("Features saved to 'features_EfficientNet.npy'")
    
    # Step 4: Prepare data for PSO
    print("\n4. Preparing data for feature selection...")
    y = np.array(labels)
    X = features
    
    # Step 5: PSO Feature Selection
    print("\n5. Running PSO for feature selection...")
    options = {'c1': 2, 'c2': 2, 'w': 0.9, 'k': 3, 'p': 2}
    dimensions = X.shape[1]
    
    optimizer = ps.discrete.BinaryPSO(n_particles=30, dimensions=dimensions, options=options)
    
    start_time = time.time()
    best_cost, best_pos = optimizer.optimize(objective_function, iters=50, X=X, y=y, verbose=True)
    elapsed_time = time.time() - start_time
    
    print(f"Best PSO cost (1 - best accuracy): {best_cost}")
    print(f"Number of selected features: {np.sum(best_pos)}")
    print(f"Total time taken: {elapsed_time:.2f} seconds")
    
    # Save PSO results
    np.save('pso_efficientnet.npy', best_pos)
    print("PSO results saved to 'pso_efficientnet.npy'")
    
    # Step 6: Train final SVM model
    print("\n6. Training final SVM model...")
    selected_features = best_pos.astype(bool)
    X_selected = X[:, selected_features]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    
    # Train SVM
    clf = SVC(kernel='linear', random_state=42, probability=True)
    clf.fit(X_train, y_train)
    
    # Step 7: Evaluate model
    print("\n7. Evaluating model...")
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Step 8: Save model
    print("\n8. Saving model...")
    joblib.dump(clf, "svm_linear_Eff_model.pkl")
    print("Model saved to 'svm_linear_Eff_model.pkl'")
    
    # Step 9: Create visualizations
    print("\n9. Creating visualizations...")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(CATEGORIES))
    plt.xticks(tick_marks, [cat.replace('im_', '') for cat in CATEGORIES], rotation=45)
    plt.yticks(tick_marks, [cat.replace('im_', '') for cat in CATEGORIES])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    # Annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center", 
                    color="black", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("Confusion matrix saved to 'confusion_matrix.png'")
    
    # PSO Convergence Curve
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer.cost_history)
    plt.title("PSO Convergence Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (1 - accuracy)")
    plt.grid(True)
    plt.savefig('pso_convergence.png', dpi=300, bbox_inches='tight')
    print("PSO convergence curve saved to 'pso_convergence.png'")
    
    # Step 10: Generate summary report
    print("\n10. Generating summary report...")
    
    summary = {
        "model_type": "EfficientNetB0 + PSO + SVM",
        "total_images": len(images),
        "categories": CATEGORIES,
        "feature_extraction": "EfficientNetB0 with global average pooling",
        "total_features": features.shape[1],
        "selected_features": int(np.sum(best_pos)),
        "feature_reduction": f"{((features.shape[1] - np.sum(best_pos)) / features.shape[1] * 100):.1f}%",
        "test_accuracy": accuracy,
        "training_time": elapsed_time,
        "pso_iterations": 50,
        "pso_particles": 30
    }
    
    # Save summary
    with open('training_summary.txt', 'w') as f:
        f.write("CERVICAL CELL CLASSIFICATION MODEL TRAINING SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        for key, value in summary.items():
            f.write(f"{key.replace('_', ' ').title()}: {value}\n")
    
    print("Training summary saved to 'training_summary.txt'")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model files created:")
    print("- svm_linear_Eff_model.pkl (SVM model)")
    print("- pso_efficientnet.npy (Feature selection mask)")
    print("- features_EfficientNet.npy (Extracted features)")
    print("- confusion_matrix.png (Confusion matrix visualization)")
    print("- pso_convergence.png (PSO convergence curve)")
    print("- training_summary.txt (Training summary)")
    
    return summary

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train cervical cell classification model')
    parser.add_argument('--dataset_path', type=str, default='raw_dataset',
                       help='Path to the dataset directory')
    
    args = parser.parse_args()
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset path '{args.dataset_path}' does not exist")
        print("Please provide the correct path to your dataset")
        return
    
    # Train the model
    try:
        summary = train_model(args.dataset_path)
        print(f"\nFinal Test Accuracy: {summary['test_accuracy']:.4f}")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 