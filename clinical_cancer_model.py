#!/usr/bin/env python3
"""
Clinical Cervical Cancer Prediction Model
Analyzes patient data to predict cervical cancer risk
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """Load and preprocess the cervical cancer dataset"""
    print("Loading cervical cancer dataset...")
    
    try:
        # Load the dataset
        df = pd.read_csv('cervical_cancer.csv')
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum())
        
        # Handle missing values - replace with median for numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # For categorical columns, replace with mode
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
        
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        raise

def prepare_features(df):
    """Prepare features for the clinical model"""
    print("\nPreparing features...")
    
    # Separate features and target
    # Assuming the last 4 columns are health reports (target variables)
    feature_columns = df.columns[:-4].tolist()
    target_columns = df.columns[-4:].tolist()
    
    print(f"Feature columns: {feature_columns}")
    print(f"Target columns: {target_columns}")
    
    # For this example, we'll use the first target column as our main prediction target
    # You can modify this based on your specific needs
    main_target = target_columns[0]  # 'Hinselmann' or similar
    
    X = df[feature_columns]
    y = df[main_target]
    
    # Encode categorical variables if any
    label_encoders = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    # Feature selection
    print("Performing feature selection...")
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    
    print(f"Selected features: {selected_features}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    return X_scaled, y, selected_features, scaler, label_encoders, main_target

def train_models(X, y):
    """Train multiple classification models"""
    print("\nTraining models...")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Initialize models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        roc_auc_str = f"{roc_auc:.4f}" if roc_auc is not None else 'N/A'
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc_str}")
        print(f"Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # Print detailed classification report
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred))
    
    return results, X_test, y_test

def save_models(results, selected_features, scaler, label_encoders, main_target):
    """Save the trained models and preprocessing objects"""
    print("\nSaving models...")
    
    # Save the best model (highest accuracy)
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    # Save all components
    joblib.dump(best_model, 'clinical_model.pkl')
    joblib.dump(selected_features, 'clinical_features.pkl')
    joblib.dump(scaler, 'clinical_scaler.pkl')
    joblib.dump(label_encoders, 'clinical_encoders.pkl')
    
    # Save model info
    model_info = {
        'best_model': best_model_name,
        'target_variable': main_target,
        'accuracy': results[best_model_name]['accuracy'],
        'roc_auc': results[best_model_name]['roc_auc'],
        'all_results': {name: {
            'accuracy': results[name]['accuracy'],
            'roc_auc': results[name]['roc_auc'],
            'cv_mean': results[name]['cv_mean'],
            'cv_std': results[name]['cv_std']
        } for name in results.keys()}
    }
    
    joblib.dump(model_info, 'clinical_model_info.pkl')
    
    print(f"Best model saved: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")
    roc_auc_value = results[best_model_name]['roc_auc']
    roc_auc_str = f"{roc_auc_value:.4f}" if roc_auc_value is not None else 'N/A'
    print(f"ROC AUC: {roc_auc_str}")

def generate_report(results, X_test, y_test, selected_features):
    """Generate a comprehensive report"""
    print("\nGenerating report...")
    
    report = []
    report.append("CLINICAL CERVICAL CANCER PREDICTION MODEL REPORT")
    report.append("=" * 50)
    report.append("")
    
    # Model comparison
    report.append("MODEL COMPARISON:")
    report.append("-" * 20)
    for name, result in results.items():
        report.append(f"{name}:")
        report.append(f"  Accuracy: {result['accuracy']:.4f}")
        roc_auc_value = result['roc_auc']
        roc_auc_str = f"{roc_auc_value:.4f}" if roc_auc_value is not None else 'N/A'
        report.append(f"  ROC AUC: {roc_auc_str}")
        report.append(f"  CV Score: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        report.append("")
    
    # Best model details
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    report.append(f"BEST MODEL: {best_model_name}")
    report.append("-" * 30)
    report.append(f"Accuracy: {best_result['accuracy']:.4f}")
    roc_auc_value = best_result['roc_auc']
    roc_auc_str = f"{roc_auc_value:.4f}" if roc_auc_value is not None else 'N/A'
    report.append(f"ROC AUC: {roc_auc_str}")
    report.append("")
    
    # Feature importance (for tree-based models)
    if hasattr(best_result['model'], 'feature_importances_'):
        report.append("FEATURE IMPORTANCE:")
        report.append("-" * 20)
        importances = best_result['model'].feature_importances_
        feature_importance = list(zip(selected_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        for feature, importance in feature_importance[:10]:  # Top 10 features
            report.append(f"{feature}: {importance:.4f}")
        report.append("")
    
    # Save report
    with open('clinical_model_report.txt', 'w') as f:
        f.write('\n'.join(report))
    
    print("Report saved to clinical_model_report.txt")

def main():
    """Main function to run the complete pipeline"""
    print("CLINICAL CERVICAL CANCER PREDICTION MODEL")
    print("=" * 50)
    
    try:
        # Load and preprocess data
        df = load_and_preprocess_data()
        
        # Prepare features
        X, y, selected_features, scaler, label_encoders, main_target = prepare_features(df)
        
        # Train models
        results, X_test, y_test = train_models(X, y)
        
        # Save models
        save_models(results, selected_features, scaler, label_encoders, main_target)
        
        # Generate report
        generate_report(results, X_test, y_test, selected_features)
        
        print("\n" + "=" * 50)
        print("MODEL TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print("\nFiles created:")
        print("- clinical_model.pkl (trained model)")
        print("- clinical_features.pkl (selected features)")
        print("- clinical_scaler.pkl (feature scaler)")
        print("- clinical_encoders.pkl (label encoders)")
        print("- clinical_model_info.pkl (model information)")
        print("- clinical_model_report.txt (detailed report)")
        
    except Exception as e:
        print(f"Error during model training: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 