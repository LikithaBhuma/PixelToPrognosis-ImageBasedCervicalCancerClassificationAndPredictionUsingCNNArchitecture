#!/usr/bin/env python3
"""
Test script for the Flask cervical cell classification application
"""

import requests
import json
import os
from PIL import Image
import numpy as np

def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing health endpoint...")
    try:
        response = requests.get('http://localhost:5000/health')
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Models loaded: {data['models_loaded']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app. Is it running?")
        return False

def test_model_info_endpoint():
    """Test the model info endpoint"""
    print("\nTesting model info endpoint...")
    try:
        response = requests.get('http://localhost:5000/model_info')
        if response.status_code == 200:
            data = response.json()
            print("✅ Model info retrieved")
            print(f"   Model type: {data['model_type']}")
            print(f"   Input shape: {data['input_shape']}")
            print(f"   Categories: {len(data['categories'])}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app")
        return False

def create_test_image():
    """Create a simple test image"""
    # Create a 224x224 RGB test image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save as temporary file
    test_image_path = 'test_image.png'
    img.save(test_image_path)
    return test_image_path

def test_prediction_endpoint():
    """Test the prediction endpoint with a test image"""
    print("\nTesting prediction endpoint...")
    
    # Create test image
    test_image_path = create_test_image()
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/predict', files=files)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Prediction successful")
            print(f"   Predicted class: {data['predicted_class']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            print(f"   Description: {data['description']}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app")
        return False
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def test_base64_endpoint():
    """Test the base64 prediction endpoint"""
    print("\nTesting base64 prediction endpoint...")
    
    # Create test image and convert to base64
    test_image_path = create_test_image()
    
    try:
        with open(test_image_path, 'rb') as f:
            import base64
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        payload = {'image': f'data:image/png;base64,{image_data}'}
        response = requests.post('http://localhost:5000/predict_base64', 
                               json=payload, 
                               headers={'Content-Type': 'application/json'})
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Base64 prediction successful")
            print(f"   Predicted class: {data['predicted_class']}")
            print(f"   Confidence: {data['confidence']:.3f}")
            return True
        else:
            print(f"❌ Base64 prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask app")
        return False
    finally:
        # Clean up test image
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def check_model_files():
    """Check if required model files exist"""
    print("\nChecking model files...")
    
    required_files = [
        'svm_linear_Eff_model.pkl',
        'pso_efficientnet.npy'
    ]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("FLASK APPLICATION TEST SUITE")
    print("=" * 60)
    
    # Check model files first
    model_files_ok = check_model_files()
    
    if not model_files_ok:
        print("\n⚠️  Some model files are missing.")
        print("   Please train the model first using:")
        print("   python train_model.py --dataset_path raw_dataset")
        print("\n   Or ensure the model files are in the current directory.")
        return
    
    # Test endpoints
    tests = [
        test_health_endpoint,
        test_model_info_endpoint,
        test_prediction_endpoint,
        test_base64_endpoint
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! The Flask application is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the Flask application.")
    
    print("\nTo start the Flask application:")
    print("python app.py")
    print("\nThen open: http://localhost:5000")

if __name__ == "__main__":
    main() 