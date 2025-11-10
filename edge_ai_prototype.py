"""
Edge AI Prototype: Recyclable Items Image Classification
This script trains a lightweight model and converts it to TensorFlow Lite for edge deployment.
"""

# ============================================================================
# PART 1: IMPORT NECESSARY LIBRARIES
# ============================================================================

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import json
from datetime import datetime

print("TensorFlow version:", tf.__version__)
print("All libraries imported successfully!")

# ============================================================================
# PART 2: CONFIGURATION & SETUP
# ============================================================================

CONFIG = {
    "model_name": "RecyclableItemsClassifier",
    "input_shape": (224, 224, 3),
    "batch_size": 32,
    "epochs": 20,
    "learning_rate": 0.001,
    "validation_split": 0.2,
    "num_classes": 3,  # plastic, metal, paper
    "classes": ["plastic", "metal", "paper"]
}

# Create directories for models and reports
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)
os.makedirs("data", exist_ok=True)

# ============================================================================
# PART 3: SYNTHETIC DATA GENERATION (for demonstration)
# ============================================================================

def generate_synthetic_dataset(num_samples=300, img_shape=(224, 224, 3)):
    """
    Generate synthetic recyclable items dataset.
    In production, use real images from Kaggle or similar sources.
    """
    print("\n[1/5] Generating synthetic dataset...")
    
    X = []
    y = []
    
    # Generate 100 samples per class
    for class_idx in range(CONFIG["num_classes"]):
        for i in range(num_samples // CONFIG["num_classes"]):
            # Create synthetic images with different color patterns
            img = np.random.rand(*img_shape).astype(np.float32)
            
            if class_idx == 0:  # Plastic - bluish
                img[:, :, 2] += 0.3  # Increase blue channel
            elif class_idx == 1:  # Metal - grayish
                img += 0.2
            else:  # Paper - brownish
                img[:, :, 0] += 0.2  # Increase red channel
                img[:, :, 1] += 0.1  # Increase green channel
            
            img = np.clip(img, 0, 1)
            X.append(img)
            y.append(class_idx)
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    
    print(f"✓ Generated {len(X)} synthetic images with shape {X.shape}")
    return X, y

# ============================================================================
# PART 4: BUILD LIGHTWEIGHT MODEL (MobileNetV2)
# ============================================================================

def build_edge_model(input_shape, num_classes):
    """
    Build a lightweight MobileNetV2 model optimized for edge deployment.
    MobileNetV2 is ideal for Edge AI due to its small size and speed.
    """
    print("\n[2/5] Building lightweight Edge AI model...")
    
    # Transfer learning with MobileNetV2
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom top layers
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=CONFIG["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ Model built successfully")
    print(f"✓ Total parameters: {model.count_params():,}")
    
    return model

# ============================================================================
# PART 5: TRAIN MODEL
# ============================================================================

def train_model(model, X_train, y_train, X_val, y_val):
    """
    Train the Edge AI model with data augmentation.
    """
    print("\n[3/5] Training Edge AI model...")
    
    # Data augmentation for better generalization
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    # Train model
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=CONFIG["batch_size"]),
        validation_data=(X_val, y_val),
        epochs=CONFIG["epochs"],
        verbose=1
    )
    
    print("✓ Model training completed")
    return history

# ============================================================================
# PART 6: EVALUATE MODEL
# ============================================================================

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    """
    print("\n[4/5] Evaluating model performance...")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "classification_report": classification_report(
            y_test, y_pred_classes,
            target_names=CONFIG["classes"],
            output_dict=True
        )
    }
    
    print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"✓ Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=CONFIG["classes"]))
    
    return metrics, y_pred_classes

# ============================================================================
# PART 7: CONVERT TO TENSORFLOW LITE
# ============================================================================

def convert_to_tflite(model, output_path="models/recyclable_classifier.tflite"):
    """
    Convert trained model to TensorFlow Lite for edge deployment.
    TFLite reduces model size by ~4x and improves inference speed.
    """
    print("\n[5/5] Converting model to TensorFlow Lite...")
    
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    original_size = model.count_params() * 4 / (1024**2)  # Approximate
    tflite_size = os.path.getsize(output_path) / (1024**2)
    
    print(f"✓ TFLite model saved to {output_path}")
    print(f"✓ TFLite model size: {tflite_size:.2f} MB")
    
    return output_path

# ============================================================================
# PART 8: INFERENCE WITH TFLITE MODEL
# ============================================================================

def run_tflite_inference(tflite_path, test_sample):
    """
    Run inference using the TFLite model.
    This simulates edge device inference.
    """
    print("\n[BONUS] Running TFLite inference...")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Prepare input
    test_sample = np.expand_dims(test_sample, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_sample)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data[0])
    confidence = output_data[0][predicted_class]
    
    print(f"✓ Predicted class: {CONFIG['classes'][predicted_class]}")
    print(f"✓ Confidence: {confidence*100:.2f}%")
    
    return predicted_class, confidence

# ============================================================================
# PART 9: SAVE REPORT
# ============================================================================

def save_report(metrics, model, tflite_size):
    """
    Generate and save comprehensive report.
    """
    report = {
        "assignment": "Edge AI Prototype - Recyclable Items Classification",
        "timestamp": datetime.now().isoformat(),
        "model_config": CONFIG,
        "metrics": {
            "test_accuracy": metrics["test_accuracy"],
            "test_loss": metrics["test_loss"]
        },
        "model_info": {
            "total_parameters": int(model.count_params()),
            "tflite_size_mb": tflite_size
        },
        "framework": "TensorFlow Lite",
        "deployment_benefits": {
            "real_time_inference": "Sub-100ms inference on edge devices",
            "reduced_latency": "No cloud communication required",
            "privacy": "Data processing happens on device",
            "bandwidth": "No need to send raw data to cloud",
            "cost_effective": "Reduced cloud computing costs"
        }
    }
    
    report_path = "reports/edge_ai_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\n✓ Report saved to {report_path}")
    return report

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("EDGE AI PROTOTYPE: RECYCLABLE ITEMS CLASSIFICATION")
    print("="*70)
    
    # Step 1: Generate dataset
    X, y = generate_synthetic_dataset(num_samples=300)
    
    # Step 2: Split data
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Testing: {len(X_test)} samples")
    
    # Step 3: Build model
    model = build_edge_model(CONFIG["input_shape"], CONFIG["num_classes"])
    
    # Step 4: Train model
    history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluate model
    metrics, y_pred = evaluate_model(model, X_test, y_test)
    
    # Step 6: Convert to TFLite
    tflite_path = convert_to_tflite(model)
    tflite_size = os.path.getsize(tflite_path) / (1024**2)
    
    # Step 7: Run TFLite inference
    run_tflite_inference(tflite_path, X_test[0])
    
    # Step 8: Save report
    report = save_report(metrics, model, tflite_size)
    
    print("\n" + "="*70)
    print("✓ EDGE AI PROTOTYPE COMPLETED SUCCESSFULLY!")
    print("="*70)