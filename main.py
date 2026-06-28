import os
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l1_l2
from keras.optimizers import Adam
import joblib
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
IMG_SIZE = (224, 224) # 64,64
THRESHOLD = 0.45  # Optimized threshold from analysis
DATASET_PATH = './data2'


# ==========================================
# DATA LOADING FUNCTIONS
# ==========================================
def load_labels_from_csv(csv_path):
    """Load labels from CSV file."""
    print(f"  Loading labels from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    if 'image' not in df.columns or 'forged' not in df.columns:
        print(f"  ERROR: Required columns not found in {csv_path}")
        return {}
    
    label_map = {}
    for _, row in df.iterrows():
        base_name = os.path.splitext(row['image'])[0]
        label_map[base_name] = int(row['forged'])
    
    forged_count = sum(1 for v in label_map.values() if v == 1)
    authentic_count = sum(1 for v in label_map.values() if v == 0)
    print(f"  Labels loaded: {authentic_count} authentic, {forged_count} forged")
    
    return label_map


def load_images_from_folder(folder_path, label_map, img_size=IMG_SIZE):
    """Load images as flattened vectors for MLP."""
    if not os.path.exists(folder_path):
        return np.array([]), np.array([])
    
    images, labels = [], []
    image_files = [f for f in os.listdir(folder_path) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    print(f"  Processing {len(image_files)} images from {folder_path}")
    
    for filename in image_files:
        base_name = os.path.splitext(filename)[0]
        if base_name not in label_map:
            continue
            
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        
        img_resized = cv2.resize(img, img_size)
        img_flat = img_resized.astype(np.float32).flatten() / 255.0
        
        images.append(img_flat)
        labels.append(label_map[base_name])
    
    if len(images) == 0:
        return np.array([]), np.array([])
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"  Loaded {len(images)} images (input size: {images.shape[1]})")
    print(f"  Class balance: {np.sum(labels == 0)} authentic, {np.sum(labels == 1)} forged")
    
    return images, labels


def load_dataset(dataset_path, img_size=IMG_SIZE):
    """Loads dataset from train/, val/, test/ folders."""
    print("Loading dataset from:", dataset_path)
    print("-" * 50)
    
    csv_files = {
        'train': os.path.join(dataset_path, 'train.txt'),
        'val': os.path.join(dataset_path, 'val.txt'),
        'test': os.path.join(dataset_path, 'test.txt'),
    }
    
    for split, csv_path in csv_files.items():
        if not os.path.exists(csv_path):
            print(f"❌ CSV file not found: {csv_path}")
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    label_maps = {}
    for split, csv_path in csv_files.items():
        print(f"\n{split.upper()} set:")
        label_maps[split] = load_labels_from_csv(csv_path)
    
    print("\n" + "-" * 50)
    
    X_train, y_train = load_images_from_folder(
        os.path.join(dataset_path, 'train'), label_maps.get('train', {}), img_size
    )
    X_val, y_val = load_images_from_folder(
        os.path.join(dataset_path, 'val'), label_maps.get('val', {}), img_size
    )
    X_test, y_test = load_images_from_folder(
        os.path.join(dataset_path, 'test'), label_maps.get('test', {}), img_size
    )
    
    print("-" * 50)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print("-" * 50)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================
# MODEL ARCHITECTURE
# ==========================================
def build_mlp_model(input_dim):
    """
    MLP with strong regularization for small dataset.
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        
        Dense(64, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(32, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(16, activation='relu', kernel_regularizer=l1_l2(l1=1e-5, l2=1e-4)),
        Dropout(0.4),
        
        Dense(1, activation='sigmoid'),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


# ==========================================
# MAIN TRAINING PIPELINE
# ==========================================
def main():
    print("=" * 60)
    print("INVOICE FORGERY DETECTION - MLP")
    print("=" * 60)
    
    # 1. Load Dataset
    print("\n[1/6] Loading Dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        DATASET_PATH, img_size=IMG_SIZE
    )
    
    if len(X_train) == 0:
        print("\n❌ Dataset not loaded. Please check the path.")
        return
    
    # 2. Combine Train and Validation
    print("\n[2/6] Preparing Data...")
    X_train_combined = np.vstack([X_train, X_val])
    y_train_combined = np.hstack([y_train, y_val])
    print(f"  Combined training set: {len(X_train_combined)} samples")
    
    # 3. Scale Features
    print("\n[3/6] Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Apply SMOTE for Class Balance
    print("\n[4/6] Applying SMOTE Oversampling...")
    n_authentic = np.sum(y_train_combined == 0)
    n_forged = np.sum(y_train_combined == 1)
    
    # Target: 30% forgeries after oversampling
    target_forged_ratio = 0.30
    n_forged_target = int(target_forged_ratio * n_authentic / (1 - target_forged_ratio))
    sampling_strategy = {0: n_authentic, 1: max(n_forged, n_forged_target)}
    
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_combined)
    
    print(f"  Original: {len(X_train_scaled)} samples")
    print(f"  Resampled: {len(X_train_resampled)} samples")
    print(f"  New balance: {np.sum(y_train_resampled == 0)} authentic, {np.sum(y_train_resampled == 1)} forged")
    
    # 5. Build and Train Model
    input_dim = X_train_scaled.shape[1]
    print(f"\n[5/6] Building and Training Model...")
    print(f"  Input dimension: {input_dim}")
    
    model = build_mlp_model(input_dim)
    model.summary()
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1,
    )
    
    history = model.fit(
        X_train_resampled,
        y_train_resampled,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    
    # 6. Evaluate Model
    print("\n[6/6] Evaluating Model...")
    
    # Get predictions
    y_pred_probs = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_probs >= THRESHOLD).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print("\n" + "=" * 50)
    print("FINAL EVALUATION METRICS:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("=" * 50)
    
    # Classification Report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print(classification_report(
        y_test, y_pred,
        target_names=['Authentic', 'Forged'],
        zero_division=0,
    ))
    
    # Visualizations
    print("\nGenerating Visualizations...")
    
    # 1. Training History
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Confusion Matrix
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["Authentic", "Forged"],
        cmap="Blues",
    )
    plt.title(f"Confusion Matrix (Threshold={THRESHOLD})")
    plt.show()
    
    # 3. Prediction Distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram by class
    axes[0].hist(y_pred_probs[y_test == 0], bins=20, alpha=0.5, 
                 label='Authentic', color='blue', density=True)
    axes[0].hist(y_pred_probs[y_test == 1], bins=20, alpha=0.5, 
                 label='Forged', color='red', density=True)
    axes[0].axvline(THRESHOLD, color='black', linestyle='--', 
                   label=f'Threshold={THRESHOLD}')
    axes[0].set_xlabel('Predicted Probability')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Prediction Distribution by Class')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, y_pred_probs)
    auc = roc_auc_score(y_test, y_pred_probs)
    
    axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    joblib.dump(scaler, 'scaler.pkl')
    # Save Model
    model.save('invoice_forgery_mlp_model.h5')

    print("\n" + "=" * 50)
    print("✓ Model saved as 'invoice_forgery_mlp_model.h5'")
    print("✓ Visualizations displayed successfully")
    print("=" * 50)

if __name__ == "__main__":
    main()