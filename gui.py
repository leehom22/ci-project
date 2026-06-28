import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model
import joblib
import os

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_PATH = 'invoice_forgery_mlp_model.h5'
SCALER_PATH = 'scaler.pkl'
IMG_SIZE = (224, 224)
THRESHOLD = 0.45

class ForgeryDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Invoice Forgery Detector")
        self.root.geometry("500x600")
        self.root.configure(bg="#f0f4f8")

        # Load Model and Scaler
        try:
            self.model = load_model(MODEL_PATH)
            self.scaler = joblib.load(SCALER_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model or scaler. Ensure you have trained the model first.\n\nDetails: {e}")
            self.root.destroy()
            return

        # UI Elements
        self.title_label = tk.Label(root, text="Invoice Forgery Detection", font=("Helvetica", 16, "bold"), bg="#f0f4f8")
        self.title_label.pack(pady=20)

        self.upload_btn = tk.Button(root, text="Upload Invoice Image", command=self.upload_image, font=("Helvetica", 12), bg="#4caf50", fg="white", padx=10, pady=5)
        self.upload_btn.pack(pady=10)

        self.panel = tk.Label(root, bg="#f0f4f8")
        self.panel.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Helvetica", 14, "bold"), bg="#f0f4f8")
        self.result_label.pack(pady=10)
        
        self.confidence_label = tk.Label(root, text="", font=("Helvetica", 12), bg="#f0f4f8")
        self.confidence_label.pack(pady=5)

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff")]
        )
        if len(file_path) > 0:
            self.display_image(file_path)
            self.predict_image(file_path)

    def display_image(self, path):
        # Resize for UI display purposes
        img = Image.open(path)
        img = img.resize((300, 300), Image.Resampling.LANCZOS)
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def predict_image(self, path):
        # 1. Preprocess exactly as in training
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            messagebox.showerror("Error", "Could not read the image file.")
            return
            
        img_resized = cv2.resize(img, IMG_SIZE)
        img_flat = img_resized.astype(np.float32).flatten() / 255.0
        
        # 2. Reshape and Scale
        img_reshaped = img_flat.reshape(1, -1)
        img_scaled = self.scaler.transform(img_reshaped)
        
        # 3. Predict
        pred_prob = self.model.predict(img_scaled)[0][0]
        
        # 4. Display Results
        if pred_prob >= THRESHOLD:
            self.result_label.config(text="⚠️ ALERT: FORGED INVOICE DETECTED", fg="red")
        else:
            self.result_label.config(text="✅ AUTHENTIC INVOICE", fg="green")
            
        self.confidence_label.config(text=f"Forgery Probability Score: {pred_prob:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryDetectionApp(root)
    root.mainloop()