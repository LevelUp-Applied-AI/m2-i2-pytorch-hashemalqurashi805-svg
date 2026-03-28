import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
import datetime
import os

# ─── Experiment Tracking Function ─────────────────────────────────────────────
def log_experiment(metrics, params):
    """حفظ نتائج التجربة في مجلد experiments"""
    if not os.path.exists('experiments'):
        os.makedirs('experiments')
    
    experiment_data = {
        "timestamp": str(datetime.datetime.now()),
        "parameters": params,
        "metrics": metrics
    }
    
    filename = f"experiments/run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w') as f:
        json.dump(experiment_data, f, indent=4)
    
    print(f"\n✅ تم حفظ نتائج التجربة في: {filename}")

# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    def __init__(self):
        super().__init__()
        # إعداد الطبقات
        self.layer1 = nn.Linear(5, 32)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    # ── 1. Load Data ──────────────────────────────────────────────────────────
    # ملاحظة: تأكد أن ملف housing.csv موجود في مجلد data
    try:
        df = pd.read_csv('data/housing.csv')
        print(f"Data Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ خطأ: ملف data/housing.csv غير موجود!")
        return

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]

    # ── 3. Standardize Features ───────────────────────────────────────────────
    X_mean = X.mean()
    X_std  = X.std()
    X_scaled = (X - X_mean) / X_std

    # ── 4. Convert to Tensors ─────────────────────────────────────────────────
    X_tensor = torch.tensor(X_scaled.values, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    print(f"X_tensor shape: {X_tensor.shape}, y_tensor shape: {y_tensor.shape}")

    # ── 5. Instantiate Model, Loss, and Optimizer ─────────────────────────────
    model     = HousingModel()
    criterion = nn.MSELoss()
    learning_rate = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    last_loss = 0.0

    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        last_loss = loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]: Loss = {last_loss:.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    with torch.no_grad():
        final_predictions = model(X_tensor).numpy()
        actuals = y_tensor.numpy()

    results_df = pd.DataFrame({'actual': actuals.flatten(), 'predicted': final_predictions.flatten()})
    results_df.to_csv('predictions.csv', index=False)
    print("Saved predictions.csv")

    # ── 8. Log Experiment Results (تتبع التجارب) ──────────────────────────────
    # هنا نقوم بتنفيذ المهمة المطلوبة في الـ Stretch Task
    metrics = {
        "final_loss": round(last_loss, 6),
        "epochs": num_epochs
    }
    params = {
        "learning_rate": learning_rate,
        "model_architecture": "5 -> 32 (ReLU) -> 1",
        "optimizer": "Adam"
    }
    log_experiment(metrics, params)


if __name__ == "__main__":
    main()