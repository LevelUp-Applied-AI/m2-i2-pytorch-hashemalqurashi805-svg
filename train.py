import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# ─── Model Definition ─────────────────────────────────────────────────────────

class HousingModel(nn.Module):
    """Neural network for predicting housing prices from property features.
    Architecture: Linear(5, 32) -> ReLU -> Linear(32, 1)
    """

    def __init__(self):
        """Define the model layers."""
        super().__init__()
        # تعريف الطبقات الثلاث
        self.layer1 = nn.Linear(5, 32)   # 5 input features → 32 hidden units
        self.relu   = nn.ReLU()         # activation function
        self.layer2 = nn.Linear(32, 1)   # 32 hidden → 1 output (price prediction)

    def forward(self, x):
        """Define the forward pass."""
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x


# ─── Main Training Script ─────────────────────────────────────────────────────

def main():
    """Load data, train HousingModel, and save predictions."""

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    df = pd.read_csv('data/housing.csv')
    print(f"DataFrame Shape: {df.shape}")

    # ── 2. Separate Features and Target ──────────────────────────────────────
    feature_cols = ['area_sqm', 'bedrooms', 'floor', 'age_years', 'distance_to_center_km']
    X = df[feature_cols]
    y = df[['price_jod']]  # double brackets to keep shape (N, 1)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ── 6. Training Loop ──────────────────────────────────────────────────────
    num_epochs = 100
    print("\nStarting Training...")
    
    for epoch in range(num_epochs):
        # Forward pass
        predictions = model(X_tensor)
        loss = criterion(predictions, y_tensor)
        
        # Zero grads
        optimizer.zero_grad()
        
        # Backward
        loss.backward()
        
        # Update
        optimizer.step()
        
        # Print every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:3d}/{num_epochs}]: Loss = {loss.item():.4f}")

    # ── 7. Save Predictions ───────────────────────────────────────────────────
    with torch.no_grad(): # Use no_grad for inference
        final_predictions = model(X_tensor).numpy()
    
    actuals = y.values
    
    results_df = pd.DataFrame({
        'actual': actuals.flatten(),
        'predicted': final_predictions.flatten()
    })
    
    results_df.to_csv('predictions.csv', index=False)
    print("\n✓ Saved predictions.csv")


if __name__ == "__main__":
    main()