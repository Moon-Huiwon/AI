#%%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing, load_diabetes, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#%%
# 데이터 로드 및 분할
data = fetch_california_housing()
# data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%%
"""
PCA 차원 축소 활용 X_train.shape[1] -> 3
"""

# PCA (차원 축소)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("원본 차원: ", X_train.shape[1])
print("PCA 축소 차원: ", X_train_pca.shape[1])

# Pytorch tensor로 변환
X_train_tensor = torch.from_numpy(X_train_pca).float()
y_train_tensor = torch.from_numpy(y_train).float()

X_test_tensor = torch.from_numpy(X_test_pca).float()
y_test_tensor = torch.from_numpy(y_test).float()

# Dataset과 DataLoader
train_ds = TensorDataset(X_train_tensor, y_train_tensor)
test_ds = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = nn.Linear(3, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 100
for epoch in range(1, epochs+1):
    # 학습
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    
    train_loss = total_loss / len(train_loader.dataset)

# 테스트 평가
model.eval()
total_loss = 0
with torch.no_grad():
    y_pred_pca = []
    for xb, yb in test_loader:
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        y_pred_pca.extend(logits.detach().cpu().numpy())

y_pred_pca = np.array(y_pred_pca)

test_loss = total_loss / len(test_loader.dataset)
print(f"Test MSE Loss: {test_loss:.4f}")

#%%
"""
Autoencoder 활용 8 -> 5 -> 3 -> 5 -> 8
"""
# tensor 변환
X_train_tensor_ae = torch.from_numpy(X_train_scaled).float()
y_train_tensor_ae = torch.from_numpy(y_train).float()

X_test_tensor_ae = torch.from_numpy(X_test_scaled).float()
y_test_tensor_ae = torch.from_numpy(y_test).float()

# Dataset과 DataLoader
train_ds_ae = TensorDataset(X_train_tensor_ae, y_train_tensor_ae)
test_ds_ae = TensorDataset(X_test_tensor_ae, y_test_tensor_ae)

train_loader_ae = DataLoader(train_ds_ae, batch_size=32, shuffle=True)
test_loader_ae = DataLoader(test_ds_ae, batch_size=32)

encoder = nn.Sequential(
    nn.Linear(X_train.shape[1], 5),
    nn.ReLU(),
    nn.Linear(5, 3)
)

decoder = nn.Sequential(
    nn.Linear(3, 5),
    nn.ReLU(),
    nn.Linear(5, X_train.shape[1])
)

ae_optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
ae_criterion = nn.MSELoss()

# 학습
for epoch in range(1, epochs + 1):
    encoder.train()
    decoder.train()
    total_loss = 0
    for xb, _ in train_loader_ae:
        ae_optimizer.zero_grad()
        z = encoder(xb)
        x_recon = decoder(z)
        
        loss = criterion(x_recon, xb) # autoencoder: 기존 값으로 복구가 되었는지 확인하는 것
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * xb.size(0)
    train_loss = total_loss / len(train_loader_ae.dataset)

# latent vector 추출 및 회귀용 데이터셋 생성
encoder.eval()
decoder.eval()
with torch.no_grad():
    latent_train = encoder(X_train_tensor_ae)
    latent_test = encoder(X_test_tensor_ae)

train_latent_ds = TensorDataset(latent_train, y_train_tensor_ae)
test_latent_ds = TensorDataset(latent_test, y_test_tensor_ae)

train_latent_loader = DataLoader(train_latent_ds, batch_size=32, shuffle=True)
test_latent_loader = DataLoader(test_latent_ds, batch_size=32)

latent_reg_model = nn.Linear(3, 1)
latent_optimizer = torch.optim.Adam(latent_reg_model.parameters(), lr=1e-3)
criterion_reg = nn.MSELoss()

# 학습
for epoch in range(1, epochs + 1):
    latent_reg_model.train()
    total_loss = 0
    for xb, yb in train_latent_loader:
        latent_optimizer.zero_grad()
        logits = latent_reg_model(xb)
        loss = criterion_reg(logits, yb)
        loss.backward()
        latent_optimizer.step()
        total_loss += loss.item() * xb.size(0)
    
    train_loss = total_loss / len(train_latent_loader.dataset)

# 테스트 평가
latent_reg_model.eval()
total_loss = 0
with torch.no_grad():
    y_pred_ae = []
    for xb, yb in test_latent_loader:
        logits = latent_reg_model(xb)
        loss = criterion_reg(logits, yb)
        total_loss += loss.item() * xb.size(0)
        y_pred_ae.extend(logits.cpu().numpy())
    
    y_pred_ae = np.array(y_pred_ae)
    test_loss = total_loss / len(test_latent_loader.dataset)

#%%
y_true = y_test_tensor.numpy()

def regression_report(y_true, y_pred, label="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"=== {label} ===")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 : {r2:.4f}\n")

# 성능 비교
regression_report(y_true, y_pred_pca, "PCA 기반 회귀")
regression_report(y_true, y_pred_ae, "오토인코더 latent 기반 회귀")

#%%
import matplotlib.pyplot as plt
import numpy as np


# 샘플 수가 너무 많으면 일부 샘플만 시각화하는 것도 좋습니다
num_samples = 100

# 실제값 대비 PCA 예측 산점도
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_true[:num_samples], y_pred_pca[:num_samples], alpha=0.7, label='PCA Predicted')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('PCA Model Predictions vs Actual')
plt.legend()
plt.grid(True)

# 실제값 대비 Autoencoder latent 예측 산점도
plt.subplot(1, 2, 2)
plt.scatter(y_true[:num_samples], y_pred_ae[:num_samples], alpha=0.7, label='AE Predicted')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', label='Ideal')
plt.xlabel('Actual values')
plt.ylabel('Predicted values')
plt.title('Autoencoder Latent Model Predictions vs Actual')
plt.legend()
plt.grid(True)

plt.show()

# 라인 플롯 예제 (순서대로 실제 값, PCA 예측, AE 예측을 비교)
plt.figure(figsize=(10, 6))
plt.plot(y_true[:num_samples], label='Actual', marker='o')
plt.plot(y_pred_pca[:num_samples], label='PCA Predicted', marker='x')
plt.plot(y_pred_ae[:num_samples], label='AE Predicted', marker='^')
plt.xlabel('Sample index')
plt.ylabel('Value')
plt.title('Actual vs Predictions (First 100 samples)')
plt.legend()
plt.grid(True)
plt.show()

# %%
