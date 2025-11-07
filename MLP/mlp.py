#%%
import torch

"""
torch의 텐서
"""
x = torch.rand(3,3)
y = torch.ones(3,3)
print(x)
print(x.T)
print(y)
print(x + y)
print(x @ y)
# %%
"""
역전파: backward
"""
x = torch.tensor([2.0, 3.0], requires_grad=True) # gradient 계산해주는 코드
y = x ** 2 + 3 * x + 1
z = y.sum()
z.backward() # backward를 통해 gradient 계산
print(x.grad) # gradient 확인 (2x + 3 값과 동일)

# gradient 계산을 안하기 때문에 에러 발생
x = torch.tensor([2.0, 3.0])
y = x ** 2 + 3 * x + 1
z = y.sum()
z.backward() # backward 불가능
print(x.grad) # grad 없음

#%%
# 장치 관리 (cpu, gpu 설정)
device = 'cpu'
# windows - cuda
if torch.cuda.is_available():
    device = torch.device('cuda')
# apple silicon - mps
if torch.mps.is_available():
    device = torch.device('mps')
print(device)

# x.to(device)

#%%
# 난수 고정
import numpy as np
import random
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# windows - cuda
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# apple silicon - mps
if torch.mps.is_available():
    torch.mps.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
# %%
"""
선형 회귀 모델을 Pytorch로 구현
"""
from torch import nn
from torch import optim

# 더미 데이터 생성
X = torch.linspace(0, 10, 100).unsqueeze(1) # 두번째 차원에 크기 1인 차원 추가. (batch_size, feature_size)로 만들어야함.
X.shape # torch.Size([100, 1])
y_true = 2 * X + 1 + 0.5 * torch.randn_like(X)

model = nn.Linear(1, 1) # input=1, output=1 
criterion = nn.MSELoss() # 손실함수로 MSE 활용
optimizer = optim.SGD(model.parameters(), lr=0.01) # SGD를 활용하여 파라미터 갱신할 예정

epochs = 200
for epoch in range(epochs):
    # 기울기 초기화
    optimizer.zero_grad()
    # 예측 값 계산
    y_pred = model(X)
    # 손실 값 계산
    loss = criterion(y_pred, y_true)
    # 역전파 진행 (미분 계산)
    loss.backward()
    # 파라미터 업데이트
    optimizer.step()
    # 20번의 학습마다 기록
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}') # tensor 형식에서 숫자 추출하려면 .item() 사용

print('학습된 가중치 + 편향')
for name, param in model.named_parameters(): # 파라미터 이름과, 값
    print(f'{name}: {param.data}')

#%%
# 실제 데이터와 회귀선 비교

X_np = X.squeeze().numpy()
y_true_np = y_true.squeeze().numpy()
w_learned = model.weight.item()
b_learned = model.bias.item()
print(w_learned, b_learned)
y_line_np = w_learned * X_np + b_learned

import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
plt.scatter(X_np, y_true_np, s=20, color='b', alpha=0.6, label='Synthetic Data')
plt.plot(X_np, y_line_np, color='red', linewidth=2, label=f'Learned Model')

plt.title('Dummy Data')
plt.xlabel('X (Feature)')
plt.ylabel('y_true (Label)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# %%
"""
MLP 만들기 (입력 - 은닉 - 출력 구조)
"""
import torch
from torch import nn

input_dim = 20
hidden_dim = 10
output_dim = 1

fc1 = nn.Linear(input_dim, hidden_dim) # 20, 10
relu = nn.ReLU()
fc2 = nn.Linear(hidden_dim, output_dim) # 10, 1

X = torch.rand(size=(40, 20)) # 20개의 feature를 가진 40개의 데이터
X = fc1(X) # (40, 10)
X = relu(X) # 음수 전부 0으로 바꿈
y_pred = fc2(X) 
print(y_pred.size()) # (40, 1)


#%%
"""
숫자 판독기 만들기
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# 데이터 로드하기
digits = load_digits()
X = digits.data.astype(np.float32)
y = digits.target.astype(np.int64)

fig, ax = plt.subplots(figsize=(13, 5), nrows=2, ncols=5)
for i in range(10):
    ax[i // 5, i % 5].imshow(X[i].reshape(8, 8), cmap="gray")
    ax[i // 5, i % 5].set_title(f"Label: {y[i]}")
    ax[i // 5, i % 5].axis("off")
fig.tight_layout()


# 학습/검증 데이터 분리 및 표준화
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
) # 8:2

X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp
) # 1:1

scaler = StandardScaler().fit(X_train) # 표준화 처리 (X_train의 평균, 표준편차 저장) -> valid나 test에 대해 할 경우 Leakage(자료 누출)문제 발생
X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Dataset과 DataLoader
X_train_t = torch.from_numpy(X_train) # 텐서로 변환
X_valid_t = torch.from_numpy(X_valid)
X_test_t = torch.from_numpy(X_test)

y_train_t = torch.from_numpy(y_train)
y_valid_t = torch.from_numpy(y_valid)
y_test_t = torch.from_numpy(y_test)

train_ds = TensorDataset(X_train_t, y_train_t) # index를 기준으로 하나의 샘플 데이터 구분 (X, y)
valid_ds = TensorDataset(X_valid_t, y_valid_t)
test_ds = TensorDataset(X_test_t, y_test_t)

BATCH_SIZE = 32
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True) # 데이터를 batch_size 만큼 묶어서 세트로 설정, shuffle을 통해 일반화
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

for idx, batch in enumerate(train_loader):
    print(idx, batch)
    print(len(batch[1]))
    break
# %%
