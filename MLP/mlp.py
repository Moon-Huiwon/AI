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
X = torch.linspace(0, 10, 100).unsqueeze(1) # 두번째 차원에 크기 1인 차원 추가
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
