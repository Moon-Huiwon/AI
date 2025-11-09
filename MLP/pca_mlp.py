#%%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#%%
# 데이터 로드 및 분할
data = fetch_california_housing()
X, y = data.data, data.target
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


# PCA (차원 축소)
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
X_valid_pca = pca.transform(X_valid_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("원본 차원: ", X_train.shape[1])
print("PCA 축소 차원: ", X_train_pca.shape[1])

#%%
# input dimension 5 -> out_dim 1 : nn.Linear(5,1) 억지로 torch 이용해서 주성분회귀
# autoencoder 이론(정의) 찾아보기 latent -> PCA랑 동일한 역할
# autoencoder 공부해라--------


#%%
