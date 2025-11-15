# 1.autoencoder라는 신경망모형 학습 방법론이 왜 이상탐지라는 Task에 사용될 수 있는지 이해하기
# 2. 평가 metric이 F1 score를 쓰는 이유는? 데이터의 불균형 때문에 
# (정상 데이터:98% 이상치:2% 일때 accuracy로 성능 평가시 모든 데이터를 정상으로 판단하면 정확도가 98%로 성능이 좋은 것으로 해석 가능)
# (이는 이상치를 제대로 검출하지 못한 것임. 따라서 실제 이상을 잘 잡아내는 재현율(Recall), 그리고 잡아낸 것 중 실제 이상인 비율인 정밀도(Precision)을 모두 고려한 F1 score를 활용한다.)
# 3. 코드 분해하기
# 4. 임의의 이상탐지 방법론이 주어졌을 때, 데이터의 분할-모형적합-평가의 일련의 과정 정리하기
# 5. 일반화된 이상탐지 task 수행 방법론을 이해하고 난 뒤에, 이를 시계열로 확장하는 방법 이해하기
# 6. Unsupervised learning 임을 명심할 것

# 코드 참고: https://dacon.io/competitions/official/235930/codeshare/5508?page=1&dtype=recent
#%%
import random
import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
import subprocess
import sys

#%%
try:
    import wandb
except:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
    with open("./wandb_api.txt", "r") as f:
        key = f.readline()
    subprocess.run(["wandb", "login"], input=key[0], encoding='utf-8')
    import wandb

project = "credit_card_anomaly"
entity = "huiwon"

run = wandb.init(
    project=project,
    entity=entity,
    tags=["basecode"]
)

config = {
    'epochs': 4,
    'lr': 1e-2,
    'bs': 128, # 기본 배치사이즈 (64, 128, 256, 512)
    'seed': 41,
    'similarity': 'cossim'
}

config['cuda'] = torch.cuda.is_available()
wandb.config.update(config)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
"""
하이퍼파라미터
"""
EPOCHS = 400 # 에포크 수
LR = 1e-2 # 학습률
BS = 16384 # 배치 사이즈
SEED = 41
#%%
"""
시드 고정
"""
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED) # Seed 고정
# %%
"""
데이터 로드 및 데이터 생성
"""
train_df = pd.read_csv('../MLP/data/credit/train.csv')
train_df = train_df.drop(columns=['ID'])

val_df = pd.read_csv('../MLP/data/credit/val.csv')
val_df = val_df.drop(columns=['ID'])

train_dataset = torch.from_numpy(train_df.values)
train_ds = TensorDataset(train_dataset)
train_loader = DataLoader(train_ds, batch_size=BS, shuffle=True)

y_val_dataset = torch.from_numpy(val_df['Class'].values)
X_val_dataset = torch.from_numpy(val_df.drop(columns=['Class']).values)
val_ds = TensorDataset(X_val_dataset, y_val_dataset)
val_loader = DataLoader(val_ds, batch_size=BS, shuffle=False)

#%%
"""
Train (학습)
"""
encoder = nn.Sequential(
    nn.Linear(30, 64),
    nn.BatchNorm1d(64), # 배치단위로 정규화 -> 배치 사이즈가 클 때 혹은 깊은 모델일 때 사용하는 것이 좋음
    nn.LeakyReLU(),
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.LeakyReLU(),
)

decoder = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.LeakyReLU(),
    nn.Linear(64, 30),
)

optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR)
criterion = nn.L1Loss()

"""
mode='max': 지정한 평가지표가 크게(정확도가 높아질 때) 개선될수록 좋다는 뜻입니다. 즉, 검증 정확도가 더이상 늘지 않으면 학습률을 줄임.

factor=0.5: 학습률을 현재 값의 50%로 감소시킵니다.

patience=10: 10 에폭(epoch) 동안 평가지표가 개선되지 않으면 학습률을 줄임.

threshold_mode='abs': 절대값 기준으로 개선 여부를 판단함.

min_lr=1e-8: 학습률이 지정한 최소값 이하로 더이상 내려가지 않도록 제한.

verbose=True: 학습률이 변경될 때마다 메시지를 출력.
"""
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, threshold_mode='abs', min_lr=1e-8)

# 학습
best_score = 0
for epoch in range(EPOCHS):
    encoder.train()
    decoder.train()
    train_loss = []
    for x in train_loader:
        x = x[0].float()
        optimizer.zero_grad()
        e_x = encoder(x)
        _x = decoder(e_x)
        
        loss = criterion(x, _x)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
    
    if config['similarity'] == 'cossim':
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    encoder.eval()
    decoder.eval()
    pred = []
    true = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.float()
            e_x = encoder(x)
            _x = decoder(e_x)
            
            diff = cos(x, _x).tolist()
            batch_pred = np.where(np.array(diff)<0.95, 1, 0).tolist()
            pred += batch_pred
            true += y.tolist()
    
    score = f1_score(true, pred, average='macro') # "macro": 모든 클래스의 F1 score를 똑같이 평균
    print(f'Epoch: [{epoch}] Train loss:[{np.mean(train_loss)}] Val Score:[{score}]')

    wandb.log({'train_loss': np.mean(train_loss)})
    
    if scheduler is not None:
        scheduler.step(score) # 학습률 변경할지 판단하기 위해 score 값 전달
    
    if best_score < score:
        best_score = score
        torch.save({
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'score': best_score
        }, './model/credit/best_model.pth', _use_new_zipfile_serialization=False)
    

#%%
"""
추론
"""
# 학습할 때 사용했던 모델 structure 그대로 가져오기
encoder = nn.Sequential(
    nn.Linear(30, 64),
    nn.BatchNorm1d(64),
    nn.LeakyReLU(),
    nn.Linear(64, 128),
    nn.BatchNorm1d(128),
    nn.LeakyReLU(),
)

decoder = nn.Sequential(
    nn.Linear(128, 64),
    nn.BatchNorm1d(64),
    nn.LeakyReLU(),
    nn.Linear(64, 30),
)

# 저장된 파일에서 state_dict 불러오기
checkpoint = torch.load('./model/credit/best_model.pth', map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict']) # encoder 모델에 적용
decoder.load_state_dict(checkpoint['decoder_state_dict']) # decoder 모델에 적용

# 데이터 불러오기 및 tensor 변환 & DataLoader
test_df = pd.read_csv('../MLP/data/credit/test.csv')
test_df = test_df.drop(columns=['ID'])

test_dataset = torch.from_numpy(test_df.values)
test_ds = TensorDataset(test_dataset)
test_loader = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=0)

encoder.eval()
decoder.eval()

cos = nn.CosineSimilarity(dim=1, eps=1e-6)
pred = []
with torch.no_grad():
    for x in test_loader:
        x = x[0].float()
        e_x = encoder(x)
        _x = decoder(e_x)

        diff = cos(x, _x).tolist()
        batch_pred = np.where(np.array(diff)<0.95, 1, 0).tolist()
        pred += batch_pred

submit = pd.read_csv('../MLP/data/credit/sample_submission.csv')
submit['Class'] = pred
submit.to_csv('../MLP/result/credit/submit_autoencoder.csv', index=False)

#%%
# wandb에 모델 저장하기
artifact = wandb.Artifact(
    f'model1',
    type='model',
    metadata=config
)
artifact.add_file("./model/credit/best_model.pth")
wandb.log_artifact(artifact)
#%%
# wandb에서 모델 불러오기
artifact = wandb.use_artifact(f'{entity}/{project}/model1:v0', type='model')
artifact.metadata.items() # 데이터(config) 불러오기
model_dir = artifact.download() # wandb가 저장해놓은 로컬 경로
#%%
# model_name = sorted([x for x in os.listdir(model_dir) if x.endswith('pth')])
model_name = [x for x in os.listdir(model_dir) if x.endswith('pth')][0]
if config['cuda']:
    checkpoint = torch.load(f'{model_dir}/{model_name}')
else:
    checkpoint = torch.load(f'{model_dir}/{model_name}', map_location='cpu')
encoder.load_state_dict(checkpoint['encoder_state_dict']) # encoder 모델에 적용
decoder.load_state_dict(checkpoint['decoder_state_dict']) # decoder 모델에 적용
    
#%%
# wandb config 업데이트 하고 끝내기
wandb.config.update(config, allow_val_change=True)
wandb.run.finish()
# %%
