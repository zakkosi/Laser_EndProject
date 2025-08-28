import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import cv2
import os
from torchvision.models import resnet18
import matplotlib.pyplot as plt
from tqdm import tqdm

class LaserDataset(Dataset):
    def __init__(self, data_info_path):
        with open(data_info_path, 'r') as f:
            self.data_info = json.load(f)
    
    def __len__(self):
        return len(self.data_info)
    
    def __getitem__(self, idx):
        item = self.data_info[idx]
        
        # 차집합 이미지 로드
        img_path = item['diff_image_path']
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img / 255.0  # 정규화
        
        # 좌표 (이미지 크기로 정규화)
        coords = torch.FloatTensor([
            item['x'] / img.shape[1],  # x를 width로 나누어 0-1 범위
            item['y'] / img.shape[0]   # y를 height로 나누어 0-1 범위
        ])
        
        # 텐서 변환
        img_tensor = torch.FloatTensor(img).unsqueeze(0)  # (1, H, W)
        
        return img_tensor, coords

class LaserResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained=True)
        # grayscale 입력을 위해 첫 번째 conv layer 수정
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 좌표 출력을 위해 FC layer 수정 (0-1 범위 출력)
        self.backbone.fc = nn.Linear(512, 2)
    
    def forward(self, x):
        coords = self.backbone(x)
        coords = torch.sigmoid(coords)  # 0-1 범위로 제한
        return coords

def train():
    # GPU 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"사용 장치: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 메모리: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB")
    
    # 모델 저장 디렉토리 생성
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 데이터셋 로드
    dataset = LaserDataset("dataset/dataset_info.json")
    print(f"데이터셋 크기: {len(dataset)}개 샘플")
    
    # train/val 분할
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # RTX 4090이니까 배치 사이즈 크게
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    
    # 모델 초기화
    model = LaserResNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # 학습 기록
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    # Early stopping 설정
    patience = 15
    patience_counter = 0
    
    print("학습 시작...")
    for epoch in range(50):  # 에폭 수 줄임
        # 학습
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:3d} Train")
        for images, coords in train_pbar:
            images, coords = images.to(device), coords.to(device)
            
            optimizer.zero_grad()
            pred = model(images)
            loss = criterion(pred, coords)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # 실시간 loss 표시
            train_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 검증
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:3d} Val  ")
        with torch.no_grad():
            for images, coords in val_pbar:
                images, coords = images.to(device), coords.to(device)
                pred = model(images)
                loss = criterion(pred, coords)
                val_loss += loss.item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        # Early stopping 체크
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 베스트 모델 저장
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss
            }, 'models/best_laser_cnn.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch}. No improvement for {patience} epochs.")
            break
        
        if epoch % 5 == 0 or patience_counter > patience * 0.7:
            print(f"Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, Best={best_val_loss:.6f}, Patience={patience_counter}/{patience}")
    
    # 최종 모델도 저장
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss
    }, 'models/final_laser_cnn.pth')
    
    # 학습 곡선 저장
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Laser CNN Training Progress')
    plt.legend()
    plt.yscale('log')
    plt.grid(True)
    plt.savefig('results/training_curve.png', dpi=150)
    
    print(f"학습 완료!")
    print(f"최고 검증 손실: {best_val_loss:.6f}")
    print(f"총 샘플: Train {train_size}, Val {val_size}")

if __name__ == "__main__":
    train()