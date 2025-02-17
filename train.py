import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NoiseReductionDataset
from model import SimpleCNN
from constants import DATASET_DIR, IMAGE_SIZE
import os
from tqdm import tqdm # プログレスバー

def train():
    # ハイパーパラメータ
    batch_size = 64
    learning_rate = 0.001
    epochs = 100
    image_size = (IMAGE_SIZE, IMAGE_SIZE)  # 定数を使用
    num_workers = 4 # データローダーのワーカー数 (CPUのコア数に合わせて調整)

    # データセットとデータローダー
    train_dataset = NoiseReductionDataset(DATASET_DIR, image_size=image_size, train=True)
    print(f"Training on {len(train_dataset)} samples")
    val_dataset = NoiseReductionDataset(DATASET_DIR, image_size=image_size, train=False)
    print(f"Validating on {len(val_dataset)} samples")  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("Data loaders created.")

    # モデル、損失関数、オプティマイザー
    model = SimpleCNN().to('cuda') # GPUにモデルを転送
    criterion = nn.MSELoss()       # 平均二乗誤差損失
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Start training...")

    # 学習ループ
    for epoch in range(epochs):
        model.train() # 学習モードに設定
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train", leave=False) # プログレスバー設定
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to('cuda'), targets.to('cuda') # GPUにデータを転送

            optimizer.zero_grad() # 勾配初期化
            outputs = model(inputs) # 順伝播
            loss = criterion(outputs, targets) # 損失計算
            loss.backward() # 逆伝播
            optimizer.step() # パラメータ更新

            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'}) # プログレスバーに損失表示

        avg_train_loss = train_loss / len(train_loader)

        # 検証
        model.eval() # 評価モードに設定
        val_loss = 0.0
        with torch.no_grad(): # 勾配計算を無効化
            progress_bar_val = tqdm(val_loader, desc=f"Epoch [{epoch+1}/{epochs}] Val", leave=False) # 検証用プログレスバー
            for inputs, targets in progress_bar_val:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar_val.set_postfix({'val_loss': f'{loss.item():.4f}'}) # 検証損失表示

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # モデル保存 (エポックごとに保存)
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), f'checkpoints/model_epoch_{epoch+1}.pth') # モデルパラメータを保存

    print("Training finished!")

if __name__ == '__main__':
    train()