import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import NoiseReductionDataset
from rednet import REDNet
from unet import UNet
from constants import DATASET_DIR, IMAGE_SIZE
import os
from tqdm import tqdm
import glob
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image
from vgg_loss import VGGLoss

def get_latest_checkpoint():
    # チェックポイントファイルを検索
    checkpoint_files = glob.glob('checkpoints/checkpoint_*.pth')
    if not checkpoint_files:
        return None
    # エポック番号でソートして最新のものを返す
    latest_file = max(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    return latest_file

def inference_at_checkpoint(model, val_dataset, epoch):
    """チェックポイントでの推論実行"""
    model.eval()
    os.makedirs('inference_results', exist_ok=True)
    
    # 検証データセットから1枚目の画像を取得
    input, target = val_dataset[0]  # インデックス0の画像を取得
    input = input.unsqueeze(0).to('cuda')  # バッチ次元を追加してGPUに転送
    target = target.unsqueeze(0).to('cuda')
    
    with torch.no_grad():
        output = model(input)
        
    # 入力、出力、ターゲットを並べて保存
    result = torch.cat([input, output, target], dim=0)
    save_image(result, f'inference_results/result_epoch_{epoch}.png', nrow=1, normalize=True)

def train():
    # ハイパーパラメータ
    batch_size = 1
    initial_lr = 0.001  # 初期学習率
    epochs = 300
    image_size = (IMAGE_SIZE, IMAGE_SIZE)  # 定数を使用
    num_workers = 0  # デバッグのために0に設定

    # データセットとデータローダー
    train_dataset = NoiseReductionDataset(DATASET_DIR, image_size=image_size, train=True)
    print(f"Training on {len(train_dataset)} samples")
    val_dataset = NoiseReductionDataset(DATASET_DIR, image_size=image_size, train=False)
    print(f"Validating on {len(val_dataset)} samples")  
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print("Data loaders created.")

    # モデル、損失関数、オプティマイザー
    model = REDNet().to('cuda') # GPUにモデルを転送
    content_criterion = nn.MSELoss()  # ピクセルレベルの損失
    perceptual_criterion = VGGLoss().to('cuda')  # 知覚損失
    
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.02) 
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # GradScalerの初期化
    scaler = torch.amp.GradScaler()

    # 開始エポックの初期化
    start_epoch = 0

    # チェックポイントの確認と読み込み
    latest_checkpoint = get_latest_checkpoint()
    if (latest_checkpoint):
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # スケジューラの状態を読み込み
        start_epoch = checkpoint['epoch']
        print(f"Resuming training from epoch {start_epoch}")

    print("Start training...")

    # 学習ループ
    for epoch in range(start_epoch, epochs):
        model.train() # 学習モードに設定
        train_loss = 0.0
        current_lr = scheduler.get_last_lr()[0]  # 現在の学習率を取得
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}] Train (lr: {current_lr:.6f})", leave=False) # プログレスバー設定
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to('cuda'), targets.to('cuda') # GPUにデータを転送

            optimizer.zero_grad() # 勾配初期化
            outputs = model(inputs) # 順伝播
            
            # コンテンツ損失とVGG損失の組み合わせ
            content_loss = content_criterion(outputs, targets)
            perceptual_loss = perceptual_criterion(outputs, targets)
            loss = content_loss + 0.1 * perceptual_loss  # 重み付け係数は調整可能
            
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
                loss = content_criterion(outputs, targets)
                val_loss += loss.item()
                progress_bar_val.set_postfix({'val_loss': f'{loss.item():.4f}'}) # 検証損失表示

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        # エポック終了時にスケジューラのステップを実行
        scheduler.step()

        # モデル保存 (10エポックごとに保存)
        if (epoch + 1) % 5 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # スケジューラの状態を保存
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss
            }
            torch.save(checkpoint, f'checkpoints/checkpoint_{epoch+1}.pth')
            
            # チェックポイント保存時に推論を実行
            inference_at_checkpoint(model, val_dataset, epoch + 1)

    print("Training finished!")

if __name__ == '__main__':
    train()