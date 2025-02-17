import torch
from model import UNet
from PIL import Image
from torchvision import transforms
from constants import IMAGE_SIZE
import os

def infer(input_image_path, output_image_path, model_path='checkpoints/model_epoch_100.pth'): # モデルパスを適宜修正
    """
    推論を実行し、ノイズ除去された画像を保存する

    Args:
        input_image_path (str): 入力画像 (ノイズ画像) のパス
        output_image_path (str): 出力画像 (ノイズ除去画像) の保存パス
        model_path (str): 学習済みモデルのパス
    """
    # モデル準備
    model = UNet().to('cuda') # モデルをGPUに転送
    model.load_state_dict(torch.load(model_path)) # 学習済みパラメータをロード
    model.eval() # 評価モードに設定

    # 画像読み込みと前処理 (アスペクト比を維持したリサイズとパディング)
    input_image = Image.open(input_image_path).convert('RGB')
    
    def get_padding_size(original_size):
        width, height = original_size
        # 長辺を基準にスケールを計算
        scale = IMAGE_SIZE / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        # パディングサイズを計算
        pad_width = IMAGE_SIZE - new_width
        pad_height = IMAGE_SIZE - new_height
        # 左右上下のパディングを均等に
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        return pad_left, pad_top, pad_right, pad_bottom

    transform = transforms.Compose([
        transforms.Lambda(lambda img: transforms.functional.resize(img, 
            (int(img.size[1] * IMAGE_SIZE / max(img.size)), 
             int(img.size[0] * IMAGE_SIZE / max(img.size))))),
        transforms.Lambda(lambda img: transforms.functional.pad(img, 
            get_padding_size(img.size), 
            padding_mode='constant', 
            fill=0)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to('cuda') # バッチ次元を追加してGPUに転送

    # 推論実行
    with torch.no_grad(): # 勾配計算を無効化
        output_tensor = model(input_tensor)

    # 後処理 (パディングを除去してオリジナルのアスペクト比に戻す)
    output_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1) # バッチ次元削除, CPUへ転送, 値域を [0, 1] にクリップ
    output_image_tensor = output_tensor
    output_image_pil = transforms.ToPILImage()(output_image_tensor)
    
    # オリジナルのサイズにリサイズして保存
    # output_image_pil = output_image_pil.resize(input_image.size, Image.LANCZOS)
    output_image_pil.save(output_image_path)
    print(f"Processed image saved to: {output_image_path}")

if __name__ == '__main__': 
    # 推論実行 (例)
    input_path = 'dataset/train/noisy/4e128d30514c2777bc1cb29bdd6f8b12.png'   # 入力画像パスを修正
    output_path = 'output/denoised_image.png'      # 出力画像パスを修正
    os.makedirs('output', exist_ok=True) # outputディレクトリを作成
    infer(input_path, output_path)