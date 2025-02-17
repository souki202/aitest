import torch
from model import SimpleCNN
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
    model = SimpleCNN().to('cuda') # モデルをGPUに転送
    model.load_state_dict(torch.load(model_path)) # 学習済みパラメータをロード
    model.eval() # 評価モードに設定

    # 画像読み込みと前処理 (学習時と同じリサイズとテンソル変換)
    input_image = Image.open(input_image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), # 定数を使用
        transforms.ToTensor(),
    ])
    input_tensor = transform(input_image).unsqueeze(0).to('cuda') # バッチ次元を追加してGPUに転送

    # 推論実行
    with torch.no_grad(): # 勾配計算を無効化
        output_tensor = model(input_tensor)

    # 後処理 (テンソルからPIL Imageへ変換, 値域を [0, 255] に戻す)
    output_image_tensor = output_tensor.squeeze(0).cpu().clamp(0, 1) # バッチ次元削除, CPUへ転送, 値域を [0, 1] にクリップ
    output_image_pil = transforms.ToPILImage()(output_image_tensor) # テンソルをPIL Imageに変換

    # 画像保存
    output_image_pil.save(output_image_path)
    print(f"Processed image saved to: {output_image_path}")

if __name__ == '__main__': 
    # 推論実行 (例)
    input_path = 'path/to/your/noisy_image.png'   # 入力画像パスを修正
    output_path = 'output/denoised_image.png'      # 出力画像パスを修正
    os.makedirs('output', exist_ok=True) # outputディレクトリを作成
    infer(input_path, output_path)