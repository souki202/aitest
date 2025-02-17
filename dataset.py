import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from constants import DATASET_DIR, IMAGE_SIZE

class NoiseReductionDataset(Dataset):
    def __init__(self, root_dir, image_size=(IMAGE_SIZE, IMAGE_SIZE), train=True):
        """
        データセットクラス

        Args:
            root_dir (str): データセットのルートディレクトリ (train/ または val/ ディレクトリを含む)
            image_size (tuple): リサイズ後の画像サイズ (height, width), デフォルトは定数で指定
            train (bool): 学習データセットかどうか (True: 学習, False: 検証)
        """
        self.root_dir = root_dir
        self.image_size = image_size
        self.train = train

        if train:
            self.input_dir = os.path.join(root_dir, 'train', 'noisy')  # ノイズ画像ディレクトリ
            self.target_dir = os.path.join(root_dir, 'train', 'clean') # 教師画像ディレクトリ
        else:
            self.input_dir = os.path.join(root_dir, 'val', 'noisy')
            self.target_dir = os.path.join(root_dir, 'val', 'clean')

        self.input_images = sorted([f for f in os.listdir(self.input_dir) if os.path.isfile(os.path.join(self.input_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.target_images = sorted([f for f in os.listdir(self.target_dir) if os.path.isfile(os.path.join(self.target_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # データ変換
        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self._resize_with_padding(img, image_size)),
            transforms.ToTensor(),
        ])

    def _resize_with_padding(self, image, target_size):
        """アスペクト比を維持しながらリサイズし、余白を黒で埋める"""
        # 元画像のサイズとアスペクト比を取得
        orig_width, orig_height = image.size
        orig_aspect = orig_width / orig_height
        target_width, target_height = target_size
        target_aspect = target_width / target_height

        # リサイズ後のサイズを計算
        if target_aspect > orig_aspect:
            # ターゲットの方が横長: 高さに合わせてリサイズ
            new_height = target_height
            new_width = int(new_height * orig_aspect)
        else:
            # ターゲットの方が縦長: 幅に合わせてリサイズ
            new_width = target_width
            new_height = int(new_width / orig_aspect)

        # アスペクト比を維持してリサイズ
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # 黒背景の新しい画像を作成
        padded_image = Image.new('RGB', target_size, (0, 0, 0))
        
        # リサイズした画像を中央に配置
        paste_x = (target_width - new_width) // 2
        paste_y = (target_height - new_height) // 2
        padded_image.paste(resized_image, (paste_x, paste_y))

        return padded_image

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        target_path = os.path.join(self.target_dir, self.target_images[idx])

        input_image = Image.open(input_path).convert('RGB') # RGBに変換
        target_image = Image.open(target_path).convert('RGB')

        input_tensor = self.transform(input_image)
        target_tensor = self.transform(target_image)

        return input_tensor, target_tensor

if __name__ == '__main__':
    # データセットの動作確認 (例)
    dataset = NoiseReductionDataset(DATASET_DIR, image_size=(IMAGE_SIZE, IMAGE_SIZE), train=True)
    print(f"Dataset size: {len(dataset)}")
    input_tensor, target_tensor = dataset[0]
    print(f"Input tensor shape: {input_tensor.shape}, Target tensor shape: {target_tensor.shape}")