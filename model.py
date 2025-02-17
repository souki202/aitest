import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=3, num_filters=32):
        """
        シンプルなCNNモデル

        Args:
            num_channels (int): 入力画像のチャンネル数 (RGBの場合は3)
            num_filters (int): 畳み込み層のフィルタ数 (基本となるフィルタ数)
        """
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(num_filters, num_channels, kernel_size=3, padding=1) # 出力は入力と同じチャンネル数

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x) # 最終層は活性化関数なし (線形出力)
        return x

if __name__ == '__main__':
    # モデルの動作確認 (例)
    model = SimpleCNN(num_channels=3, num_filters=32)
    dummy_input = torch.randn(1, 3, 256, 256) # バッチサイズ1, チャンネル数3, 高さ256, 幅256
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # torch.Size([1, 3, 256, 256]) となるはず