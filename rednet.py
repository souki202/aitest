import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Skip connection (identity mapping or 1x1 conv for channel/stride adjustment)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity() # Identity mapping if no adjustment needed

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual) # Skip connection: add shortcut to output
        out = self.relu(out) # ReLU after skip connection
        return out


class REDNet(nn.Module):
    def __init__(self, num_channels=3, num_filters=64, num_residual_blocks=4): # Residual Block数を調整可能に
        """
        RED-Net (Residual Encoder-Decoder Network) モデル

        Args:
            num_channels (int): 入力画像のチャンネル数 (RGBの場合は3)
            num_filters (int): 基本となるフィルタ数
            num_residual_blocks (int): 各Encoder/Decoderブロック内のResidual Block数
        """
        super(REDNet, self).__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters
        self.num_residual_blocks = num_residual_blocks

        # Encoder (Downsampling Path)
        self.enc_conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.enc_bn1 = nn.BatchNorm2d(num_filters)
        self.enc_relu1 = nn.ReLU(inplace=True)

        # プーリング層の定義を__init__に追加
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder path with consistent channel numbers
        self.enc_res_blocks1 = self._make_residual_blocks(num_residual_blocks, num_filters, num_filters)
        self.enc_res_blocks2 = self._make_residual_blocks(num_residual_blocks, num_filters, num_filters * 2)
        self.enc_res_blocks3 = self._make_residual_blocks(num_residual_blocks, num_filters * 2, num_filters * 4)

        # Bottleneck with consistent channel numbers
        self.bottleneck_res_blocks = self._make_residual_blocks(num_residual_blocks, num_filters * 4, num_filters * 8)

        # Decoder path with consistent channel numbers
        self.upconv1 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2, bias=False)
        self.dec_bn1 = nn.BatchNorm2d(num_filters * 4)
        self.dec_relu1 = nn.ReLU(inplace=True)
        self.dec_conv1 = nn.Conv2d(num_filters * 8, num_filters * 4, kernel_size=1)  # 1x1 conv for channel reduction
        self.dec_res_blocks1 = self._make_residual_blocks(num_residual_blocks, num_filters * 4, num_filters * 4)

        self.upconv2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2, bias=False)
        self.dec_bn2 = nn.BatchNorm2d(num_filters * 2)
        self.dec_relu2 = nn.ReLU(inplace=True)
        self.dec_conv2 = nn.Conv2d(num_filters * 4, num_filters * 2, kernel_size=1)  # 1x1 conv for channel reduction
        self.dec_res_blocks2 = self._make_residual_blocks(num_residual_blocks, num_filters * 2, num_filters * 2)

        self.upconv3 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=2, stride=2, bias=False)
        self.dec_bn3 = nn.BatchNorm2d(num_filters)
        self.dec_relu3 = nn.ReLU(inplace=True)
        self.dec_conv3 = nn.Conv2d(num_filters * 2, num_filters, kernel_size=1)  # 1x1 conv for channel reduction
        self.dec_res_blocks3 = self._make_residual_blocks(num_residual_blocks, num_filters, num_filters)

        # Output Layer
        self.output_conv = nn.Conv2d(num_filters, num_channels, kernel_size=1) # 1x1 conv for channel reduction

    def _make_residual_blocks(self, num_blocks, in_channels, out_channels=None, stride=1):
        """
        Residual Blockを複数生成するヘルパー関数

        Args:
            num_blocks (int): 生成するResidual Blockの数
            in_channels (int): 入力チャンネル数
            out_channels (int, optional): 出力チャンネル数 (Noneの場合は in_channels と同じ). Defaults to None.
            stride (int, optional): 最初のResidual Blockのstride. Defaults to 1.

        Returns:
            nn.Sequential: Residual BlockをSequentialにまとめたもの
        """
        if out_channels is None:
            out_channels = in_channels # デフォルトは入力チャンネル数と同じ
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride)) # 最初のResidual Blockはstride指定可能
        for _ in range(1, num_blocks): # 2個目以降はstride=1
            layers.append(ResidualBlock(out_channels, out_channels)) # チャンネル数は固定
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        enc = self.enc_conv1(x)
        enc = self.enc_bn1(enc)
        enc1 = self.enc_relu1(enc) # enc1: Conv + BN + ReLU

        enc1_out = self.enc_res_blocks1(enc1) # Residual Blocks x num_residual_blocks
        enc_pooled = self.pool1(enc1_out) # Pooling後の出力

        enc2_out = self.enc_res_blocks2(enc_pooled) # Residual Blocks x num_residual_blocks
        enc_pooled = self.pool2(enc2_out)

        enc3_out = self.enc_res_blocks3(enc_pooled) # Residual Blocks x num_residual_blocks
        enc_pooled = self.pool3(enc3_out)

        # Bottleneck (Residual Blocks)
        bottleneck = self.bottleneck_res_blocks(enc_pooled) # Residual Blocks x num_residual_blocks

        # Decoder with Skip Connections and Channel Reduction
        dec1 = self.upconv1(bottleneck)
        dec1 = self.dec_bn1(dec1)
        dec1 = self.dec_relu1(dec1)
        dec1 = torch.cat([dec1, enc3_out], dim=1)
        dec1 = self.dec_conv1(dec1)  # Channel reduction after concatenation
        dec1_out = self.dec_res_blocks1(dec1)

        dec2 = self.upconv2(dec1_out)
        dec2 = self.dec_bn2(dec2)
        dec2 = self.dec_relu2(dec2)
        dec2 = torch.cat([dec2, enc2_out], dim=1)
        dec2 = self.dec_conv2(dec2)  # Channel reduction after concatenation
        dec2_out = self.dec_res_blocks2(dec2)

        dec3 = self.upconv3(dec2_out)
        dec3 = self.dec_bn3(dec3)
        dec3 = self.dec_relu3(dec3)
        dec3 = torch.cat([dec3, enc1_out], dim=1)
        dec3 = self.dec_conv3(dec3)  # Channel reduction after concatenation
        dec3_out = self.dec_res_blocks3(dec3)

        # Output Layer
        output = self.output_conv(dec3_out) # No activation function at the output layer
        return output


if __name__ == '__main__':
    # モデルの動作確認 (例)
    model = REDNet(num_channels=3, num_filters=64, num_residual_blocks=4) # Residual Block数を指定
    dummy_input = torch.randn(1, 3, 256, 256) # バッチサイズ1, チャンネル数3, 高さ256, 幅256
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # torch.Size([1, 3, 256, 256]) となるはず