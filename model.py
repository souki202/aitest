import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_channels=3, num_filters=64):
        """
        U-Net モデル

        Args:
            num_channels (int): 入力画像のチャンネル数 (RGBの場合は3)
            num_filters (int): 基本となるフィルタ数 (Encoderの最初の畳み込み層のフィルタ数)
        """
        super(UNet, self).__init__()
        self.num_channels = num_channels
        self.num_filters = num_filters

        # Encoder (Downsampling Path)
        self.enc1_conv = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.enc1_relu = nn.ReLU(inplace=True)
        self.enc2_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.enc2_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # サイズを1/2に

        self.enc3_conv = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, padding=1) # フィルタ数2倍
        self.enc3_relu = nn.ReLU(inplace=True)
        self.enc4_conv = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, padding=1)
        self.enc4_relu = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5_conv = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=3, padding=1) # フィルタ数さらに2倍
        self.enc5_relu = nn.ReLU(inplace=True)
        self.enc6_conv = nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, padding=1)
        self.enc6_relu = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = nn.Conv2d(num_filters * 4, num_filters * 8, kernel_size=3, padding=1) # フィルタ数さらに2倍
        self.bottleneck_relu = nn.ReLU(inplace=True)

        # Decoder (Upsampling Path)
        self.upconv1 = nn.ConvTranspose2d(num_filters * 8, num_filters * 4, kernel_size=2, stride=2) # サイズを2倍に
        self.dec1_conv = nn.Conv2d(num_filters * 8, num_filters * 4, kernel_size=3, padding=1) # Skip connectionからの入力を考慮して入力チャンネル数 * 2
        self.dec1_relu = nn.ReLU(inplace=True)
        self.dec2_conv = nn.Conv2d(num_filters * 4, num_filters * 4, kernel_size=3, padding=1)
        self.dec2_relu = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(num_filters * 4, num_filters * 2, kernel_size=2, stride=2)
        self.dec3_conv = nn.Conv2d(num_filters * 4, num_filters * 2, kernel_size=3, padding=1) # Skip connectionからの入力を考慮
        self.dec3_relu = nn.ReLU(inplace=True)
        self.dec4_conv = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, padding=1)
        self.dec4_relu = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(num_filters * 2, num_filters, kernel_size=2, stride=2)
        self.dec5_conv = nn.Conv2d(num_filters * 2, num_filters, kernel_size=3, padding=1) # Skip connectionからの入力を考慮
        self.dec5_relu = nn.ReLU(inplace=True)
        self.dec6_conv = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.dec6_relu = nn.ReLU(inplace=True)

        # Output Layer
        self.output_conv = nn.Conv2d(num_filters, num_channels, kernel_size=1) # 1x1 conv for channel reduction

    def forward(self, x):
        # Encoder
        enc1 = self.enc1_conv(x)
        enc1 = self.enc1_relu(enc1)
        enc1_out = self.enc2_conv(enc1) # Skip connection用
        enc1_out = self.enc2_relu(enc1_out)
        enc_pooled = self.pool1(enc1_out) # Pooling後の出力

        enc2 = self.enc3_conv(enc_pooled)
        enc2 = self.enc3_relu(enc2)
        enc2_out = self.enc4_conv(enc2) # Skip connection用
        enc2_out = self.enc4_relu(enc2_out)
        enc_pooled = self.pool2(enc2_out)

        enc3 = self.enc5_conv(enc_pooled)
        enc3 = self.enc5_relu(enc3)
        enc3_out = self.enc6_conv(enc3) # Skip connection用
        enc3_out = self.enc6_relu(enc3_out)
        enc_pooled = self.pool3(enc3_out)

        # Bottleneck
        bottleneck = self.bottleneck_conv(enc_pooled)
        bottleneck = self.bottleneck_relu(bottleneck)

        # Decoder with Skip Connections
        dec1 = self.upconv1(bottleneck)
        dec1 = torch.cat([dec1, enc3_out], dim=1) # Skip connection (Encoder layer 3 output)
        dec1 = self.dec1_conv(dec1)
        dec1 = self.dec1_relu(dec1)
        dec1 = self.dec2_conv(dec1)
        dec1 = self.dec2_relu(dec1)

        dec2 = self.upconv2(dec1)
        dec2 = torch.cat([dec2, enc2_out], dim=1) # Skip connection (Encoder layer 2 output)
        dec2 = self.dec3_conv(dec2)
        dec2 = self.dec3_relu(dec2)
        dec2 = self.dec4_conv(dec2)
        dec2 = self.dec4_relu(dec2)

        dec3 = self.upconv3(dec2)
        dec3 = torch.cat([dec3, enc1_out], dim=1) # Skip connection (Encoder layer 1 output)
        dec3 = self.dec5_conv(dec3)
        dec3 = self.dec5_relu(dec3)
        dec3 = self.dec6_conv(dec3)
        dec3 = self.dec6_relu(dec3)

        # Output Layer
        output = self.output_conv(dec3) # No activation function at the output layer
        return output


if __name__ == '__main__':
    # モデルの動作確認 (例)
    model = UNet(num_channels=3, num_filters=64)
    dummy_input = torch.randn(1, 3, 256, 256) # バッチサイズ1, チャンネル数3, 高さ256, 幅256
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # torch.Size([1, 3, 256, 256]) となるはず