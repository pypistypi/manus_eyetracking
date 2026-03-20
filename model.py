import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class SiameseEyeModel(nn.Module):
    def __init__(self, n_classes=4):
        super(SiameseEyeModel, self).__init__()
        # Используем предобученный Unet с MobileNetV3
        self.base_model = smp.Unet(
            encoder_name="mobilenet_v3_small",
            encoder_weights="imagenet",
            in_channels=1,  # Каждый энкодер принимает по 1 каналу (ч/б)
            classes=n_classes
        )

        # Модифицируем вход первого слоя, чтобы принимать 2 канала (яркий + темный)
        # Это и делает сеть "Сиамской" на входе
        original_conv = self.base_model.encoder.conv_stem
        self.base_model.encoder.conv_stem = nn.Conv2d(
            2, original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

    def forward(self, bright_img, dark_img):
        # Склеиваем два изображения в один тензор с 2 каналами
        x = torch.cat([bright_img, dark_img], dim=1)
        return self.base_model(x)
