import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class EyeSegmentationModel(nn.Module):
    def __init__(self, n_classes=4): # 0: фон, 1: радужка, 2: зрачок
        super(EyeSegmentationModel, self).__init__()
        # Используем архитектуру Unet - она лучшая для закрашивания (сегментации)
        self.model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights="imagenet",
            in_channels=3, # Работаем с обычным цветным/чб фото
            classes=n_classes,
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = EyeSegmentationModel()
    print("Архитектура сегментационной нейросети успешно создана!")
