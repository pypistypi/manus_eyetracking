import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from segmentation_model import EyeSegmentationModel

# --- НАСТРОЙКИ ---
DATA_DIR = 'datasets/segmentation/images'
ANN_FILE = 'datasets/segmentation/annotations/labels_iris+pypil_2026-03-18-01-52-14.json'
PRETRAINED_PATH = 'pretrained_eye_model.pth'  # Путь к вашей обученной модели
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 30  # Для дообучения достаточно 30 эпох


# -----------------

class EyeDataset(Dataset):
    def __init__(self, img_dir, ann_file):
        self.img_dir = img_dir
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        img_info = self.coco.loadImgs(img_id)[0]

        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Не найден файл: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = np.zeros((256, 256), dtype=np.uint8)
        for ann in anns:
            cat_id = ann['category_id']  # 1 - iris, 2 - pupil
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], cat_id)

        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(image), torch.tensor(mask, dtype=torch.long)

    def __len__(self):
        return len(self.ids)


def fine_tune():
    dataset = EyeDataset(DATA_DIR, ANN_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 1. Создаем модель
    model = EyeSegmentationModel(n_classes=3).to(DEVICE)

    # 2. ЗАГРУЖАЕМ ПРЕДОБУЧЕННЫЕ ВЕСА
    print(f"Загрузка базовой модели из {PRETRAINED_PATH}...")
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))

    # 3. Настройка обучения
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Меньший шаг обучения для доводки
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Начинаем дообучение на 63 ручных снимках...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 5 == 0:
            print(f"Эпоха [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "eye_segmentation_final.pth")
    print("Финальная модель сохранена как eye_segmentation_final.pth!")


if __name__ == "__main__":
    fine_tune()
