import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from segmentation_model import EyeSegmentationModel  # Импорт вашей модели

# --- НАСТРОЙКИ ---
DATA_DIR = 'datasets/segmentation/images'
ANN_FILE = 'datasets/segmentation/annotations/labels_iris+pypil_2026-03-18-01-52-14.json'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPOCHS = 50


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

        # Загрузка изображения
        img_path = os.path.join(self.img_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Создание пустой маски (0 - фон)
        mask = np.zeros((256, 256), dtype=np.uint8)

        for ann in anns:
            # category_id: 1 - iris, 2 - pupil (согласно вашему JSON)
            cat_id = ann['category_id']
            # Рисуем полигон на маске
            for seg in ann['segmentation']:
                poly = np.array(seg).reshape((len(seg) // 2, 2)).astype(np.int32)
                cv2.fillPoly(mask, [poly], cat_id)

        # Подготовка для нейросети
        image = image.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(image), torch.tensor(mask, dtype=torch.long)

    def __len__(self):
        return len(self.ids)


def train():
    dataset = EyeDataset(DATA_DIR, ANN_FILE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EyeSegmentationModel(n_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Начинаем обучение на {DEVICE}...")
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

        if (epoch + 1) % 10 == 0:
            print(f"Эпоха [{epoch + 1}/{EPOCHS}], Потеря: {total_loss / len(loader):.4f}")

    torch.save(model.state_state_dict(), "eye_segmentation_best.pth")
    print("Обучение завершено! Модель сохранена как eye_segmentation_best.pth")


if __name__ == "__main__":
    train()
