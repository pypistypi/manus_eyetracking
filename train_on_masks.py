import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from segmentation_model import EyeSegmentationModel

# --- НАСТРОЙКИ ---
BASE_DIR = 'datasets/big_dataset'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4  # Уменьшил для стабильности, можно увеличить до 8 если памяти много
EPOCHS = 50


# -----------------

class BigEyeDataset(Dataset):
    def __init__(self, base_dir):
        self.img_dir = os.path.join(base_dir, 'images')
        self.mask_i_dir = os.path.join(base_dir, 'masks_i')
        self.mask_p_dir = os.path.join(base_dir, 'masks_p')
        # Берем только файлы изображений .jpg
        self.images = [f for f in os.listdir(self.img_dir) if f.lower().endswith('.jpg')]
        print(f"Найдено {len(self.images)} изображений для обучения.")

    def __getitem__(self, index):
        img_name = self.images[index]
        base_name = os.path.splitext(img_name)[0]

        # 1. Загружаем фото глаза (.jpg)
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось прочитать изображение: {img_path}")
        img = cv2.resize(img, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 2. Загружаем маски (.png)
        # Пробуем найти маски с расширением .png
        path_i = os.path.join(self.mask_i_dir, f"{base_name}_i.png")
        path_p = os.path.join(self.mask_p_dir, f"{base_name}_p.png")

        mask_i = cv2.imread(path_i, 0)
        mask_p = cv2.imread(path_p)

        if mask_i is None or mask_p is None:
            # Если не нашли .png, попробуем .jpg на всякий случай
            mask_i = cv2.imread(path_i.replace('.png', '.jpg'), 0)
            mask_p = cv2.imread(path_p.replace('.png', '.jpg'))

        if mask_i is None or mask_p is None:
            raise FileNotFoundError(f"Маски для {base_name} не найдены (проверены .png и .jpg)")

        mask_i = cv2.resize(mask_i, (256, 256))
        mask_p = cv2.resize(mask_p, (256, 256))

        # 3. Создаем итоговую маску классов
        final_mask = np.zeros((256, 256), dtype=np.uint8)

        # Радужка (1) - где маска_i светлая
        final_mask[mask_i > 10] = 1

        # Зрачок (2) - где маска_p имеет значимую яркость
        mask_p_gray = cv2.cvtColor(mask_p, cv2.COLOR_BGR2GRAY)
        final_mask[mask_p_gray > 30] = 2

        # Подготовка тензоров
        img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        return torch.tensor(img), torch.tensor(final_mask, dtype=torch.long)

    def __len__(self):
        return len(self.images)


def train_big():
    dataset = BigEyeDataset(BASE_DIR)
    if len(dataset) == 0:
        print("Ошибка: Датасет пуст! Проверьте пути к папкам.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Создаем модель (3 класса: фон, радужка, зрачок)
    model = EyeSegmentationModel(n_classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"Запуск обучения на {DEVICE}...")
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

        print(f"Эпоха [{epoch + 1}/{EPOCHS}], Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "pretrained_eye_model.pth")
    print("Обучение завершено! Модель сохранена как pretrained_eye_model.pth")


if __name__ == "__main__":
    train_big()
