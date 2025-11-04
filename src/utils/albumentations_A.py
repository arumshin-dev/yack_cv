import albumentations as A
from albumentations.pytorch import ToTensorV2


# Train Transform (데이터 증강 포함)
def train_compose():
    train_transform = A.Compose([
        # 크기 조정
        A.Resize(800, 800),

        # 데이터 증강
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.2),

        # Normalization
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return train_transform

# Validation Transform (증강 없음)
def val_compose():
    val_transform = A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    return val_transform