# 토치비전 v2 상위호환
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

def transforms():
    # Train Transform (데이터 증강 포함)
    train_transform = A.Compose([
        A.Resize(800, 800),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        # A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.Blur(blur_limit=3, p=0.2),
        A.Rotate(limit=(180, 180), p=0.3, border_mode=cv2.BORDER_REFLECT_101), # 어차피 180도 잘리는 부분 X

        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))


    # Validation Transform (증강 없음)
    val_transform = A.Compose([
        A.Resize(800, 800),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))

    return train_transform, val_transform