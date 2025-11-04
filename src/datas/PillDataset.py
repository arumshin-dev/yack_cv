import torch
import os
import cv2
from torch.utils.data import Dataset

class PillDataset(Dataset):
    def __init__(self, img_dir, images_df, annotations_df, categories_df, transform=None):
        self.img_dir = img_dir
        self.images_df = images_df.reset_index(drop=True)
        self.annotations_df = annotations_df
        self.categories_df = categories_df
        self.transform = transform

    def __len__(self):
        return len(self.images_df)

    def __getitem__(self, idx):
        img_info = self.images_df.iloc[idx]
        img_id = img_info['id']
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        # 이미지 로드
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Annotation
        img_annotations = self.annotations_df[self.annotations_df['image_id'] == img_id]

        boxes = []
        labels = []
        for _, ann in img_annotations.iterrows():
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x+w, y+h])
            labels.append(ann['category_id'])

        # Transform 적용
        if self.transform and len(boxes) > 0:
            transformed = self.transform(
                image=img,
                bboxes=boxes,
                labels=labels
            )
            img = transformed['image']
            boxes = list(transformed['bboxes'])
            labels = list(transformed['labels'])
        elif self.transform:
            transformed = self.transform(image=img, bboxes=[], labels=[])
            img = transformed['image']
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Tensor 변환
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([img_id])
        }

        return img, target