from torch.utils.data import DataLoader


def collate_fn(batch):
    return tuple(zip(*batch))

def data_loader(train_dataset, val_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2
    )

    print("\n✅ 모든 준비 완료! YOLO 학습에 사용할 train/val 분할 완성")
    return train_loader, val_loader