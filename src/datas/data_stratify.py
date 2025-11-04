import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.model_selection import iterative_train_test_split
from src.datas.PillDataset import PillDataset


def data_stratify(images_df, annotations_df, categories_df, train_transform, val_transform, TRAIN_IMG_DIR):

    # 1-1. 모든 이미지 ID
    all_image_ids = images_df['id'].unique()

    # 1-2. 이미지별 포함된 category_id 리스트 생성
    img_to_cats = annotations_df.groupby('image_id')['category_id'].apply(list).to_dict()

    # 1-3. 전체 이미지 순서에 맞춰 라벨 리스트 생성 (없으면 빈 리스트)
    labels_list = [img_to_cats.get(img_id, []) for img_id in all_image_ids]

    # 1-4. MultiLabelBinarizer로 (이미지 수 × 클래스 수) 행렬 생성
    mlb = MultiLabelBinarizer()
    y_multilabel = mlb.fit_transform(labels_list)
    print(f"  - y 행렬 생성 완료 (형태: {y_multilabel.shape})")

    # 1-5. X는 단순히 이미지 인덱스로 설정
    X_indices = np.arange(len(images_df)).reshape(-1, 1)

    # 1-6. 계층화 분할 (skmultilearn)
    print("✂️ 계층화 분할 실행 중 (skmultilearn)...")
    np.random.seed(42)
    X_train_idx, y_train, X_val_idx, y_val = iterative_train_test_split(
        X_indices,
        y_multilabel,
        test_size=0.1
    )


    train_indices = X_train_idx.flatten()
    val_indices = X_val_idx.flatten()

    train_images_df = images_df.iloc[train_indices].reset_index(drop=True)
    val_images_df = images_df.iloc[val_indices].reset_index(drop=True)

    train_ids = set(train_images_df['id'])
    val_ids = set(val_images_df['id'])

    train_annotations_df = annotations_df[annotations_df['image_id'].isin(train_ids)]
    val_annotations_df = annotations_df[annotations_df['image_id'].isin(val_ids)]

    print(f"✅ 초기 분할 완료!")
    print(f"  - Train 이미지: {len(train_images_df)}장")
    print(f"  - Val 이미지:   {len(val_images_df)}장")

    # ==============================
    # 3️⃣ Validation 최소 1장 보정
    # ==============================
    print("\n🔧 Validation 클래스 최소 1장 보정 중...")

    # 3-1. Validation에 없는 클래스 찾기
    all_classes = categories_df['id'].tolist()
    val_present = val_annotations_df['category_id'].unique().tolist()
    zero_val_classes = [c for c in all_classes if c not in val_present]

    moved_images = set()

    # 3-2. 없는 클래스가 있으면 train → val로 이동
    for cat_id in zero_val_classes:
        candidate_imgs = train_annotations_df[train_annotations_df["category_id"] == cat_id]["image_id"].unique()
        if len(candidate_imgs) == 0:
            continue  # 혹시 해당 클래스 이미지 자체가 없으면 스킵
        chosen_img = np.random.choice(candidate_imgs, 1)[0]
        moved_images.add(chosen_img)

    # 3-3. 실제 이동 적용
    if moved_images:
        print(f"  - {len(moved_images)}개의 이미지 이동 (Validation에 없는 클래스 보정)")

        moved_df = train_images_df[train_images_df["id"].isin(moved_images)]

        # train → val 이동
        train_images_df = train_images_df[~train_images_df["id"].isin(moved_images)].reset_index(drop=True)
        val_images_df = pd.concat([val_images_df, moved_df], ignore_index=True)

        # annotations 갱신 (전체 이미지 기준으로 다시 필터)
        train_annotations_df = annotations_df[annotations_df["image_id"].isin(train_images_df["id"])]
        val_annotations_df = annotations_df[annotations_df["image_id"].isin(val_images_df["id"])]

    else:
        print("  - 모든 클래스가 이미 Validation에 최소 1장 이상 포함되어 있음 ✅")

    print(f"  - 최종 Train 이미지: {len(train_images_df)}장")
    print(f"  - 최종 Val 이미지:   {len(val_images_df)}장")

    # ==============================
    # 4️⃣ (선택) Dataset/DataLoader (YOLO 학습에는 불필요)
    # ==============================
    train_dataset = PillDataset(
        TRAIN_IMG_DIR,
        train_images_df,
        train_annotations_df,
        categories_df,
        transform=train_transform  # 🚨 YOLO 학습엔 필요 X
    )

    val_dataset = PillDataset(
        TRAIN_IMG_DIR,
        val_images_df,
        val_annotations_df,
        categories_df,
        transform=val_transform  # 🚨 YOLO 학습엔 필요 X
    )
    return train_dataset, val_dataset, train_images_df, val_images_df, train_annotations_df, val_annotations_df
