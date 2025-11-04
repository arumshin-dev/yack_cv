from tqdm import tqdm
from globals import BASE_DIR
YOLO_DIR = f"{BASE_DIR}/yolo_dataset"
from src.YOLO.convert_to_yolo_format import convert_to_yolo_format


def convert_data(train_images_df, train_annotations_df, val_images_df, val_annotations_df, category_id_mapping):
    print("📝 Train 데이터 변환 중...")
    train_success = 0
    for _, img_info in tqdm(train_images_df.iterrows(), total=len(train_images_df)):
        if convert_to_yolo_format(
            img_info,
            train_annotations_df,
            f"{YOLO_DIR}/images/train",
            f"{YOLO_DIR}/labels/train",
            category_id_mapping
        ):
            train_success += 1

    print(f"\n Train 데이터 변환 완료: {train_success}/{len(train_images_df)}개")

    # Val 데이터 변환
    print("\n📝 Val 데이터 변환 중...")
    val_success = 0
    for _, img_info in tqdm(val_images_df.iterrows(), total=len(val_images_df)):
        if convert_to_yolo_format(
            img_info,
            val_annotations_df,
            f"{YOLO_DIR}/images/val",
            f"{YOLO_DIR}/labels/val",
            category_id_mapping
        ):
            val_success += 1

    print(f"✅ Val 데이터 변환 완료: {val_success}/{len(val_images_df)}개")
