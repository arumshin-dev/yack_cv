import os
from globals import BASE_DIR
def make_yolo_dir(categories_df):

    YOLO_DIR = f"{BASE_DIR}/yolo_dataset"
    os.makedirs(f"{YOLO_DIR}/images/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/images/val", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/train", exist_ok=True)
    os.makedirs(f"{YOLO_DIR}/labels/val", exist_ok=True)

    print("✅ YOLO 폴더 구조 생성 완료!")

    # 카테고리 ID를 0부터 시작하도록 매핑
    category_id_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted(categories_df['id'].unique()))}
    num_classes = len(category_id_mapping)

    print(f"📊 총 클래스 수: {num_classes}개")
    print(f"카테고리 매핑 (처음 5개): {dict(list(category_id_mapping.items())[:5])}")

    return category_id_mapping, num_classes