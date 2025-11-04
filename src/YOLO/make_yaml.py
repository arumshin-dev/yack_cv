# 클래스 이름 리스트 생성
from globals import BASE_DIR
import yaml

YOLO_DIR = f"{BASE_DIR}/yolo_dataset"

def make_yaml(categories_df):
    class_names = []
    for cat_id in sorted(categories_df['id'].unique()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        class_names.append(cat_name)

    # 카테고리 ID를 0부터 시작하도록 매핑
    category_id_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted(categories_df['id'].unique()))}
    num_classes = len(category_id_mapping)

    # data.yaml 내용
    data_yaml = {
        'path': YOLO_DIR,
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names
    }

    # 저장
    yaml_path = f"{YOLO_DIR}/data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)