import shutil
from tqdm import tqdm
import os
# 이미지 복사
from globals import TRAIN_IMG_DIR

def convert_to_yolo_format(img_info, annotations, save_img_dir, save_label_dir, category_id_mapping):
    img_id = img_info['id']
    file_name = img_info['file_name']
    img_w = img_info['width']
    img_h = img_info['height']


    src_img_path = os.path.join(TRAIN_IMG_DIR, file_name)
    dst_img_path = os.path.join(save_img_dir, file_name)

    if os.path.exists(src_img_path):
        shutil.copy(src_img_path, dst_img_path)
    else:
        return False

    # 라벨 파일 생성
    label_file = os.path.join(save_label_dir, file_name.replace('.png', '.txt').replace('.jpg', '.txt'))

    img_annotations = annotations[annotations['image_id'] == img_id]

    with open(label_file, 'w') as f:
        for _, ann in img_annotations.iterrows():
            # COCO bbox: [x, y, width, height]
            x, y, w, h = ann['bbox']
            category_id = ann['category_id']

            # YOLO 형식으로 변환: [class_id, x_center, y_center, width, height] (normalized)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            # 카테고리 ID를 0부터 시작하도록 변환
            class_id = category_id_mapping[category_id]

            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    return True

