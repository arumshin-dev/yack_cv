"""
util 함수를 모아놓은 파일
"""
import os
import shutil

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

import globals

# Bounding Box 시각화
def visualize_annotations(train_img_dir,
                          images_df,
                          annotations_df,
                          categories_df,
                          img_id,
                          figsize=(8, 8)
                          ):
    """특정 이미지에 바운딩 박스를 그려서 시각화"""

    # 이미지 정보 가져오기
    img_info = images_df[images_df['id'] == img_id].iloc[0]
    img_path = os.path.join(train_img_dir, img_info['file_name'])

    # 이미지가 존재하는지 확인
    if not os.path.exists(img_path):
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
        return

    # 이미지 로드
    img = mpimg.imread(img_path)

    # 해당 이미지의 annotation 가져오기
    img_annotations = annotations_df[annotations_df['image_id'] == img_id]

    if len(img_annotations) == 0:
        print(f"⚠️ Image ID {img_id}에 annotation이 없습니다.")
        return

    # 시각화 (크기 축소)
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(img)

    # 색상 리스트
    colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta', 'orange', 'purple']

    print(f"🔍 Image ID {img_id} - {len(img_annotations)}개 객체 검출:")

    # 폰트 경로 지정 (윈도우 기본 폰트 폴더)
    font_path = globals.FONT_PATH

    # FontProperties 객체 생성
    font_prop = fm.FontProperties(fname=font_path, size=15)

    for idx, (_, ann) in enumerate(img_annotations.iterrows()):
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']

        # 카테고리 이름 가져오기
        cat_match = categories_df[categories_df['id'] == category_id]
        if len(cat_match) == 0:
            category_name = f"Unknown (ID:{category_id})"
        else:
            category_name = cat_match['name'].values[0]

        color = colors[idx % len(colors)]

        print(f"  [{idx+1}] {category_name} - bbox: {bbox}")

        # 바운딩 박스 그리기
        rect = patches.Rectangle(
            (bbox[0], bbox[1]), bbox[2], bbox[3],
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)

        # 레이블 추가
        ax.text(bbox[0], bbox[1]-10, f'{category_name}',
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=color, alpha=0.8, pad=2),
                fontproperties=font_prop
        )

    ax.axis('off')
    plt.title(f"Image ID: {img_id} | 객체 수: {len(img_annotations)}개\n{img_info['file_name']}",
              fontsize=11, pad=10, fontproperties=font_prop)
    plt.tight_layout()
    plt.show()

print("✅ 개선된 시각화 함수 준비 완료!")


# 특정 이미지의 annotation 상세 확인
def check_image_annotations(img_id, images_df, annotations_df, categories_df):
    """특정 이미지의 annotation 상세 정보 확인"""

    # 이미지 정보
    img_info = images_df[images_df['id'] == img_id].iloc[0]
    print(f"📋 Image ID: {img_id}")
    print(f"파일명: {img_info['file_name']}")
    print(f"크기: {img_info['width']} x {img_info['height']}")

    # Annotation 정보
    img_annotations = annotations_df[annotations_df['image_id'] == img_id]
    print(f"\n📦 Annotation 개수: {len(img_annotations)}")

    if len(img_annotations) > 0:
        print("\n상세 정보:")
        for idx, (_, ann) in enumerate(img_annotations.iterrows()):
            cat_id = ann['category_id']
            cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
            bbox = ann['bbox']
            print(f"  [{idx+1}] {cat_name}")
            print(f"      Category ID: {cat_id}")
            print(f"      BBox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")
    else:
        print("⚠️ Annotation이 없습니다!")

    print("\n" + "="*60)


def convert_to_yolo_format(img_info, annotations, save_img_dir, save_label_dir, category_id_mapping, TRAIN_IMG_DIR):
    """COCO bbox를 YOLO 형식으로 변환"""
    img_id = img_info['id']
    file_name = img_info['file_name']
    img_w = img_info['width']
    img_h = img_info['height']

    # 이미지 복사
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