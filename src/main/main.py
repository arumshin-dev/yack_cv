import os
import json
import pandas as pd

import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from tqdm import tqdm
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg
import matplotlib.font_manager as fm

from collections import defaultdict
from ultralytics import YOLO
from IPython.display import Image as IPImage, display

from src.datas.PillDataset import PillDataset
#from src.utils import util
from src.utils.make_yaml import make_class_list
from src.utils.make_yaml import get_class_name_en
from src.utils.util import visualize_annotations
from src.utils.util import convert_to_yolo_format
from src.utils.util import check_image_annotations
from src.utils.font import set_font
from src.utils.font import add_font
from src.utils.albumentations_A import train_compose
from src.utils.albumentations_A import val_compose
from src.utils.chageBbox import change_bboxes
from src.utils.korean import set_korean_font

import globals

# 데이터 기본 경로 (압축 해제한 위치)
BASE_DIR = globals.BASE_DIR
JSON_PATH = f"{BASE_DIR}/train_combined.json"

# 학습 및 테스트 데이터 경로
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
TRAIN_ANN_DIR = f"{BASE_DIR}/train_annotations"
TEST_IMG_DIR = f"{BASE_DIR}/test_images"

YOLO_DIR = f"{BASE_DIR}/yolo_dataset"

def main():
    # 한글 폰트 설정
    set_korean_font()

    """ main start """
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    """ # 경로 확인 """
    check_datapath()

    """ # 테스트 이미지 출력 """
    show_test_images(TEST_IMG_DIR)

    """ # Annotation 파일 수집 및 통합 """
    train_data, all_json_files = process_annotation(TRAIN_ANN_DIR)
    update_dict = {
        2339: [95, 630, 350, 425],
        2101: [600, 708, 235, 451],
        3679: [88, 864, 250, 230],
        805: [620, 770, 226, 224],
        2789: [590, 295, 210, 215],
        568: [365, 852, 200, 200],
        2374: [115, 853, 227, 226],
        2430: [637, 203, 224, 219],
        2778: [125, 770, 315, 275],
    665: [567, 625, 311, 315],
    972: [653, 889, 217, 217]
    }

    change_bboxes(JSON_PATH, update_dict)

    """ # 데이터 탐색 """
    images_df, categories_df, annotations_df = search_data(train_data)

    """  TEST   """
    #get_class_name_en(categories_df, images_df)

    ### FONT ###
    #set_font()
    #add_font()

    """ # 어노테이션 시각화 """
    process_visualize_annotations(images_df, categories_df, annotations_df)

    # 위에서 본 이미지들 확인
    check_image_annotations(1023, images_df, annotations_df, categories_df)  # 첫 번째 이미지
    check_image_annotations(599, images_df, annotations_df, categories_df)   # 두 번째 이미지

    """ JSON 파일을 확인 """
    check_json(all_json_files)

    """ 데이터셋, 데이터로더 처리 """
    train_images_df, val_images_df, train_annotations_df, val_annotations_df = process_data(images_df, categories_df, annotations_df)

    """ YOLO 데이터셋 """
    category_id_mapping, num_classes = process_yolo_dataset(categories_df)

    """ # Train, Val 데이터 변환 """
    train_success, val_success = convert_data(train_images_df, val_images_df, train_annotations_df, val_annotations_df, category_id_mapping)

    """ 클래스 이름 리스트 생성 """
    yaml_path = make_class_list(categories_df, images_df, num_classes, train_success, val_success, YOLO_DIR)

    """ 모델 인스턴스 생성 """
    model = make_model()

    """ 모델 학습 """
    train_model(model, yaml_path)

    """ 모델 결과 """
    result_model()

    process_visualize_clean(model, val_images_df, device)

    predict_model(model, val_images_df, val_annotations_df, categories_df, device, category_id_mapping)

    predictions = predict_weight_model(device, TEST_IMG_DIR)

    submission_df = result_submission(predictions, category_id_mapping)

    save_submission(submission_df)

    """ main end """


def check_datapath():
    # 경로 확인
    print("📂 경로 설정 완료:")
    print(f"BASE_DIR      : {BASE_DIR}")
    print(f"TRAIN_IMG_DIR : {TRAIN_IMG_DIR}")
    print(f"TEST_IMG_DIR  : {TEST_IMG_DIR}")

    # 실제 폴더 및 파일 존재 여부 확인
    print("📂 경로 설정:")
    for name, path in [("BASE_DIR", BASE_DIR),
                       ("TRAIN_IMG_DIR", TRAIN_IMG_DIR),
                       ("TRAIN_ANN_DIR", TRAIN_ANN_DIR),
                       ("TEST_IMG_DIR", TEST_IMG_DIR)]:
        exists = "✅" if os.path.exists(path) else "❌"
        print(f"{exists} {name}: {path}")

def show_test_images(test_img_dir):
    # 테스트 이미지 폴더
    #TEST_IMG_DIR = "/content/data/test_images"

    # 파일 목록 불러오기
    test_files = sorted(os.listdir(test_img_dir))

    # 이미지가 있는지 확인
    print(f"총 테스트 이미지 개수: {len(test_files)}")
    print("예시 파일명:", test_files[:5])

    # 앞부분 9개만 미리보기
    sample_files = test_files[:9]

    # 시각화
    plt.figure(figsize=(12, 12))
    for i, img_name in enumerate(sample_files):
        img_path = os.path.join(test_img_dir, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.title(img_name, fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Annotation 파일 수집 및 통합
def process_annotation(train_ann_dir):
    # 모든 JSON 파일 찾기
    all_json_files = []
    for root, dirs, files in os.walk(train_ann_dir):
        for file in files:
            if file.endswith('.json'):
                all_json_files.append(os.path.join(root, file))

    print(f"✅ 총 JSON 파일 개수: {len(all_json_files)}")
    print(f"예시 파일:\n{all_json_files[0]}")

    # file_name을 키로 하여 데이터 수집
    images_dict = {}  # {file_name: image_info}
    annotations_by_image = defaultdict(list)  # {file_name: [annotations]}
    categories_dict = {}  # {category_id: category_name}

    print("\n📊 JSON 파일 처리 중...")
    for idx, json_path in enumerate(all_json_files):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 이미지 정보 수집
            if 'images' in data and len(data['images']) > 0:
                img = data['images'][0]
                file_name = img['file_name']
                #dl_name = img['dl_name']
                #dl_name_en = img['dl_name_en']

                # 이미지 정보는 한 번만 저장 (중복 방지)
                if file_name not in images_dict:
                    images_dict[file_name] = {
                        'file_name': file_name,
                        'width': img.get('width'),
                        'height': img.get('height'),
                        #'dl_name': img.get('dl_name'),
                        #'dl_name_en': img.get('dl_name_en'),
                    }

            # Annotation 수집 (같은 file_name끼리 묶음)
            if 'annotations' in data:
                for ann in data['annotations']:
                    annotations_by_image[file_name].append({
                        'category_id': ann['category_id'],
                        'bbox': ann['bbox'],
                        'area': ann.get('area', ann['bbox'][2] * ann['bbox'][3]),
                        'iscrowd': ann.get('iscrowd', 0)
                    })

            # 카테고리 수집
            if 'categories' in data:
                for cat in data['categories']:
                    categories_dict[cat['id']] = cat['name']

            # 진행상황 출력 (500개마다)
            if (idx + 1) % 500 == 0:
                print(f"  처리 중... {idx + 1}/{len(all_json_files)}")

        except Exception as e:
            print(f"❌ 오류 ({os.path.basename(json_path)}): {e}")
            continue

    # COCO 형식으로 최종 정리
    combined_data = {
        'images': [],
        'annotations': [],
        'categories': []
    }

    image_id = 0
    annotation_id = 0

    print("\n🔗 이미지와 Annotation 연결 중...")
    for file_name, img_info in images_dict.items():
        # 이미지 추가
        img_info['id'] = image_id
        combined_data['images'].append(img_info)

        # 해당 이미지의 모든 annotation 추가
        for ann in annotations_by_image[file_name]:
            combined_data['annotations'].append({
                'id': annotation_id,
                'image_id': image_id,
                'category_id': ann['category_id'],
                'bbox': ann['bbox'],
                'area': ann['area'],
                'iscrowd': ann['iscrowd']
            })
            annotation_id += 1

        image_id += 1

    # 카테고리 정리
    combined_data['categories'] = [
        {'id': cat_id, 'name': cat_name}
        for cat_id, cat_name in sorted(categories_dict.items())
    ]

    print(f"\n✅ 통합 완료!")
    print(f"  - 총 이미지: {len(combined_data['images'])}")
    print(f"  - 총 Annotation: {len(combined_data['annotations'])}")
    print(f"  - 총 카테고리: {len(combined_data['categories'])}")
    print(f"  - 평균 이미지당 객체 수: {len(combined_data['annotations']) / len(combined_data['images']):.2f}개")

    # 통합 데이터 저장
    train_data = combined_data

    # 나중에 재사용할 수 있도록 파일로 저장
    output_path = f"{BASE_DIR}/train_combined.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    print(f"\n💾 통합 파일 저장: {output_path}")

    return train_data, all_json_files

def search_data(train_data):
    # 데이터 탐색

    # 이미지 정보
    images_df = pd.DataFrame(train_data['images'])
    print(f"📷 총 이미지 개수: {len(images_df)}")
    print(images_df.head())

    # 카테고리 정보
    categories_df = pd.DataFrame(train_data['categories'])
    print(f"\n🏷️ 총 카테고리(알약 종류): {len(categories_df)}")
    print(categories_df)

    # Annotation 정보
    annotations_df = pd.DataFrame(train_data['annotations'])
    print(f"\n📦 총 Annotation 개수: {len(annotations_df)}")
    print(annotations_df.head())

    # 카테고리별 분포
    category_counts = Counter(annotations_df['category_id'])
    print("\n📊 카테고리별 객체 개수:")
    for cat_id, count in sorted(category_counts.items()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        print(f"  Class {cat_id} ({cat_name}): {count}개")

    # 이미지당 객체 수 분포
    img_obj_counts = annotations_df.groupby('image_id').size()
    print(f"\n📈 이미지당 객체 수 통계:")
    print(f"  - 평균: {img_obj_counts.mean():.2f}개")
    print(f"  - 최소: {img_obj_counts.min()}개")
    print(f"  - 최대: {img_obj_counts.max()}개")

    return images_df, categories_df, annotations_df

def process_visualize_annotations(images_df, categories_df, annotations_df):
    valid_image_ids = annotations_df['image_id'].unique()
    print(f"📊 Annotation이 있는 이미지: {len(valid_image_ids)}개")
    print(f"📊 전체 이미지: {len(images_df)}개")

    if len(valid_image_ids) < len(images_df):
        print(f"⚠️ Annotation이 없는 이미지: {len(images_df) - len(valid_image_ids)}개")

    # 객체 수별 분포 다시 확인
    img_obj_counts_df = annotations_df.groupby('image_id').size().reset_index(name='count')
    print(f"\n📊 객체 수별 이미지 분포:")
    print(img_obj_counts_df['count'].value_counts().sort_index())

    # 여러 객체가 있는 이미지 찾기
    multi_obj_images = img_obj_counts_df[img_obj_counts_df['count'] >= 2]
    print(f"\n✅ 2개 이상 객체: {len(multi_obj_images)}개")

    # 시각화
    print("\n🎨 샘플 이미지 시각화 (크기 조정):")

    if len(multi_obj_images) > 0:
        # 여러 객체가 있는 이미지 우선
        print("여러 객체가 있는 이미지:")
        sample_ids = multi_obj_images['image_id'].sample(min(3, len(multi_obj_images))).values
    else:
        # 없으면 랜덤
        print("랜덤 샘플:")
        sample_ids = img_obj_counts_df['image_id'].sample(min(3, len(img_obj_counts_df))).values

    for img_id in sample_ids:
        visualize_annotations(TRAIN_IMG_DIR,
                        images_df,
                        annotations_df,
                        categories_df,
                        img_id,
                        figsize=(8, 8))

def check_json(all_json_files):
    #  원본 JSON 파일에서 직접 확인

    # Image ID 1023의 파일명으로 원본 JSON 찾기
    target_file = "K-001900-016548-031705-033208_0_2_0_2_75_000_200.png"

    print(f"🔍 {target_file}에 해당하는 원본 JSON 파일들:\n")

    json_count = 0
    for json_path in all_json_files:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if 'images' in data and len(data['images']) > 0:
                if data['images'][0]['file_name'] == target_file:
                    json_count += 1
                    print(f"[{json_count}] {os.path.basename(json_path)}")

                    if 'annotations' in data:
                        for ann in data['annotations']:
                            cat_id = ann['category_id']
                            # categories에서 이름 찾기
                            cat_name = "Unknown"
                            if 'categories' in data:
                                for cat in data['categories']:
                                    if cat['id'] == cat_id:
                                        cat_name = cat['name']
                                        break
                            bbox = ann['bbox']
                            print(f"    - {cat_name} (ID: {cat_id})")
                            print(f"      BBox: {bbox}")
                    print()
        except:
            continue

    print(f"✅ 총 {json_count}개의 JSON 파일 발견")
    print(f"\n💡 결론: 원본 데이터에도 {json_count}개의 annotation만 있음")
    print("    → 병합 과정은 정상이며, 데이터셋 자체가 이렇게 제공")

def process_data(images_df, categories_df, annotations_df):
    # Train/Val Split
    train_ids, val_ids = train_test_split(
        images_df['id'].values,
        test_size=0.2,
        random_state=42
    )

    train_images_df = images_df[images_df['id'].isin(train_ids)].reset_index(drop=True)
    val_images_df = images_df[images_df['id'].isin(val_ids)].reset_index(drop=True)

    train_annotations_df = annotations_df[annotations_df['image_id'].isin(train_ids)]
    val_annotations_df = annotations_df[annotations_df['image_id'].isin(val_ids)]

    train_transform = train_compose()
    val_transform = val_compose()

    # Dataset 생성
    train_dataset = PillDataset(
        TRAIN_IMG_DIR,
        train_images_df,
        train_annotations_df,
        categories_df,
        transform=train_transform
    )

    val_dataset = PillDataset(
        TRAIN_IMG_DIR,
        val_images_df,
        val_annotations_df,
        categories_df,
        transform=val_transform
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2
    )

    print("✅ 데이터 증강이 적용된 Dataset/DataLoader 생성 완료!")
    print(f"  - Train: {len(train_dataset)}개")
    print(f"  - Val: {len(val_dataset)}개")
    print(f"  - Train batches: {len(train_loader)}")
    print(f"  - Val batches: {len(val_loader)}")

    # 샘플 확인
    images, targets = next(iter(train_loader))
    print(f"\n✅ 샘플 배치:")
    print(f"  - Batch size: {len(images)}")
    print(f"  - 이미지 shape: {images[0].shape}")
    print(f"  - 객체 수: {len(targets[0]['labels'])}개")

    return train_images_df, val_images_df, train_annotations_df, val_annotations_df

# Collate 함수
def collate_fn(batch):
    return tuple(zip(*batch))

def process_yolo_dataset(categories_df):
    # YOLO 데이터셋 폴더 구조 생성
    #YOLO_DIR = f"{BASE_DIR}/yolo_dataset"
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

def convert_data(train_images_df, val_images_df, train_annotations_df, val_annotations_df, category_id_mapping):
    # Train 데이터 변환
    print("📝 Train 데이터 변환 중...")
    train_success = 0
    for _, img_info in tqdm(train_images_df.iterrows(), total=len(train_images_df)):
        if convert_to_yolo_format(
                img_info,
                train_annotations_df,
                f"{YOLO_DIR}/images/train",
                f"{YOLO_DIR}/labels/train",
                category_id_mapping,
                TRAIN_IMG_DIR
        ):
            train_success += 1

    print(f"✅ Train 데이터 변환 완료: {train_success}/{len(train_images_df)}개")

    # Val 데이터 변환
    print("\n📝 Val 데이터 변환 중...")
    val_success = 0
    for _, img_info in tqdm(val_images_df.iterrows(), total=len(val_images_df)):
        if convert_to_yolo_format(
                img_info,
                val_annotations_df,
                f"{YOLO_DIR}/images/val",
                f"{YOLO_DIR}/labels/val",
                category_id_mapping,
                TRAIN_IMG_DIR
        ):
            val_success += 1

    print(f"✅ Val 데이터 변환 완료: {val_success}/{len(val_images_df)}개")

    return train_success, val_success

def make_model():
    model = YOLO('yolov8m.pt')
    return model

def train_model(model, yaml_path):
    # 학습 파라미터
    results = model.train(
        data=yaml_path,
        epochs=1,  ##20,  # 최대 20 에폭  ##임시로 에폭을 1로 설정함.
        imgsz=800,  # 이미지 크기
        batch=8,  # 배치 크기
        patience=10,  # Early stopping patience (10 에폭 동안 개선 없으면 중단)
        save=True,  # 모델 저장
        device=0 if torch.cuda.is_available() else 'cpu',  # GPU 자동 선택
        project=f'{BASE_DIR}/yolo_runs',  # 결과 저장 폴더
        name='pill_detection',
        exist_ok=True,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,  # 초기 learning rate
        lrf=0.01,  # 최종 learning rate
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,  # box loss gain
        cls=0.5,  # cls loss gain
        dfl=1.5,  # dfl loss gain
        label_smoothing=0.0,
        val=True,  # Validation 수행
        plots=True,  # 학습 그래프 자동 생성
        verbose=True
    )

    print("\n 학습 완료!")
    print(f" 결과 저장 위치: {BASE_DIR}/yolo_runs/pill_detection")

def result_model():
    # 한글 폰트 설정
    plt.rcParams['font.family'] = globals.FONT_TYPE  ##'NanumBarunGothic'
    plt.rcParams['axes.unicode_minus'] = False

    # 폰트 경로 지정 (윈도우 기본 폰트 폴더)
    font_path = globals.FONT_PATH

    # FontProperties 객체 생성
    font_prop = fm.FontProperties(fname=font_path, size=15)

    #font_name = fm.FontProperties(fname=font_path).get_name()
    #plt.rc('font', family=font_name)

    # 결과 디렉터리 설정
    result_dir = f"{BASE_DIR}/yolo_runs/pill_detection"

    print("📈 YOLOv8 학습 결과 요약")
    print("=" * 60)


    # 1️⃣ Loss 그래프
    results_img = f"{result_dir}/results.png"
    if os.path.exists(results_img):
        print("\n1. 🔹 Loss 변화 그래프")
        img = mpimg.imread(results_img)
        plt.figure(figsize=(14, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('학습 결과 (Loss, mAP, Precision, Recall)', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ results.png를 찾을 수 없습니다.")

    # 2️⃣ Confusion Matrix
    cm_img = f"{result_dir}/confusion_matrix.png"
    if os.path.exists(cm_img):
        print("\n2. 🔹 Confusion Matrix")
        img = mpimg.imread(cm_img)
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('혼동 행렬 (Confusion Matrix)', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ confusion_matrix.png를 찾을 수 없습니다.")

    # 3️⃣ Box Precision Curve (올바른 파일명!)
    boxp_img = f"{result_dir}/BoxP_curve.png"
    if os.path.exists(boxp_img):
        print("\n3. 🔹 Box Precision Curve")
        img = mpimg.imread(boxp_img)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('정밀도 곡선 (Precision Curve)', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ BoxP_curve.png를 찾을 수 없습니다.")

    # 4️⃣ Box F1 Curve (올바른 파일명!)
    boxf1_img = f"{result_dir}/BoxF1_curve.png"
    if os.path.exists(boxf1_img):
        print("\n4. 🔹 Box F1 Score Curve")
        img = mpimg.imread(boxf1_img)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('F1 점수 곡선 (F1 Curve)', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ BoxF1_curve.png를 찾을 수 없습니다.")

    # 5️⃣ Box Precision-Recall Curve
    boxpr_img = f"{result_dir}/BoxPR_curve.png"
    if os.path.exists(boxpr_img):
        print("\n5. 🔹 Precision-Recall Curve")
        img = mpimg.imread(boxpr_img)
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title('정밀도-재현율 곡선 (PR Curve)', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ BoxPR_curve.png를 찾을 수 없습니다.")

    # 6️⃣ Validation 예측 결과
    val_batch0 = f"{result_dir}/val_batch0_pred.jpg"
    if os.path.exists(val_batch0):
        print("\n6. 🔹 Validation 예측 결과 (Batch 0)")
        img = mpimg.imread(val_batch0)
        plt.figure(figsize=(16, 12))
        plt.imshow(img)
        plt.axis('off')
        plt.title('검증 데이터 예측 결과', fontsize=14, pad=10, fontproperties=font_prop)
        plt.tight_layout()
        plt.show()
    else:
        print("❌ val_batch0_pred.jpg를 찾을 수 없습니다.")

    # 7️⃣ 최종 성능 지표
    print("\n" + "=" * 60)
    print("📊 최종 성능 지표")
    print("=" * 60)

    csv_path = f"{result_dir}/results.csv"
    if os.path.exists(csv_path):
        import pandas as pd
        results_df = pd.read_csv(csv_path)
        results_df.columns = results_df.columns.str.strip()

        # 마지막 epoch
        last_row = results_df.iloc[-1]

        print(f"\n🏆 최종 Epoch {int(last_row['epoch'])} 결과:")
        print(f"  • mAP50-95: {last_row['metrics/mAP50-95(B)']:.4f} ")
        print(f"  • mAP50:    {last_row['metrics/mAP50(B)']:.4f}")
        print(f"  • Precision: {last_row['metrics/precision(B)']:.4f}")
        print(f"  • Recall:    {last_row['metrics/recall(B)']:.4f}")

        # Best 값
        best_map = results_df['metrics/mAP50-95(B)'].max()
        best_epoch = results_df['metrics/mAP50-95(B)'].idxmax() + 1
        print(f"\n🥇 Best mAP50-95: {best_map:.4f} (Epoch {best_epoch})")
    else:
        print("❌ results.csv를 찾을 수 없습니다.")

    # 8️⃣ Best 모델 경로
    best_model = f"{result_dir}/weights/best.pt"
    print(f"\n💾 Best 모델 경로:")
    if os.path.exists(best_model):
        print(f"   ✅ {best_model}")
        size_mb = os.path.getsize(best_model) / (1024 * 1024)
        print(f"   📦 파일 크기: {size_mb:.2f} MB")
    else:
        print(f"   ❌ 파일을 찾을 수 없습니다.")

    print("\n" + "=" * 60)
    print("✅ 학습 결과 요약 완료!")
    print("=" * 60)

def visualize_clean(img_path, model, device, conf_threshold=0.35, iou_threshold=0.5):
    """
    겹침 없는 깔끔한 시각화
    """

    # 예측 (threshold 조정)
    results = model.predict(
        img_path,
        conf=conf_threshold,    # 낮은 confidence 제외
        iou=iou_threshold,      # 겹치는 박스 제거
        max_det=4,              # 최대 4개
        device=device,
        verbose=False
    )
    result = results[0]

    # 이미지 로드
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # PIL로 변환
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)

    # 폰트 로드
    try:
        font = ImageFont.truetype(globals.FONT_PATH, 16)
    except:
        font = ImageFont.load_default()

    # 박스별로 위치 조정하여 겹침 방지
    boxes_info = []
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = result.names[cls]

        # 이름 짧게
        if len(class_name) > 12:
            class_name = class_name[:12] + '...'

        boxes_info.append({
            'box': (x1, y1, x2, y2),
            'conf': conf,
            'cls': cls,
            'name': class_name
        })

    # confidence 높은 순으로 정렬
    boxes_info.sort(key=lambda x: x['conf'], reverse=True)

    # 그리기
    for idx, info in enumerate(boxes_info):
        x1, y1, x2, y2 = info['box']

        # 색상
        np.random.seed(info['cls'])
        color = tuple(np.random.randint(100, 255, 3).tolist())

        # 박스
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # 라벨 위치 조정 (위쪽에 공간 없으면 아래로)
        label = f"{info['name']} {info['conf']:.2f}"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # 위쪽 공간 확인
        if y1 - text_h - 8 < 0:
            # 아래쪽에 표시
            text_y = y2 + 2
            bg_y1, bg_y2 = y2, y2 + text_h + 6
        else:
            # 위쪽에 표시
            text_y = y1 - text_h - 4
            bg_y1, bg_y2 = y1 - text_h - 8, y1

        # 배경
        draw.rectangle([x1, bg_y1, x1 + text_w + 6, bg_y2], fill=color)

        # 텍스트
        draw.text((x1 + 3, text_y), label, fill=(255, 255, 255), font=font)

    return np.array(img_pil)

def process_visualize_clean(model, val_images_df, device):
    # 샘플 시각화
    sample_images = val_images_df.sample(6)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (_, img_info) in enumerate(sample_images.iterrows()):
        img_path = os.path.join(TRAIN_IMG_DIR, img_info['file_name'])

        img_result = visualize_clean(
            img_path,
            model,
            device,
            conf_threshold=0.4,
            iou_threshold=0.5
        )

        axes[idx].imshow(img_result)
        axes[idx].set_title(f"ID: {img_info['id']}", fontsize=11)
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(f"{BASE_DIR}/yolo_clean_predictions.png", dpi=120)
    plt.show()


def predict_model(model, val_images_df, val_annotations_df, categories_df, device, category_id_mapping):
    print("🔬 정확한 mAP@[0.75:0.95] 계산")
    print("=" * 60)

    # 1. Validation 데이터로 예측
    predictions_list = []

    print("\n📊 Validation 예측 중...")
    for _, img_info in val_images_df.iterrows():
        img_id = int(img_info['id'])
        img_path = os.path.join(TRAIN_IMG_DIR, img_info['file_name'])

        # 예측
        results = model.predict(img_path, conf=0.001, device=device, verbose=False)
        result = results[0]

        # COCO 형식으로 변환
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            yolo_cls = int(box.cls[0])

            # 원본 카테고리 ID
            category_id = None
            for orig_id, yolo_id in category_id_mapping.items():
                if yolo_id == yolo_cls:
                    category_id = int(orig_id)
                    break

            if category_id is None:
                continue

            # COCO bbox 형식: [x, y, width, height]
            predictions_list.append({
                'image_id': img_id,
                'category_id': category_id,
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': conf
            })

    print(f"✅ 총 {len(predictions_list)}개 예측 완료")

    # 2. COCO GT 준비 (완전한 형식)
    gt_annotations = {
        'info': {
            'description': 'Pill Detection Validation',
            'version': '1.0',
            'year': 2025
        },
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }

    # 이미지 정보
    for _, img in val_images_df.iterrows():
        gt_annotations['images'].append({
            'id': int(img['id']),
            'file_name': str(img['file_name']),
            'width': int(img['width']),
            'height': int(img['height'])
        })

    # Annotation 정보
    for _, ann in val_annotations_df.iterrows():
        gt_annotations['annotations'].append({
            'id': int(ann['id']),
            'image_id': int(ann['image_id']),
            'category_id': int(ann['category_id']),
            'bbox': [float(x) for x in ann['bbox']],
            'area': float(ann['area']),
            'iscrowd': int(ann.get('iscrowd', 0))
        })

    # 카테고리 정보
    for _, cat in categories_df.iterrows():
        gt_annotations['categories'].append({
            'id': int(cat['id']),
            'name': str(cat['name'])
        })

    print(f"✅ GT 데이터 준비 완료")

    # 3. JSON 저장
    gt_path = f"{BASE_DIR}/val_gt_coco.json"
    pred_path = f"{BASE_DIR}/val_pred_coco.json"

    with open(gt_path, 'w') as f:
        json.dump(gt_annotations, f, indent=2)

    with open(pred_path, 'w') as f:
        json.dump(predictions_list, f, indent=2)

    print(f"✅ JSON 파일 저장 완료")
    print(f"   GT: {gt_path}")
    print(f"   Pred: {pred_path}")

    # 4. COCO 평가
    print("\n📊 COCO 평가 실행 중...")
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(pred_path)

    # mAP@[0.75:0.95] 계산
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.array([0.75, 0.80, 0.85, 0.90, 0.95])
    coco_eval.evaluate()
    coco_eval.accumulate()

    print("\n🎯 대회 평가 지표 (mAP@[0.75:0.95]):")
    print("=" * 60)
    coco_eval.summarize()

    # mAP 추출
    map_75_95_exact = coco_eval.stats[0]

    print(f"\n🏆 최종 결과:")
    print(f"  mAP@[0.75:0.95]: {map_75_95_exact:.4f} ")
    print(f"  (IoU 0.75, 0.80, 0.85, 0.90, 0.95의 평균)")
    print("=" * 60)

def predict_weight_model(device, test_img_dir):
    # Best 모델 로드
    best_model_path = f"{BASE_DIR}/yolo_runs/pill_detection/weights/best.pt"
    model = YOLO(best_model_path)

    # Test 이미지 목록
    test_img_dir = f"{BASE_DIR}/test_images"
    test_images = sorted(os.listdir(test_img_dir))

    print(f"Test 이미지 개수: {len(test_images)}")

    # 추론
    predictions = {}
    for img_name in tqdm(test_images):
        img_path = os.path.join(test_img_dir, img_name)
        results = model.predict(img_path, conf=0.51, iou=0.5, max_det=4, device=device, verbose=False)
        predictions[img_name] = results[0]

    return predictions

def result_submission(predictions, category_id_mapping):
    submission_rows = []
    annotation_id = 1

    for img_name, result in predictions.items():
        # image_id: 파일명에서 숫자만 추출
        image_id = int(img_name.replace('.png', '').replace('.jpg', ''))

        # 각 박스마다 한 행
        for box in result.boxes:
            yolo_cls = int(box.cls[0])

            # 원본 카테고리 ID
            category_id = None
            for orig_id, yolo_id in category_id_mapping.items():
                if yolo_id == yolo_cls:
                    category_id = int(orig_id)
                    break

            if category_id is None:
                continue

            score = float(box.conf[0])

            # BBox
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bbox_x = int(x1)
            bbox_y = int(y1)
            bbox_w = int(x2 - x1)
            bbox_h = int(y2 - y1)

            submission_rows.append({
                'annotation_id': annotation_id,
                'image_id': image_id,
                'category_id': category_id,
                'bbox_x': bbox_x,
                'bbox_y': bbox_y,
                'bbox_w': bbox_w,
                'bbox_h': bbox_h,
                'score': score
            })

            annotation_id += 1

    # DataFrame
    submission_df = pd.DataFrame(submission_rows)

    print(submission_df.head(5))

    return submission_df

def save_submission(submission_df):
    submission_path = f"{BASE_DIR}/submission.csv"
    submission_df.to_csv(submission_path, index=False)

    print(f"저장 완료: {submission_path}")

    # 헤더 확인
    with open(submission_path, 'r') as f:
        print(f" 헤더:")
        print(f.readline().strip())
        print(f"첫 5줄:")
        f.seek(0)
        for i, line in enumerate(f):
            if i < 6:
                print(line.strip())


if __name__ == "__main__":
    main()