import json
import os
from collections import defaultdict
from globals import BASE_DIR

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