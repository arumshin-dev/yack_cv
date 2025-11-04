import pandas as pd

def make_csv(predictions, category_id_mapping):
    submission_rows = []
    annotation_id = 1

    # YOLO 카테고리 → 원본 category_id 매핑 (이건 기존에 쓰던 거 그대로!)
    # 예: {1: 0, 2: 1, 3: 2, ...} 이런 식
    # 만약 category_id_mapping 변수가 있다면 그대로 사용
    # 없으면 아래 주석 해제하고 불러오기
    # category_id_mapping = {cat_id: idx for idx, cat_id in enumerate(sorted(categories_df['id'].unique()))}

    for img_name, result in predictions.items():
        # 🔹 이미지 ID (파일명 숫자만 추출)
        image_id = int(img_name.replace('.png', '').replace('.jpg', ''))

        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]

        # 🔹 각 박스마다 한 행씩 저장
        for box, score, label in zip(boxes, scores, labels):
            yolo_cls = int(label)

            # 원본 category_id 복원
            category_id = None
            for orig_id, yolo_id in category_id_mapping.items():
                if yolo_id == yolo_cls:
                    category_id = int(orig_id)
                    break

            if category_id is None:
                continue  # 혹시 매핑 누락된 경우 skip

            x1, y1, x2, y2 = box
            bbox_x = int(x1)
            bbox_y = int(y1)
            bbox_w = int(x2 - x1)
            bbox_h = int(y2 - y1)

            submission_rows.append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox_x": bbox_x,
                "bbox_y": bbox_y,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
                "score": float(score),
            })
            annotation_id += 1

    # ✅ DataFrame 생성
    submission_df = pd.DataFrame(submission_rows)

    print("🔹 제출용 DataFrame 미리보기:")
    print(submission_df.head())

    # ✅ CSV로 저장
    output_path = "../..//ensemble_submission_M11_TTA_conf1.csv"
    submission_df.to_csv(output_path, index=False)
    print(f"✅ 앙상블 결과 CSV 저장 완료: {output_path}")