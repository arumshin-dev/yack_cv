import pandas as pd
from ultralytics import YOLO


def train_summary(categories_df, annotations_df, best_model_path):
    print("📊 클래스별 mAP 상세 분석")

    # 1. 훈련이 끝난 best 모델 로드 (results 객체나 경로 하드코딩)
    # best_model_path = "../../models/yolo_runs/yolo_ensemble_large/weights/best.pt"
    model = YOLO(best_model_path)

    # 2. val 세트로 검증 실행
    metrics = model.val(
        data="../../ai05-level1-project/yolo_dataset/data.yaml",
        split="val",
        verbose=False
    )

    # 3. 클래스별 mAP50-95 값 추출
    # maps_per_class = metrics.box.maps_per_class  # (클래스 수,) 배열
    maps_per_class = metrics.box.maps  # (클래스 수,) 배열

    # 4. 클래스 이름 매핑 (categories_df 사용)
    # categories_df를 'id' 기준으로 정렬
    categories_df_sorted = categories_df.sort_values('id').reset_index(drop=True)

    # YOLO 모델의 클래스 순서 (model.names)와 categories_df 순서가 다를 수 있으므로
    # model.names (YOLO 내부 순서)를 기준으로 매핑합니다.
    results_list = []
    for class_index, map_score in enumerate(maps_per_class):
        # YOLO 모델 내부의 class_index에 해당하는 클래스 이름 찾기
        class_name = model.names[class_index]

        # categories_df에서 해당 이름의 원본 'id' 찾기
        category_id = categories_df[categories_df['name'] == class_name]['id'].values[0]

        # 원본 데이터의 객체 수 (제공해주신 목록)
        count = annotations_df[annotations_df['category_id'] == category_id].shape[0]

        results_list.append({
            "Class Name": class_name,
            "Object Count": count,
            "mAP50-95": map_score
        })

    # 5. DataFrame으로 변환 및 mAP 낮은 순으로 정렬
    results_df = pd.DataFrame(results_list)
    print(results_df.sort_values('mAP50-95', ascending=True).to_markdown(index=False))

    # 6. 소수 클래스 점수 확인
    print("\n--- 🚨 소수 클래스 (100개 미만) 성능 ---")
    print(results_df[results_df['Object Count'] < 100].sort_values('mAP50-95', ascending=True).to_markdown(index=False))