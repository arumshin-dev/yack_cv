import pandas as pd
from collections import Counter


def search_data(train_data):
    # 이미지 정보
    images_df = pd.DataFrame(train_data['images'])
    # print(f"📷 총 이미지 개수: {len(images_df)}")
    # print(images_df.head())

    # 카테고리 정보
    categories_df = pd.DataFrame(train_data['categories'])
    # print(f"\n🏷️ 총 카테고리(알약 종류): {len(categories_df)}")
    # print(categories_df)

    # Annotation 정보
    annotations_df = pd.DataFrame(train_data['annotations'])
    # print(f"\n📦 총 Annotation 개수: {len(annotations_df)}")
    # print(annotations_df.head())

    # 카테고리별 분포

    category_counts = Counter(annotations_df['category_id'])
    # print("\n📊 카테고리별 객체 개수:")
    for cat_id, count in sorted(category_counts.items()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        # print(f"  Class {cat_id} ({cat_name}): {count}개")

    # 이미지당 객체 수 분포
    img_obj_counts = annotations_df.groupby('image_id').size()
    # print(f"\n📈 이미지당 객체 수 통계:")
    # print(f"  - 평균: {img_obj_counts.mean():.2f}개")
    # print(f"  - 최소: {img_obj_counts.min()}개")
    # print(f"  - 최대: {img_obj_counts.max()}개")
    print("\nDf 생성 완료")
    return images_df, categories_df, annotations_df