"""
한글로 되어 있는 class 이름이 Confusion matrix에서 출력이 안되는 이슈가 있어서, 영어로 교체한다.
다만, 73개의 class 이름 모두가 영어로 있는 것이 아니어서, 영어로 있는 59개만 교체한다.
"""
import yaml
import pandas as pd

def make_class_list(categories_df, images_df, num_classes, train_success, val_success, yolo_dir):
    # 클래스 이름 리스트 생성
    class_names = []
    for cat_id in sorted(categories_df['id'].unique()):
        cat_name = categories_df[categories_df['id'] == cat_id]['name'].values[0]
        class_names.append(cat_name)

    ##class_names_en = get_class_name_en(categories_df, images_df)

    # data.yaml 내용
    data_yaml = {
        'path': yolo_dir,
        'train': 'images/train',
        'val': 'images/val',
        'nc': num_classes,
        'names': class_names
    }

    # 저장
    yaml_path = f"{yolo_dir}/data.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(data_yaml, f, allow_unicode=True, sort_keys=False)

    print("✅ data.yaml 생성 완료!")
    print(f"경로: {yaml_path}")
    print(f"\n📋 설정 내용:")
    print(f"  - Train 이미지: {train_success}개")
    print(f"  - Val 이미지: {val_success}개")
    print(f"  - 클래스 수: {num_classes}개")
    print(f"  - 클래스 예시: {class_names[:3]}")

    return yaml_path

def get_class_name_en(categories_df, images_df):

    # 1. 이미지 데이터에서 매핑에 사용할 핵심 컬럼만 추출 및 고유화
    # (dl_name, dl_name_en) 쌍을 준비합니다.
    img_map_df = images_df[['dl_name', 'dl_name_en']].drop_duplicates(subset=['dl_name'])

    # 2. categories_df (73개의 기준 목록)를 바탕으로 Left Join 수행
    # Left Join을 해야 73개의 카테고리 행이 모두 유지됩니다.
    # 매칭되지 않은 행의 'dl_name_en'은 NaN이 됩니다.
    match_df = pd.merge(
        categories_df[['name']].drop_duplicates(subset=['name']),  # 73개 카테고리 이름
        img_map_df,
        left_on='name',
        right_on='dl_name',
        how='left'
    )

    #TEST
    match_df.to_csv('match_df-output_file.csv', index=False, encoding='utf-8')

    # 3. 73개 행을 순회하며 'dl_name_en'이 있는 경우에만 저장
    class_name_en_set = set()  # Set을 사용해 중복 검사 없이 빠르게 추가

    # match_df의 각 행을 순회합니다. (총 73회 루프)
    # itertuples()는 (인덱스, name, dl_name, dl_name_en) 순서로 튜플을 반환합니다.
    for _, cat_name_kr, _, name_en in match_df.itertuples():

        # name_en이 NaN이 아니고 (매칭되었고), 값이 비어있지 않은 경우에만 저장
        if pd.notna(name_en) and name_en:
            class_name_en_set.add(name_en)
        else:
            class_name_en_set.add(cat_name_kr)

    # 4. Set을 최종 리스트로 변환
    class_name_en_list = list(class_name_en_set)

    print(f"매칭되어 저장된 dl_name_en의 고유 개수: {len(class_name_en_list)}개")
    print(f"저장된 dl_name_en 목록 (일부): {class_name_en_list[:5]}...")

    return class_name_en_list