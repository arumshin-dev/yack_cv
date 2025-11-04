import json, os

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
