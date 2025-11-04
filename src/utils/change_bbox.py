import json, os, cv2
import matplotlib.pyplot as plt
from globals import BASE_DIR


JSON_PATH = f"{BASE_DIR}/train_combined.json"
IMG_DIR = f"{BASE_DIR}/train_images"


def change_bbox():

    # 수정할 이미지 + 대상 category + 변경내용
    # 구조: { file_name: [ (old_cat, new_bbox, new_cat), ... ] }
    update_plan = {
        "K-001900-016548-019607-033009_0_2_0_2_70_000_200.png": [
            (16547, [88, 864, 250, 230], 16547),   # 기존 category_id, 새 bbox, 새 category_id
        ],
        "K-002483-003743-012081-019552_0_2_0_2_90_000_200.png": [
            (12080, [600, 708, 235, 451], 12080)
        ],
        "K-003351-003832-029667_0_2_0_2_90_000_200.png": [
            (29666,[95, 650, 350, 390], 29666)
        ],
        "K-003351-018147-020238_0_2_0_2_90_000_200.png": [
            (20237, [620, 770, 226, 224], 20237)
        ],
        "K-003351-020238-031863_0_2_0_2_70_000_200.png": [
            (20237, [590, 295, 210, 215], 20237)
        ],
        "K-003351-029667-031863_0_2_0_2_70_000_200.png": [
            (3350, [365, 852, 200, 200], 3350)
        ],
        "K-003483-019861-020238-031885_0_2_0_2_70_000_200.png": [
            (20237,[115, 853, 227, 226], 20237 )
        ],
        "K-003483-019861-025367-029667_0_2_0_2_90_000_200.png": [
            (29666, [637, 203, 224, 219], 29666)
        ],
        "K-003483-027733-030308-036637_0_2_0_2_90_000_200.png": [
            (27732, [125, 770, 315, 275], 27732)
        ],
        "K-003351-016262-018357_0_2_0_2_75_000_200.png": [
            (18356, [567, 625, 311, 315], 18356)
        ],
        "K-003544-004543-012247-016551_0_2_0_2_70_000_200.png": [
            (3543, [653, 889, 217, 217], 3543)
        ]
    }

    # --- JSON 로드
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        coco = json.load(f)

    # --- 파일명으로 탐색 후 수정
    for file_name, edits in update_plan.items():
        target_img = next((img for img in coco["images"] if img["file_name"] == file_name), None)
        if not target_img:
            print(f"❌ {file_name} 이미지를 찾을 수 없습니다.")
            continue

        image_id = target_img["id"]
        anns = [a for a in coco["annotations"] if a["image_id"] == image_id]

        # print(f"✅ {file_name} → {len(anns)}개의 annotation 발견")

        for old_cat, new_bbox, new_cat in edits:
            matched = False
            for ann in anns:
                if ann["category_id"] == old_cat:
                    old_bbox, old_category = ann["bbox"], ann["category_id"]
                    ann["bbox"] = new_bbox
                    ann["category_id"] = new_cat
                    ann["area"] = int(new_bbox[2] * new_bbox[3])
                    matched = True
                    print(f"🔧 bbox {old_bbox}→{new_bbox}, cat {old_category}→{new_cat}")
                    break
            if not matched:
                print(f"⚠️ {file_name}에서 category_id={old_cat}인 annotation을 찾지 못했습니다.")

    # --- 저장
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)
    print(f"\n문제 있는 Bbox 분리 완료: {JSON_PATH}")