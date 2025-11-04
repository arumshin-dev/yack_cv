from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion as wbf
from tqdm import tqdm
import numpy as np
import os
from globals import BASE_DIR

# ✅ 모델 경로 설정
model_m_path = "../../models/M-best.pt"
model_l_path = "../../models/L-best.pt"

model_m = YOLO(model_m_path)
model_l = YOLO(model_l_path)

TEST_IMG_DIR = f"{BASE_DIR}/test_images"
test_images = sorted(os.listdir(TEST_IMG_DIR))
print(f"Test 이미지 개수: {len(test_images)}")

# ✅ 단일 모델 예측 함수
def predict_single(model, img_path, conf=0.1):
    """YOLO 모델 하나의 예측 결과 추출"""
    r = model.predict(img_path, conf=conf, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0:
        return [], [], []
    boxes = r.boxes.xyxy.cpu().numpy()        # (N,4)
    scores = r.boxes.conf.cpu().numpy().tolist()
    labels = r.boxes.cls.cpu().numpy().astype(int).tolist()

    h, w = r.orig_img.shape[:2]
    boxes = (boxes / np.array([w, h, w, h])).tolist()  # 정규화 [0,1]
    return boxes, scores, labels

# ✅ 두 모델 결과를 WBF로 병합하는 함수
def ensemble_wbf(img_path, conf=0.1, iou_thr=0.55):
    """YOLO m + l 결과를 Weighted Boxes Fusion으로 앙상블"""
    b1, s1, l1 = predict_single(model_m, img_path, conf)
    b2, s2, l2 = predict_single(model_l, img_path, conf)
    boxes_list = [b1, b2]
    scores_list = [s1, s2]
    labels_list = [l1, l2]

    if sum(len(b) for b in boxes_list) == 0:
        return np.empty((0,4)), np.empty((0,)), np.empty((0,))

    boxes, scores, labels = wbf(
        boxes_list, scores_list, labels_list,
        weights=[1.1, 1],
        iou_thr=iou_thr,
        skip_box_thr=0.0
    )

    # 원래 해상도로 되돌리기
    h, w = model_m.predict(img_path, verbose=False)[0].orig_img.shape[:2]
    boxes = (boxes * np.array([w, h, w, h])).astype(np.int32)
    return boxes, scores, labels

