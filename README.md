# 💊 Pill Detection using YOLOv8  
> Object Detection Project

이 프로젝트는 **의약품 이미지 데이터셋**을 이용하여  
YOLOv8 기반의 **Object Detection (객체 탐지)** 모델을 학습하고,  
최종적으로 **TTA(Test Time Augmentation) 및 Ensemble**을 통해  
탐지 성능을 향상시키는 것을 목표로 합니다.

---
# 📝 협업 일지
**윤재형:** https://www.notion.so/Daily-292dbba8701180e89946c1484d2d2f3a?source=copy_link<br>
**전예린:** https://www.notion.so/1-Daily-2a1a85a71fed8049857ae25509e39e35?source=copy_link<br>
**이청수:** https://www.notion.so/Daily-29306271dc2a80e283aaea99537d8729<br>

---
## 📂 프로젝트 구조
```
object-detection/
├── .venv/                                # 가상환경 (Git에는 제외)
|
├── ai05-level1-project/                  # 실제 데이터셋 (이미지 + annotation)
│   ├── train_images/
│   ├── train_annotations/
│   └── test_images/
│
├── models/                               # 학습 완료된 YOLO 모델 가중치
│   ├── L-best.pt                         # 최종 yolo8l model
│   ├── M-best.pt                         # 최종 yolo8m model
│   └── yolo_runs/                        # 훈련 중간 결과 저장 폴더
│
├── src/
│   ├── datas/                            # 데이터 로드 및 전처리 관련
│   │   ├── data_loader.py                # JSON → DataFrame 변환
│   │   ├── data_stratify.py              # 계층적 데이터 분할
│   │   ├── PillDataset.py                # 커스텀 Dataset 정의
│   │   └── transforms.py                 # 데이터 증강(transform)
│   │
│   ├── utils/                            # 유틸 함수 모음
│   │   ├── albumentations_A.py           # Albumentations 증강 정의
│   │   ├── change_bbox.py                # bbox 조정 함수
│   │   ├── check_json.py                 # JSON 구조 검증
│   │   ├── process_annotation.py         # Annotation 병합 및 전처리
│   │   ├── korean.py, font.py            # 한글 시각화 관련
│   │
│   ├── YOLO/                             # YOLO 학습용 데이터 변환
│   │   ├── convert_data.py               # train/val 데이터를 YOLO 형식으로 변환
│   │   ├── convert_to_yolo_format.py     # json -> YOLO txt label 변환
│   │   ├── make_yaml.py                  # yaml 생성
│   │   └── make_yolo_dir.py              # yolo dir 생성
│   │
│   ├── main/                             # 메인 학습 및 실행 로직
│   │   ├── main.py                       # 전체 파이프라인 실행
│   │   ├── train_large.py                # YOLOv8-L 학습
│   │   ├── train_medium.py               # YOLOv8-M 학습
│   │   ├── ensemble_wbf.py               # Weighted Box Fusion 앙상블
│   │   ├── train_summary.py              # mAP 분석 및 결과 요약
│   │   └── yolov8l.pt, yolov8m.pt        # 사전학습(pretrained) 모델
│   │   └── make_dataframe                # 데이터 프레임 생성
|   |   └── make_csv                      # Kaggle 제출 csv 파일 생성
│   └── __init__.py
│
├── globals.py                            # 경로 상수(BASE_DIR 등)
├── EDA_result.ipynb                      # 데이터 탐색(EDA) 노트북
├── ensemble_submission_M11_TTA_conf1.csv # 케글 제출용 결과 파일
├── requirements.txt                      # 가상환경 패키지 목록
└── README.md
```



---

## 🚀 주요 기능

| 기능 | 설명                                                                                                                                            |
|------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| **데이터 병합 및 검증** | JSON annotation들을 하나의 통합 DataFrame으로 병합하고 bbox 이상치 제거                                                                                         |
| **계층적 데이터 분할 (Stratified Split)** | 클래스 불균형을 고려한 train/val 분할                                                                                                                     |
| **YOLO 데이터셋 변환** | COCO 형식 → YOLOv8 형식으로 자동 변환                                                                                                                   |
| **YOLOv8 학습** | YOLOv8-M, YOLOv8-L 두 가지 모델 학습                                                                                                                 |
| **Ensemble (WBF)** | Weighted Box Fusion으로 두 모델 결과 병합                                                                                                              |
| **TTA 적용** | Test Time Augmentation으로 소폭 mAP 향상                                                                                                            |
| **결과 시각화 및 분석** | Weights & biases를 통한 클래스별 mAP, Precision, Recall 등 시각화<br/>Weights & Biases: https://wandb.ai/yoonwogud-lab/pill-detection?nw=nwuseryoonwogud |

---

## 🧠 모델 구성

| 모델 | Base | Epoch | Optimizer | lr0 | TTA | val mAP50-95       |
|------|------|--------|------------|------|------|--------------------|
| YOLOv8-M | yolov8m.pt | 100 | Adam | lr0=0.00003 | ✅ | 0.8688075765157641 |
| YOLOv8-L | yolov8l.pt | 100 | Adam | lr0=0.00003 | ✅ | 0.8733844677336318 | 
| **Ensemble (WBF)** | M + L | - | - | - | ✅ | -                  |

---

## 📦 설치 방법

```bash
git clone https://github.com/wogud1220/object-detection.git
kaggle competitions download -c ai05-level1-project
cd object-detection
python -m venv .venv
source .venv/bin/activate     # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
# 실행
python src/main/main.py