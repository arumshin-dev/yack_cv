from src.utils.korean import get_korean_font
from src.utils.korean import get_korean_font_path

# 데이터 기본 경로 (압축 해제한 위치)
BASE_DIR = "../../ai05-level1-project"  #"/content/data" #이미지 경로는 여기에 설정.
JSON_PATH = f"{BASE_DIR}/train_combined.json"

# 학습 및 테스트 데이터 경로
TRAIN_IMG_DIR = f"{BASE_DIR}/train_images"
TRAIN_ANN_DIR = f"{BASE_DIR}/train_annotations"
TEST_IMG_DIR = f"{BASE_DIR}/test_images"

YOLO_DIR = f"{BASE_DIR}/yolo_dataset"
# font의 설치된 경로
# 폰트 경로 지정 (윈도우 기본 폰트 폴더)
FONT_PATH = get_korean_font_path()
FONT_TYPE = get_korean_font()
