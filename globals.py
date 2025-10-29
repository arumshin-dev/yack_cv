from src.utils.korean import get_korean_font
from src.utils.korean import get_korean_font_path

# 데이터 기본 경로 (압축 해제한 위치)
BASE_DIR = "C:/workspace/github/data"  #"/content/data" #이미지 경로는 여기에 설정.

# font의 설치된 경로
# 폰트 경로 지정 (윈도우 기본 폰트 폴더)
FONT_PATH = get_korean_font_path()
FONT_TYPE = get_korean_font()