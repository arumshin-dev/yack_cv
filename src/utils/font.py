import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
import os
import glob
#import time

import globals


def set_font():
    #path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'  # 나눔 고딕
    path = globals.FONT_PATH
    font_name = fm.FontProperties(fname=path, size=10).get_name()  # 기본 폰트 사이즈 : 10
    plt.rc('font', family=font_name)

    fm.fontManager.addfont(path)

def add_font():
    # (1) 사용할 한글 폰트 이름 설정
    # macOS/Linux 사용자는 'AppleGothic' 또는 'NanumGothic'을,
    # Windows 사용자는 'Malgun Gothic'을 사용해 보세요.
    font_name = globals.FONT_TYPE  ##'Malgun Gothic'

    # 폰트 경로를 찾아서 설정
    font_path = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    font_list = [font.name for font in fm.fontManager.ttflist]

    # 사용할 폰트가 시스템에 있는지 확인
    if font_name not in font_list:
        print(f"경고: 시스템에 {font_name} 폰트가 없습니다. 다른 폰트를 시도하거나 설치하세요.")
        # 대체 폰트 설정 (예: Nanum Gothic)

    # (2) 폰트 설정 적용
    plt.rcParams['font.family'] = font_name

    # (3) 마이너스 부호 깨짐 방지 설정
    # Matplotlib은 기본적으로 마이너스 부호를 깨지게 표시할 수 있습니다.
    plt.rcParams['axes.unicode_minus'] = False


    #//캐시 지우기
    # 1. Matplotlib 캐시 디렉토리 경로 가져오기
    cache_dir = mpl.get_cachedir()
    print(f"Matplotlib 캐시 디렉토리: {cache_dir}")

    # 2. 캐시 디렉토리 내의 모든 폰트 캐시 파일 삭제
    # fontlist-v*.json 형식의 파일을 모두 찾아서 삭제합니다.
    try:
        deleted_count = 0
        # glob을 사용하여 경로를 탐색할 때, os.path.join을 사용하는 것이 Windows/Linux 호환성을 높입니다.
        for filename in glob.glob(os.path.join(cache_dir, 'fontlist-v*.json')):
            os.remove(filename)
            print(f"삭제 완료: {filename}")
            deleted_count += 1

        if deleted_count == 0:
            print("삭제할 Matplotlib 폰트 캐시 파일이 없거나 이미 삭제되었습니다.")

    except Exception as e:
        print(f"캐시 파일 삭제 중 오류 발생: {e}")

    # 3. 파이썬 환경 재시작 안내
    print("\n💡 캐시 파일 삭제 후, 변경 사항 적용을 위해 반드시 파이썬 커널/환경을 재시작해야 합니다 (예: 주피터 노트북 재시작).")


    # TEST plt 출력
    # (위의 폰트 설정 코드 실행 후)
    plt.figure(figsize=(8, 5))
    plt.plot([1, 2, 3], [10, 20, 30])

    # 한글 폰트 적용 확인
    plt.xlabel('시간 변화 (Time)')
    plt.ylabel('데이터 값 (Value)')
    plt.title('한글 제목 테스트')
    plt.show()