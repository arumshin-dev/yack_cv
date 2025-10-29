import platform
import matplotlib as mpl

def set_korean_font():
    system = platform.system()

    if system == 'Windows':
        mpl.rc('font', family='Malgun Gothic')
    elif system == 'Darwin':  # macOS
        mpl.rc('font', family='AppleGothic')
    else:
        mpl.rc('font', family='DejaVu Sans')  # 기본값 또는 리눅스용

    mpl.rc('axes', unicode_minus=False)

def get_korean_font():
    system = platform.system()

    family = ''
    if system == 'Windows':
        family = 'Malgun Gothic'
    elif system == 'Darwin':  # macOS
        family = 'AppleGothic'
    else:  # 기본값 또는 리눅스용
        family = 'DejaVu Sans'

    return family

def get_korean_font_path():
    system = platform.system()

    font_path = ''
    if system == 'Windows':
        font_path = 'C:/Windows/Fonts/malgun.ttf'
    elif system == 'Darwin':  # macOS
        # family = 'AppleGothic'
        font_path = 'C:/Windows/Fonts/malgun.ttf'  ##변경 필요
    else:  # 기본값 또는 리눅스용
        # family = 'DejaVu Sans'
        font_path = 'C:/Windows/Fonts/malgun.ttf'  ##변경 필요

    return font_path