# warboy-vision-models 환경 설정 가이드

## 개요

이 문서는 `yolo-od-test` 프로젝트의 실행 환경을 구성하는 방법을 설명합니다.
rsync 서버에 저장된 데이터를 다운로드하고, `warboy-vision-models`를 git clone으로 받아 환경을 완성합니다.

## 사전 요구사항

- rsync 서버(기본: `10.254.202.100`)에 SSH 접속이 가능해야 합니다
- `rsync`, `git`, `python3` 명령어가 설치되어 있어야 합니다
- `tree`는 스크립트가 자동으로 설치합니다

## 설정 방법

### 1. 설정 스크립트 다운로드

rsync 서버에서 설정 스크립트를 다운로드합니다. (서버 IP는 환경에 맞게 변경)

```bash
scp kcloud@<서버IP>:~/data/setup_yolo_od_test.sh .
```

### 2. 스크립트 실행

```bash
chmod +x setup_yolo_od_test.sh

# 대화형으로 실행 (서버 IP 확인 → nvidia / furiosa / all 선택)
./setup_yolo_od_test.sh

# 또는 환경변수로 지정 (비대화형)
SERVER_A=10.254.202.100 SETUP_TARGET=furiosa ./setup_yolo_od_test.sh
```

스크립트가 수행하는 작업:
1. rsync 서버에서 선택한 대상의 데이터를 다운로드 (`warboy-vision-models`, `yolov9`, `venv`, 서버 전용 파일 제외)
2. `warboy-vision-models`를 git clone
3. `furiosa/venv` Python 가상환경 신규 생성 및 패키지 설치

### 3. 환경 변수

```bash
# 서버 IP (기본값: 10.254.202.100, 미지정 시 대화형으로 확인)
SERVER_A=10.254.202.100

# 설치 대상 (미지정 시 대화형으로 선택)
SETUP_TARGET=furiosa   # nvidia, furiosa, all

# SSH 사용자명 (기본값: kcloud)
SERVER_USER=myuser

# 로컬 저장 경로 (기본값: 현재 디렉토리)
LOCAL_BASE_DIR=/home/myuser/workspace
```

### 4. 추가 설치

이미 furiosa만 설치한 상태에서 nvidia를 추가할 수 있습니다:

```bash
SETUP_TARGET=nvidia ./setup_yolo_od_test.sh
```

## 디렉토리 구조

### furiosa만 선택 시

```
yolo-od-test/
├── data/
│   └── setup_yolo_od_test.sh
├── dockerImage/
│   └── furiosa/
└── furiosa/
    ├── warboy-vision-models/   ← git clone (이 저장소)
    ├── venv/                   ← 신규 생성
    ├── models/
    └── datasets/
```

### 모두 선택 시

```
yolo-od-test/
├── data/
│   └── setup_yolo_od_test.sh
├── dockerImage/
│   ├── furiosa/
│   └── nvidia/
├── furiosa/
│   ├── warboy-vision-models/   ← git clone (이 저장소)
│   ├── venv/                   ← 신규 생성
│   ├── models/
│   └── datasets/
└── nvidia/
    ├── yolov9/                 ← git clone
    ├── venv/                   ← 신규 생성
    ├── models/
    └── datasets/
```

## 참고

- `warboy-vision-models`는 Furiosa Warboy NPU 기반 비전 모델 저장소입니다
- 재실행 시 이미 clone된 저장소는 `git pull`로 업데이트됩니다
- venv는 매번 로컬에서 신규 생성되므로 서버 환경에 영향받지 않습니다
- rsync는 변경된 파일만 전송하므로 재실행 시에도 효율적입니다
