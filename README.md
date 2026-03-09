# warboy-vision-models 환경 설정 가이드

## 개요

이 문서는 `yolo-od-test` 프로젝트의 Furiosa Warboy NPU 기반 비전 모델 환경을 구성하고 실행하는 방법을 설명합니다.
rsync 서버에 저장된 데이터를 다운로드하고, `warboy-vision-models`를 git clone으로 받아 환경을 완성합니다.

## 사전 요구사항

- rsync 서버(기본: `10.254.202.100`)에 SSH 접속이 가능해야 합니다
- `rsync`, `git`, `python3` 명령어가 설치되어 있어야 합니다
- `tree`는 스크립트가 자동으로 설치합니다
- Furiosa SDK가 설치되어 있어야 합니다

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
4. C++ 빌드 의존성 설치 (`cmake`, `libeigen3-dev`)
5. Post-processing 유틸리티 빌드 (`build.sh` — cbox_decode, cpose_decode, cbytetrack `.so` 파일 생성)
6. `warboy-vision` CLI 패키지 설치 (`pip install .`)

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

### 5. 소스 코드 수정 시

`warboy-vision-models` 소스 코드를 수정한 경우, 변경 사항을 반영하려면 venv에서 다시 설치해야 합니다:

```bash
source ~/yolo-od-test/furiosa/venv/bin/activate
cd ~/yolo-od-test/furiosa/warboy-vision-models

# C++ 코드 수정 시 빌드도 다시 실행
bash build.sh

# 패키지 재설치
pip install .

deactivate
```

## 실행 방법

### venv 활성화

모든 실행 명령은 furiosa venv를 활성화한 상태에서 수행해야 합니다.

```bash
source ~/yolo-od-test/furiosa/venv/bin/activate
cd ~/yolo-od-test/furiosa/warboy-vision-models
```

### run_performance_suite.py — E2E 성능 평가

`models/enf/` 디렉토리의 ENF 파일을 자동 탐지하여 모든 모델, 모든 배치 사이즈에 대해 성능 평가를 수행합니다.

```bash
# 기본 실행 (모든 모델, 모든 배치 사이즈)
python3 run_performance_suite.py

# 샘플 이미지 저장 (예: 10장)
python3 run_performance_suite.py --save-samples 10

# 샘플 이미지 저장 시작 인덱스 지정 (1-based)
python3 run_performance_suite.py --save-samples 10 --sample-start 5
```

주요 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--save-samples` | `0` | 저장할 샘플 이미지 수 (0=비활성) |
| `--sample-start` | `None` | 샘플 저장 시작 인덱스 (1-based, `--save-samples > 0`일 때만 유효) |

결과 로그는 `logs/` 디렉토리에 저장됩니다:
- `performance_full_YYYYMMDD_HHMMSS.log` — 전체 실행 로그
- `performance_result_YYYYMMDD_HHMMSS.log` — 요약 결과 테이블 (모델별/배치별 Markdown)

### warboy-vision CLI — 개별 모델 실행

`warboy-vision` CLI로 개별 모델에 대해 직접 성능 평가를 수행할 수 있습니다.

```bash
# 단일 모델 성능 평가
warboy-vision model-performance \
    --config_file tutorials/cfg/yolov8n.yaml \
    --batch-size 1

# 다른 모델, 다른 배치 사이즈
warboy-vision model-performance \
    --config_file tutorials/cfg/yolov9t.yaml \
    --batch-size 4
```

사용 가능한 config 파일 (`tutorials/cfg/`):
- `yolov8n.yaml`, `yolov8l.yaml`
- `yolov9t.yaml`, `yolov9c.yaml`, `yolov9s.yaml`

### venv 비활성화

작업이 끝나면 venv를 비활성화합니다.

```bash
deactivate
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
    │   ├── tutorials/cfg/      ← 모델별 YAML 설정 파일
    │   └── logs/               ← 실행 결과 로그
    ├── venv/                   ← 신규 생성 (warboy-vision CLI 포함)
    ├── models/
    │   └── enf/                ← ENF 모델 파일 (.enf)
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
- 소스 수정 후에는 반드시 `pip install .`을 다시 실행해야 변경 사항이 반영됩니다
- ENF 파일은 `models/enf/`에 flat 구조로 저장됩니다 (하위 디렉토리 없음)

---

<details>
<summary><b>Warboy-Vision-Models 원본 README (Original)</b></summary>

## Warboy-Vision-Models

The `warboy-vision-models` project is designed to assist users in running various deep learning vision models on [FuriosaAI](https://furiosa.ai/)'s first generation NPU (Neural Processing Unit), Warboy.
Users can follow the outlined steps in the project to execute various vision applications, such as Object Detection, Pose Estimation, Instance Segmentation, etc., using Warboy.

We hope that the resources here will help you utilize the FuriosaAI Warboy in your applications.

### Model List

Currently, the project supports all vision applications provided by YOLO series ([YOLOv9](https://github.com/WongKinYiu/yolov9), [YOLOv8](https://github.com/ultralytics/ultralytics), [YOLOv7](https://github.com/WongKinYiu/yolov7) and [YOLOv5](https://github.com/ultralytics/yolov5)).

### Installation

This project requires Python 3.9 or above.

```sh
pip install -r requirements.txt
sudo apt-get update
sudo apt-get install cmake libeigen3-dev
./build.sh
pip install .
```

After installation, you can use the `warboy-vision` CLI:
```sh
warboy-vision --help
warboy-vision <command> --help
```

### Usage Example

- **Model making**
  ```sh
  warboy-vision make-model --config_file "/path/to/your/model/cfg.yaml"
  warboy-vision export-onnx --config_file "/path/to/your/model/cfg.yaml"
  warboy-vision quantize --config_file "/path/to/your/model/cfg.yaml"
  ```

- **Demo**
  ```sh
  warboy-vision run-demo --demo_config_file "/path/to/your/demo/cfg.yaml" --mode web
  warboy-vision run-demo --demo_config_file "/path/to/your/demo/cfg.yaml" --mode file
  ```

- **Performance test**
  ```sh
  warboy-vision model-performance --config_file "/path/to/your/model/cfg.yaml"
  warboy-vision npu-performance --config_file "/path/to/your/model/cfg.yaml"
  ```

For detailed information, please refer to the [FuriosaAI documentation](https://furiosa-ai.github.io/docs/latest/en/software/installation.html).

</details>
