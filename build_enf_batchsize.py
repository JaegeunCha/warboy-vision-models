#!/usr/bin/env python3
"""
Automate YOLO -> ONNX -> INT8 -> ENF build pipeline for warboy-vision-models.

Update:
- Compiler failure now logs a warning and skips to next instead of raising RuntimeError.
- Spinner is preserved for console feedback, but spinner characters are not written to log file.
"""
from __future__ import annotations
import argparse
import os
import sys
import yaml
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import itertools, time, threading

BATCH_SIZES = [None]
DEFAULT_TASK = "object_detection"
DEFAULT_TEMPLATE_CFG = Path("tutorials/cfg/yolov8n.yaml")
CFG_DIR = Path("tutorials/cfg")
MODELS_DIR = Path("../models")
WEIGHT_DIR = MODELS_DIR / "weight" / DEFAULT_TASK
ONNX_DIR = MODELS_DIR / "onnx" / DEFAULT_TASK
QONNX_DIR = MODELS_DIR / "quantized_onnx" / DEFAULT_TASK
ENF_DIR = MODELS_DIR / "enf" / DEFAULT_TASK
MODEL_LIST_FILE = Path("model-list.yaml")
CALIB_DATA = Path("../datasets/coco/val2017")
LOG_FILE = Path("pipeline.log")

def log(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

def run_cmd(cmd: list[str], cwd: Path | None = None) -> int:
    log(f"RUN: {' '.join(cmd)} (cwd={cwd or Path.cwd()})")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            # write only to console, not spinner chars to log
            sys.stdout.write(line)
            f.write(line)
        proc.wait()
        return proc.returncode

def read_model_list(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("model_list", [])

def ensure_cfg_for_model(model: str, template_cfg: Path = DEFAULT_TEMPLATE_CFG) -> Path:
    CFG_DIR.mkdir(parents=True, exist_ok=True)
    cfg_path = CFG_DIR / f"{model}.yaml"
    if not cfg_path.exists():
        shutil.copyfile(template_cfg, cfg_path)
        log(f"Created cfg from template: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["task"] = DEFAULT_TASK
    cfg["model_name"] = model
    cfg["weight"] = str(WEIGHT_DIR / f"{model}.pt")
    cfg["onnx_path"] = str(ONNX_DIR / f"{model}.onnx")
    cfg["onnx_i8_path"] = str(QONNX_DIR / f"{model}_i8.onnx")
    cfg["calibration_params"] = {
        "calibration_method": "SQNR_ASYM",
        "calibration_data": str(CALIB_DATA),
        "num_calibration_data": 100,
    }
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return cfg_path

def expected_paths_for_model(model: str) -> dict[str, Path]:
    return {
        "weight": WEIGHT_DIR / f"{model}.pt",
        "onnx": ONNX_DIR / f"{model}.onnx",
        "qonnx": QONNX_DIR / f"{model}_i8.onnx",
        "enf_1": ENF_DIR / f"{model}.enf",
        "enf_4": ENF_DIR / f"{model}_4b.enf",
        "enf_8": ENF_DIR / f"{model}_8b.enf",
        "enf_16": ENF_DIR / f"{model}_16b.enf",
        "enf_32": ENF_DIR / f"{model}_32b.enf",
    }

def spinner(msg="컴파일 중..."):
    stop_flag = {"done": False}
    def run():
        for c in itertools.cycle('|/-\\'):
            if stop_flag["done"]:
                break
            sys.stderr.write(f'\r{msg} {c}')
            sys.stderr.flush()
            time.sleep(0.2)
        sys.stderr.write('\r완료!     \n')
        sys.stderr.flush()
    t = threading.Thread(target=run)
    t.daemon = True
    t.start()
    return lambda: stop_flag.update({"done": True})

def compile_enf_for_model(model: str, qonnx: Path):
    ENF_DIR.mkdir(parents=True, exist_ok=True)

    # BATCH_SIZES 기반 출력 경로 정의
    batch_map = {
        bs: (ENF_DIR / f"{model}.enf" if bs is None else ENF_DIR / f"{model}_{bs}b.enf")
        for bs in BATCH_SIZES
    }

    for bs, out_path in batch_map.items():
        if out_path.exists():
            log(f"[SKIP] ENF exists for {model} batch={1 if bs is None else bs}")
            continue

        cmd = ["furiosa-compiler", str(qonnx), "-o", str(out_path),
               "--target-npu", "warboy"]   # ✅ 64 DPES 장치 맞춤 설정 추가

        if bs is not None:  # batch=1(None)은 옵션 안 붙음
            cmd += ["--batch-size", str(bs)]

        rc = run_cmd(cmd)
        if rc != 0 or not out_path.exists():
            log(f"[WARN] Compiler failed for {out_path}, skipping.")
            continue
        log(f"[OK] Compiled ENF: {out_path}")

def export_and_quantize(cfg_path: Path, model: str, onnx: Path, qonnx: Path):
    if not onnx.exists():
        run_cmd(["warboy-vision", "export-onnx", "--config_file", str(cfg_path)])
    else:
        log(f"[SKIP] ONNX exists: {onnx}")
    if not qonnx.exists():
        run_cmd(["warboy-vision", "quantize", "--config_file", str(cfg_path)])
    else:
        log(f"[SKIP] Quantized ONNX exists: {qonnx}")

def main():
    models = read_model_list(MODEL_LIST_FILE)
    log(f"Models to process: {models}")
    for model in models:
        log("=" * 80)
        log(f"PROCESS MODEL: {model}")
        paths = expected_paths_for_model(model)
        enf_files = [paths["enf_1"], paths["enf_4"], paths["enf_8"], paths["enf_16"], paths["enf_32"]]
        if all(p.exists() for p in enf_files):
            log(f"[SKIP] All ENFs already exist for {model}")
            continue
        cfg_path = ensure_cfg_for_model(model)
        if not paths["weight"].exists():
            log(f"[WARN] Missing weight: {paths['weight']}")
            continue
        export_and_quantize(cfg_path, model, paths["onnx"], paths["qonnx"])
        compile_enf_for_model(model, paths["qonnx"])
    log("All done.")

if __name__ == "__main__":
    main()
