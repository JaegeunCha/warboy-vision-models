#!/usr/bin/env python3
"""
Run warboy-vision model-performance for available ENF batch sizes per model,
log results, print progress to console, and optionally run detached.

- Detects models by scanning ../models/enf/object_detection/*.enf
- Always uses tutorials/cfg/<base>.yaml
- Runs: warboy-vision model-performance --config_file tutorials/cfg/<base>.yaml --batch-size <bs>
- Captures JSON + parses extra metrics (mAP, thresholds, inference sec)
- Summarizes results in Markdown tables
"""

from __future__ import annotations
import argparse, json, os, re, subprocess, sys, time, itertools, threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml

# -----------------------------
# Constants
# -----------------------------
REPO_ROOT = Path.cwd()
CFG_DIR = REPO_ROOT / "tutorials" / "cfg"
ENF_DIR = REPO_ROOT / ".." / "models" / "enf" / "object_detection"

# 실행 시점 기반 로그 파일명
START_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
FULL_LOG_FILE = REPO_ROOT / f"performance_full_{START_TS}.log"   # 전체 로그
RESULT_LOG_FILE = REPO_ROOT / f"performance_result_{START_TS}.log"  # 요약 로그

BATCH_MAP = {
    1: "{model}.enf",
    4: "{model}_4b.enf",
    8: "{model}_8b.enf",
    16: "{model}_16b.enf",
    32: "{model}_32b.enf",
}

# -----------------------------
# Logging helpers
# -----------------------------
def log_line(msg: str, both: bool = True):
    if msg == "":  # 빈 문자열이면 그냥 개행
        if both:
            print("", flush=True)
        try:
            with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
                f.write("\n")
        except Exception:
            pass
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    if both:
        print(line, flush=True)
    try:
        with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

# -----------------------------
# Utilities
# -----------------------------
def base_model_name(model: str) -> str:
    return re.sub(r"_\d+b$", "", model)

def discover_models_from_enf(enf_dir: Path) -> list[str]:
    models = set()
    for enf in enf_dir.glob("*.enf"):
        models.add(base_model_name(enf.stem))
    return sorted(models)

def available_batches_for_model(model: str) -> List[int]:
    bs_list: List[int] = []
    for bs, pat in BATCH_MAP.items():
        enf_path = ENF_DIR / pat.format(model=model)
        if enf_path.exists():
            bs_list.append(bs)
    return sorted(bs_list)

def get_e2e_wall_imgps(res: dict) -> Optional[float]:
    """
    Return per-image wall throughput (img/s).
    Primary: 1000 / latency_ms.e2e_wall.avg
    Fallback: throughput_img_per_s.e2e_wall_per_image (if provided by runner)
    """
    lat = res.get("latency_ms", {})
    e2e_wall_avg = (lat.get("e2e_wall") or {}).get("avg")
    if e2e_wall_avg and e2e_wall_avg > 0:
        return 1000.0 / e2e_wall_avg
    thr = res.get("throughput_img_per_s", {})
    return thr.get("e2e_wall_per_image")  # may be None on old versions


def run_cmd_stream(cmd: List[str]) -> Tuple[int, List[str]]:
    log_line(f"RUN: {' '.join(cmd)}")

    lines: List[str] = []
    panic_detected = False
    with FULL_LOG_FILE.open("a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(REPO_ROOT),
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            f.write(line)
            lines.append(line)
            if "Not enough memory" in line or "panicked at" in line:
                panic_detected = True
                log_line(f"[PANIC DETECTED] {line.strip()}")
                try: proc.kill()
                except Exception: pass
                break
        proc.wait(timeout=5)
        rc = proc.returncode
    if panic_detected:
        rc = rc or 99
    return rc, lines

def extract_result_json(lines: List[str]) -> Optional[dict]:
    text = "".join(lines)
    candidates: List[str] = []
    stack = 0; start_idx = None
    for i, ch in enumerate(text):
        if ch == '{':
            if stack == 0: start_idx = i
            stack += 1
        elif ch == '}':
            if stack > 0:
                stack -= 1
                if stack == 0 and start_idx is not None:
                    candidates.append(text[start_idx:i+1]); start_idx = None
    for snippet in reversed(candidates):
        try: obj = json.loads(snippet)
        except Exception: continue
        if isinstance(obj, dict) and "throughput_img_per_s" in obj:
            return obj
    return None

# -----------------------------
# Cfg file generator
# -----------------------------
COCO_CLASSES = [ "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
 "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse",
 "sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase",
 "frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard",
 "surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana",
 "apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
 "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
 "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
 "hair drier","toothbrush" ]

def ensure_cfg_yaml(model: str, cfg_dir: Path):
    cfg_path = cfg_dir / f"{model}.yaml"
    if cfg_path.exists(): return cfg_path
    cfg = {
        "task": "object_detection",
        "model_name": model,
        "weight": f"../models/weight/object_detection/{model}.pt",
        "onnx_path": f"../models/onnx/object_detection/{model}.onnx",
        "onnx_i8_path": f"../models/quantized_onnx/object_detection/{model}_i8.onnx",
        "calibration_params": {
            "calibration_method": "SQNR_ASYM",
            "calibration_data": "../datasets/coco/val2017",
            "num_calibration_data": 100,
        },
        "conf_thres": 0.025,
        "iou_thres": 0.7,
        "input_shape": [1, 3, 640, 640],
        "anchors": [None],
        "class_names": COCO_CLASSES,
    }
    cfg_dir.mkdir(parents=True, exist_ok=True)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    log_line(f"[AUTO] Created config: {cfg_path}")
    return cfg_path

# -----------------------------
# Suite Runner
# -----------------------------
def run_one(model: str, batch_size: int) -> Optional[dict]:
    base = base_model_name(model)
    cfg_path = ensure_cfg_yaml(base, CFG_DIR)
    rc, lines = run_cmd_stream([
        "warboy-vision","model-performance","--config_file",str(cfg_path),"--batch-size",str(batch_size)
    ])
    if rc != 0: 
        log_line(f"[WARN] model-performance failed (rc={rc}) for {model} bs={batch_size}")
        return None

    result = extract_result_json(lines)
    if result is None:
        log_line(f"[WARN] JSON result not found for {model} bs={batch_size}")
        return None

    # parse extra metrics
    conf_thres = iou_thres = sec = mAP = target = None
    status = None
    for line in lines:
        m1 = re.match(r"^([0-9.]+)\s+([0-9.]+)", line.strip())
        if m1:
            conf_thres, iou_thres = float(m1.group(1)), float(m1.group(2))
        m2 = re.search(r"Inference Done in ([0-9.]+) sec", line)
        if m2:
            sec = float(m2.group(1))
        m3 = re.search(r"Accuracy check (success|failed)! -> mAP: ([0-9.]+) \[Target: ([0-9.]+)\]", line)
        if m3:
            status = m3.group(1)
            mAP = float(m3.group(2))
            target = float(m3.group(3))

    # store parsed + computed
    result["metrics"] = {
        "conf_thres": conf_thres,
        "iou_thres": iou_thres,
        "sec": sec,
        "mAP": mAP,
        "target": target,
        "status": status,
    }
    
    # NEW: compute per-image wall throughput
    e2e_wall_imgps = get_e2e_wall_imgps(result)
    result.setdefault("computed", {})["e2e_wall_per_image"] = e2e_wall_imgps

    thr = result.get("throughput_img_per_s", {})
    lat = result.get("latency_ms", {})
    log_line("")
    log_line(
        "RESULT "
        f"{model} bs={batch_size} | "
        f"e2e_wall_imgps={e2e_wall_imgps} e2e_active={thr.get('e2e_active')} infer_only={thr.get('infer_only')} | "
        f"lat_pre_avg={(lat.get('pre',{}) or {}).get('avg')} lat_infer_avg={(lat.get('infer',{}) or {}).get('avg')} "
        f"lat_post_avg={(lat.get('post',{}) or {}).get('avg')} | "
        f"mAP={mAP} target={target} status={status} | conf={conf_thres} iou={iou_thres} sec={sec}"
    )
    log_line("")
    return result


def summarize_by_batch(all_results: Dict[str, Dict[int, dict]], batch_size: int) -> str:
    def fmt(x): 
        return "NA" if x is None else f"{x:.3f}"
    
    conf_val, iou_val = "NA", "NA"
    for bs_dict in all_results.values():
        if batch_size in bs_dict:
            conf_val, iou_val = get_conf_iou(bs_dict[batch_size])
            break

    header = f"[batch_size : {batch_size}] : conf ({conf_val}), iou ({iou_val})"
    table = [
        "| model | e2e_wall_per_image (img/s) | e2e_active (img/s) | infer_only (img/s) | "
        "lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "|-------|-----------------------------|--------------------|--------------------|"
        "----------------|-----------------|----------------|-----|--------|--------|---------|",
    ]

    for model, by_bs in all_results.items():
        if batch_size not in by_bs:
            continue
        res = by_bs[batch_size]
        thr = res.get("throughput_img_per_s", {})
        lat = res.get("latency_ms", {})
        m = res.get("metrics", {})
        comp = res.get("computed", {})
        row = (
            f"| {model} | {fmt(comp.get('e2e_wall_per_image'))} | "
            f"{fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | "
            f"{fmt((lat.get('pre',{}) or {}).get('avg'))} | "
            f"{fmt((lat.get('infer',{}) or {}).get('avg'))} | "
            f"{fmt((lat.get('post',{}) or {}).get('avg'))} | "
            f"{fmt(m.get('mAP'))} | {fmt(m.get('target'))} | {m.get('status') or 'NA'} | "
            f"{fmt(m.get('sec'))} |"
        )
        table.append(row)

    return "\n".join([header, ""] + table + [""])


def summarize_by_model(model: str, by_bs: Dict[int, dict]) -> str:
    if not by_bs:
        return f"[model : {model}] (no results)\n"
    def fmt(x): 
        return "NA" if x is None else f"{x:.3f}"
    
    first_metrics = next(iter(by_bs.values())).get("metrics", {})
    conf_val = fmt(first_metrics.get("conf_thres"))
    iou_val = fmt(first_metrics.get("iou_thres"))

    header = f"[model : {model}] : conf ({conf_val}), iou ({iou_val})"
    table = [
        "| batch_size | e2e_wall_per_image (img/s) | e2e_active (img/s) | infer_only (img/s) | "
        "lat_pre (ms) | lat_infer (ms) | lat_post (ms) | mAP | Target | Status | sec (s) |",
        "|------------|-----------------------------|--------------------|--------------------|"
        "----------------|-----------------|----------------|-----|--------|--------|---------|",
    ]
    for bs in sorted(by_bs.keys()):
        res = by_bs[bs]
        thr = res.get("throughput_img_per_s", {})
        lat = res.get("latency_ms", {})
        metrics = res.get("metrics", {})
        comp = res.get("computed", {})
        row = (
            f"| {bs} | {fmt(comp.get('e2e_wall_per_image'))} | "
            f"{fmt(thr.get('e2e_active'))} | {fmt(thr.get('infer_only'))} | "
            f"{fmt((lat.get('pre',{}) or {}).get('avg'))} | "
            f"{fmt((lat.get('infer',{}) or {}).get('avg'))} | "
            f"{fmt((lat.get('post',{}) or {}).get('avg'))} | "
            f"{fmt(metrics.get('mAP'))} | {fmt(metrics.get('target'))} | {metrics.get('status') or 'NA'} | "
            f"{fmt(metrics.get('sec'))} |"
        )
        table.append(row)
    return "\n".join([header, ""] + table + [""])

def get_conf_iou(from_res: dict) -> tuple[str, str]:
    """metrics 블록에서 conf/iou 문자열 반환"""
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    m = from_res.get("metrics", {}) if from_res else {}
    return fmt(m.get("conf_thres")), fmt(m.get("iou_thres"))

def extract_metric_value(res: dict, metric: str):
    """metric 이름별 값 추출"""
    thr, lat, m, comp = (
        res.get("throughput_img_per_s", {}),
        res.get("latency_ms", {}),
        res.get("metrics", {}),
        res.get("computed", {}),
    )
    if metric == "e2e_wall_per_image":
        return comp.get("e2e_wall_per_image")
    elif metric == "e2e_active":
        return thr.get("e2e_active")
    elif metric == "infer_only":
        return thr.get("infer_only")
    elif metric == "lat_pre":
        return (lat.get("pre", {}) or {}).get("avg")
    elif metric == "lat_infer":
        return (lat.get("infer", {}) or {}).get("avg")
    elif metric == "lat_post":
        return (lat.get("post", {}) or {}).get("avg")
    elif metric == "mAP":
        return m.get("mAP")
    elif metric == "Target":
        return m.get("target")
    elif metric == "Status":
        return m.get("status")
    elif metric == "conf":
        return m.get("conf_thres")
    elif metric == "iou":
        return m.get("iou_thres")
    elif metric == "sec":
        return m.get("sec")
    return None

def summarize_transposed_by_batch(all_results: Dict[str, Dict[int, dict]], batch_size: int) -> str:
    """
    Transposed summary: 행(metric), 열(models)
    """
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    models = list(all_results.keys())

    conf_val, iou_val = "NA", "NA"
    for bs_dict in all_results.values():
        if batch_size in bs_dict:
            conf_val, iou_val = get_conf_iou(bs_dict[batch_size])
            break

    metrics = [
        "e2e_wall_per_image", "e2e_active", "infer_only",
        "lat_pre", "lat_infer", "lat_post",
        "mAP", "Target", "Status", "sec"
    ]

    rows = []
    for metric in metrics:
        row = [metric]
        for model in models:
            res = all_results[model].get(batch_size)
            if not res:
                row.append("NA")
                continue
            val = extract_metric_value(res, metric)
            row.append(fmt(val) if metric != "Status" else (val or "NA"))
        rows.append("| " + " | ".join(row) + " |")

    header = ["models"] + models
    table = [
        f"[transposed summary: batch_size={batch_size}, conf ({conf_val}), iou ({iou_val})]",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ] + rows
    return "\n".join(table) + "\n"

def summarize_transposed_by_model(model: str, by_bs: Dict[int, dict]) -> str:
    """
    Transposed summary: 행(metric), 열(batch sizes)
    """
    def fmt(x): return "NA" if x is None else f"{x:.3f}"
    batch_sizes = sorted(by_bs.keys())

    # conf/iou는 첫 배치 결과에서 가져오기
    first_res = next(iter(by_bs.values()))
    conf_val, iou_val = get_conf_iou(first_res)

    metrics = ["e2e_wall_per_image", "e2e_active", "infer_only",
               "lat_pre", "lat_infer", "lat_post",
               "mAP", "Target", "Status", "sec"]

    rows = []
    for metric in metrics:
        row = [metric]
        for bs in batch_sizes:
            res = by_bs[bs]
            val = extract_metric_value(res, metric)
            row.append(fmt(val) if metric != "Status" else (val or "NA"))
        rows.append("| " + " | ".join(row) + " |")

    header = ["batch_size"] + [str(bs) for bs in batch_sizes]
    table = [
        f"[transposed summary: model={model}, conf ({conf_val}), iou ({iou_val})]",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ] + rows
    return "\n".join(table) + "\n"

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    log_line("="*80)
    log_line(f"===== Run started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} =====")
    log_line(f"Full log file: {FULL_LOG_FILE}")
    log_line(f"Result log file: {RESULT_LOG_FILE}")
    log_line("="*80)

    models = discover_models_from_enf(ENF_DIR)
    log_line(f"Models to process: {models}")
    all_results: Dict[str, Dict[int, dict]] = {}

    for model in models:
        log_line("="*80)
        log_line(f"PROCESS MODEL: {model}")
        batches = available_batches_for_model(model)
        if not batches:
            log_line(f"[WARN] No ENF found for {model}")
            continue

        log_line(f"Available batches: {batches}")
        model_results: Dict[int, dict] = {}

        for bs in batches:
            res = run_one(model, bs)
            if res:
                model_results[bs] = res

        all_results[model] = model_results

        # 모델별 요약 저장
        if model_results:
            summary = summarize_by_model(model, model_results)
            log_line("\n" + summary)  # full 로그에 기록
            with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                rf.write(summary + "\n\n")  # result 로그에 따로 저장

            summary_t = summarize_transposed_by_model(model, model_results)
            log_line("\n" + summary_t)
            with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
                rf.write(summary_t + "\n\n")

    # 배치 사이즈별 요약 (full 로그에만 기록)
    batch_sizes_present = sorted({bs for m in all_results.values() for bs in m.keys()})
    for bs in batch_sizes_present:
        summary = summarize_by_batch(all_results, bs)
        log_line("\n" + summary)
        with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
            rf.write(summary + "\n\n")

        summary_t = summarize_transposed_by_batch(all_results, bs)
        log_line("\n" + summary_t)
        with RESULT_LOG_FILE.open("a", encoding="utf-8") as rf:
            rf.write(summary_t + "\n\n")

    log_line("All done.")


if __name__=="__main__":
    try: main()
    except KeyboardInterrupt: log_line("Interrupted by user."); raise
