import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import json
from typing import List
import asyncio
import time
from statistics import mean, median
import numpy as np

from pycocotools.cocoeval import COCOeval

from ...warboy import get_model_params_from_cfg
from ...warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine
from ...warboy.yolo.preprocess import YoloPreProcessor
from ...warboy.yolo.postprocess import ObjDetPostprocess
from ..utils import (
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    MSCOCODataLoader,
    set_test_engin_configs,
    xyxy2xywh,
)

# ------------------------------------------------------
# Config
# ------------------------------------------------------
ENF_DIR = Path("../models/enf")
QUANTIZED_ONNX_DIR = "quantized_onnx"

TARGET_ACCURACY = {
    "yolov5nu": 0.343, "yolov5su": 0.430, "yolov5mu": 0.490,
    "yolov5lu": 0.522, "yolov5xu": 0.532, "yolov5n": 0.280,
    "yolov5s": 0.374, "yolov5m": 0.454, "yolov5l": 0.490,
    "yolov5x": 0.507, "yolov7": 0.514, "yolov7x": 0.531,
    "yolov7-w6": 0.549, "yolov7-e6": 0.560, "yolov7-d6": 0.566,
    "yolov7-e6e": 0.568, "yolov8n": 0.373, "yolov8s": 0.449,
    "yolov8m": 0.502, "yolov8l": 0.529, "yolov8x": 0.539,
    "yolov9t": 0.383, "yolov9s": 0.468, "yolov9m": 0.514,
    "yolov9c": 0.530, "yolov9e": 0.556,
    "yolov5n6": 0.360, "yolov5n6u": 0.421, "yolov5s6": 0.448,
    "yolov5s6u": 0.486, "yolov5m6": 0.513, "yolov5m6u": 0.536,
    "yolov5l6": 0.537, "yolov5l6u": 0.557, "yolov5x6": 0.550,
    "yolov5x6u": 0.568,
}


# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def quantiles(arr: List[float]):
    if not arr:
        return (None, None, None, None)
    arr_sorted = sorted(arr)
    p50 = median(arr_sorted)

    def pick(p):
        if len(arr_sorted) < 100 and p == 0.99:
            return None
        if len(arr_sorted) < 10 and p == 0.90:
            return None
        idx = max(0, min(len(arr_sorted)-1, int(p * len(arr_sorted)) - 1))
        return arr_sorted[idx]
    
    return (mean(arr_sorted), p50, pick(0.90), pick(0.99))


def _process_output(img_path, annotation, outputs_dict):
    results = []
    key = str(img_path)

    if not len(outputs_dict[key]) == 1:
        print(len(outputs_dict[key]))

    for outputs in outputs_dict[key]:
        bboxes = xyxy2xywh(outputs[:, :4])
        bboxes[:, :2] -= bboxes[:, 2:] / 2

        for output, bbox in zip(outputs, bboxes):
            results.append(
                {
                    "image_id": annotation["id"],
                    "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                    "bbox": [round(x, 3) for x in bbox],
                    "score": round(output[4], 5),
                }
            )

    return results


def _process_outputs(outputs_dict, data_loader):
    all_results = []
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(_process_output, img_path, annotation, outputs_dict)
            for img_path, annotation in data_loader
        ]
        for future in futures:
            all_results.extend(future.result())
    return all_results


def test_warboy_yolo_accuracy_det(cfg: str, image_dir: str, annotation_file: str):
    """
    cfg(str): a path to config file
    image_dir(str): a path to image directory
    annotation_file(str): a path to annotation file
    """
    image_names = os.listdir(image_dir)

    images = [
        Image(image_info=os.path.join(image_dir, image_name))
        for image_name in image_names
    ]

    param = get_model_params_from_cfg(cfg)

    engin_configs = set_test_engin_configs(param, 2)

    preprocessor = YoloPreProcessor(new_shape=param["input_shape"], tensor_type="uint8")

    data_loader = MSCOCODataLoader(
        Path(image_dir),
        Path(annotation_file),
        preprocessor,
        param["input_shape"],
    )

    task = PipeLine(run_fast_api=False, run_e2e_test=True, num_channels=len(images))

    for idx, engin in enumerate(engin_configs):
        task.add(Engine(**engin), postprocess_as_img=False)
        task.add(
            ImageList(
                image_list=[image for image in images[idx :: len(engin_configs)]]
            ),
            name=engin["name"],
            postprocess_as_img=False,
        )

    task.run()
    outputs = task.outputs

    print("End Inference!")

    results = _process_outputs(outputs, data_loader)

    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    if coco_eval.stats[0] >= (TARGET_ACCURACY[param["model_name"]] * 0.9):
        print(
            f"{param['model_name']} Accuracy check success! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"
        )

    else:
        print(
            f"{param['model_name']} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[param['model_name']] * 0.9}]"
        )


async def _inference_with_metrics(model, data_loader, preprocessor, postprocessor,
                                  batch_size=1, worker_num=16, output_dir="outputs"):
    """
    수집 지표
    - e2e_wall   : (대기 포함) per-image 경과시간 ms
    - e2e_active : (대기 제외) per-image (pre + batch_infer/N + post) ms
    - inf        : per-image 순수 NPU(추론) ms
    - pre/post   : per-image 전/후처리 ms
    - batch_exec : 각 배치 run의 실행 로그 [{"n": 배치크기, "infer_ms": 배치 전체 NPU시간}, ...]
                   * 잔여 패딩 배치도 실행된 크기(=batch_size)로 기록 (HW 관점 처리량 반영)
    """
    from furiosa.runtime import create_runner
    import numpy as np
    from pathlib import Path
    import cv2, time

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_count = 0

    async def task(runner, data_loader, worker_id, worker_num):
        nonlocal saved_count
        e2e_wall_ms, e2e_active_ms = [], []
        inf_ms, pre_ms, post_ms, results = [], [], [], []
        batch_exec_log = []  # ← 배치 실행 로그
        batch_buf = []  # (inp3d, ctx, img0shape, ann, img, t0, t1, pre_ms_i)

        def visualize_and_collect(outputs, img, annotation):
            nonlocal saved_count, results
            bboxes = xyxy2xywh(outputs[:, :4]); bboxes[:, :2] -= bboxes[:, 2:] / 2
            for output, bbox in zip(outputs, bboxes):
                results.append({
                    "image_id": annotation["id"],
                    "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                    "bbox": [round(x, 3) for x in bbox],
                    "score": round(float(output[4]), 5),
                })
            if saved_count < 10:
                for det in outputs:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
                    cv2.putText(img, f"{int(cls)} {conf:.2f}",
                                (int(x1), max(int(y1)-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                out_path = Path(output_dir) / f"{saved_count+1}.jpg"
                cv2.imwrite(str(out_path), img); saved_count += 1

        for idx, (img_path, annotation) in enumerate(data_loader):
            if idx % worker_num != worker_id:
                continue

            img = cv2.imread(str(img_path))
            img0shape = img.shape[:2]

            # 전처리
            t0 = time.perf_counter()
            inp, ctx = preprocessor(img)  # (C,H,W) 또는 (1,C,H,W)
            t1 = time.perf_counter()
            pre_ms_i = (t1 - t0) * 1000.0

            if batch_size == 1:
                # 단일 배치: e2e_wall == e2e_active
                inp_4d = inp[np.newaxis, ...] if inp.ndim == 3 else inp
                t2 = time.perf_counter()
                preds = await runner.run([inp_4d])
                t3 = time.perf_counter()
                batch_infer_ms = (t3 - t2) * 1000.0
                # 배치 실행 로그 (N=1)
                batch_exec_log.append({"n": 1, "infer_ms": batch_infer_ms})

                t4 = time.perf_counter()
                outputs = postprocessor(preds, ctx, img0shape)[0]
                t5 = time.perf_counter()

                post_i = (t5 - t4) * 1000.0
                infer_i = batch_infer_ms  # N=1
                e2e_wall_i = (t5 - t0) * 1000.0
                e2e_active_i = pre_ms_i + infer_i + post_i

                pre_ms.append(pre_ms_i); inf_ms.append(infer_i); post_ms.append(post_i)
                e2e_wall_ms.append(e2e_wall_i); e2e_active_ms.append(e2e_active_i)
                visualize_and_collect(outputs, img, annotation)

            else:
                # 다중 배치: 3D로 통일 → 스택
                if inp.ndim == 4:
                    if inp.shape[0] != 1:
                        raise ValueError(f"For bs>1, expected leading batch 1. got {inp.shape}")
                    inp3d = inp[0]
                elif inp.ndim == 3:
                    inp3d = inp
                else:
                    raise ValueError(f"Unexpected input ndim for bs>1: {inp.ndim}")

                batch_buf.append((inp3d, ctx, img0shape, annotation, img, t0, t1, pre_ms_i))

                # 배치가 찼을 때 실행
                if len(batch_buf) == batch_size:
                    batched_input = np.stack([b[0] for b in batch_buf], axis=0)  # (N,C,H,W)
                    t2 = time.perf_counter()
                    preds = await runner.run([batched_input])  # heads: (N,*,H,W)...
                    t3 = time.perf_counter()
                    batch_infer_ms = (t3 - t2) * 1000.0
                    N = len(batch_buf)
                    batch_exec_log.append({"n": N, "infer_ms": batch_infer_ms})

                    heads = list(preds) if isinstance(preds, (tuple, list)) else [preds]
                    for i in range(N):
                        inp_i, ctx_i, img0shape_i, ann_i, img_i, t0_i, t1_i, pre_i = batch_buf[i]
                        # per-image heads → 4D 보장
                        per_img_heads = []
                        for h in heads:
                            if h.ndim == 4:   per_img_heads.append(h[i][np.newaxis, ...])
                            elif h.ndim == 3: per_img_heads.append(h[np.newaxis, ...])
                            else: raise ValueError(f"Unexpected head ndim (batch): {h.ndim}")

                        t4 = time.perf_counter()
                        outputs = postprocessor(per_img_heads, ctx_i, img0shape_i)[0]
                        t5 = time.perf_counter()

                        post_i = (t5 - t4) * 1000.0
                        infer_i = batch_infer_ms / N
                        e2e_active_i = pre_i + infer_i + post_i
                        e2e_wall_i   = (t5 - t0_i) * 1000.0  # 대기 포함

                        pre_ms.append(pre_i); inf_ms.append(infer_i); post_ms.append(post_i)
                        e2e_active_ms.append(e2e_active_i); e2e_wall_ms.append(e2e_wall_i)
                        visualize_and_collect(outputs, img_i, ann_i)

                    batch_buf = []

        # 잔여 묶음(패딩 실행)
        if batch_size > 1 and len(batch_buf) > 0:
            actual = len(batch_buf)
            pad_needed = batch_size - actual
            batch_padded = batch_buf + [batch_buf[-1]] * pad_needed
            batched_input = np.stack([b[0] for b in batch_padded], axis=0)
            t2 = time.perf_counter()
            preds = await runner.run([batched_input])
            t3 = time.perf_counter()
            batch_infer_ms = (t3 - t2) * 1000.0
            # 실행은 batch_size로 이루어졌으므로 HW 관점 처리량에선 N=batch_size로 기록
            batch_exec_log.append({"n": batch_size, "infer_ms": batch_infer_ms})

            heads = list(preds) if isinstance(preds, (tuple, list)) else [preds]
            for i in range(actual):
                inp_i, ctx_i, img0shape_i, ann_i, img_i, t0_i, t1_i, pre_i = batch_buf[i]
                per_img_heads = []
                for h in heads:
                    if h.ndim == 4:   per_img_heads.append(h[i][np.newaxis, ...])
                    elif h.ndim == 3: per_img_heads.append(h[np.newaxis, ...])
                    else: raise ValueError(f"Unexpected head ndim (remainder): {h.ndim}")

                t4 = time.perf_counter()
                outputs = postprocessor(per_img_heads, ctx_i, img0shape_i)[0]
                t5 = time.perf_counter()

                post_i = (t5 - t4) * 1000.0
                infer_i = batch_infer_ms / batch_size  # 패딩 포함 분배
                e2e_active_i = pre_i + infer_i + post_i
                e2e_wall_i   = (t5 - t0_i) * 1000.0

                pre_ms.append(pre_i); inf_ms.append(infer_i); post_ms.append(post_i)
                e2e_active_ms.append(e2e_active_i); e2e_wall_ms.append(e2e_wall_i)
                visualize_and_collect(outputs, img_i, ann_i)

        return e2e_wall_ms, e2e_active_ms, inf_ms, pre_ms, post_ms, batch_exec_log, results

    async with create_runner(model, worker_num=worker_num,
                             compiler_config={"use_program_loading": True}) as runner:
        parts = await asyncio.gather(*[
            task(runner, data_loader, i, worker_num) for i in range(worker_num)
        ])

    e2e_wall_all, e2e_active_all, inf_all, pre_all, post_all, batch_exec_all, results_all = [], [], [], [], [], [], []
    for e2e_wall_ms, e2e_active_ms, inf_ms, pre_ms, post_ms, batch_exec_log, results in parts:
        e2e_wall_all.extend(e2e_wall_ms)
        e2e_active_all.extend(e2e_active_ms)
        inf_all.extend(inf_ms); pre_all.extend(pre_ms); post_all.extend(post_ms)
        batch_exec_all.extend(batch_exec_log)
        results_all.extend(results)

    return {
        "e2e_wall": e2e_wall_all,       # 대기 포함 per-image(ms)
        "e2e_active": e2e_active_all,   # 대기 제외 per-image(ms)
        "inf": inf_all, "pre": pre_all, "post": post_all,
        "batch_exec": batch_exec_all    # [{"n": N, "infer_ms": batch_ms}, ...]
    }, results_all


def test_warboy_yolo_performance_det(config_file: str, image_dir: str, annotation_file: str,
                                     use_enf=True, batch_size: int=1):
    """COCO mAP + latency/throughput JSON + 10장 이미지 출력 (per-image 기준으로 일관 출력)"""

    param = get_model_params_from_cfg(config_file)
    model_name = param["model_name"]; input_shape = param["input_shape"]; anchors = param["anchors"]

    # YAML 기반 conf/iou
    engin_configs = set_test_engin_configs(param, 1)
    conf_thres = engin_configs[0]["conf_thres"]; iou_thres = engin_configs[0]["iou_thres"]

    # 모델 경로
    if use_enf:
        enf_file = f"{model_name}_{batch_size}b.enf" if batch_size > 1 else f"{model_name}.enf"
        enf_path = ENF_DIR / param["task"] / enf_file
        if enf_path.is_file(): model_path = str(enf_path)
        else: raise FileNotFoundError(f"ENF file not found: {enf_path}")
    else:
        model_path = param.get("onnx_i8_path") or os.path.join(QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"])

    # 전/후처리
    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    postprocessor = ObjDetPostprocess(
        model_name, {"conf_thres": conf_thres, "iou_thres": iou_thres, "anchors": anchors},
        None, False
    ).postprocess_func

    # COCO
    data_loader = MSCOCODataLoader(Path(image_dir), Path(annotation_file), preprocessor, input_shape)

    # 추론
    wall_start = time.time()
    metrics, results = asyncio.run(
        _inference_with_metrics(model_path, data_loader, preprocessor, postprocessor, batch_size=batch_size)
    )
    wall_elapsed = time.time() - wall_start
    print(f"Inference Done in {wall_elapsed:.2f} sec")

    # 요약 함수 (avg/p50/p90/p99 모두 반환)
    def summarize(xs: List[float]):
        avg, p50, p90, p99 = quantiles(xs)
        return {"avg": avg, "p50": p50, "p90": p90, "p99": p99}

    # 요약 지표
    e2e_active = summarize(metrics["e2e_active"])   # 대기 제외 per-image
    e2e_wall   = summarize(metrics["e2e_wall"])     # 대기 포함 per-image
    inf        = summarize(metrics["inf"])
    pre        = summarize(metrics["pre"])
    post       = summarize(metrics["post"])

    # 대기시간(= e2e_wall - e2e_active)
    wait_ms_list = [w - a for w, a in zip(metrics["e2e_wall"], metrics["e2e_active"])]
    wait = summarize(wait_ms_list)
    wait_ratio = (wait["avg"] / e2e_wall["avg"]) if (wait["avg"] and e2e_wall["avg"]) else None

    # 처리량(1장 기준만 기본 표기)
    ips_e2e_active     = (1000.0 / e2e_active["avg"]) if e2e_active["avg"] else None
    ips_inf            = (1000.0 / inf["avg"])        if inf["avg"]        else None
    ips_e2e_wall_imgps = (1000.0 / e2e_wall["avg"])   if e2e_wall["avg"]   else None

    # (옵션) 데이터셋 전체 처리량 & NPU 배치 처리량
    dataset_throughput_wall = (len(metrics["e2e_active"]) / wall_elapsed) if wall_elapsed > 0 else None
    total_imgs_in_batches   = sum(b["n"] for b in metrics["batch_exec"])
    total_infer_ms_batches  = sum(b["infer_ms"] for b in metrics["batch_exec"])
    hardware_batch_throughput = (
        (1000.0 * total_imgs_in_batches) / total_infer_ms_batches
        if total_infer_ms_batches > 0 else None
    )

    summary = {
        "model": model_path,
        "cfg": config_file,
        "images": len(metrics["e2e_active"]),
        "throughput_img_per_s": {  # 항상 '1장 기준'만 표기
            "e2e_active": ips_e2e_active,         # 권장 비교 지표
            "infer_only": ips_inf,                # 순수 NPU per-image
            "e2e_wall_per_image": ips_e2e_wall_imgps  # 대기 포함 per-image
        },
        "latency_ms": {
            "pre": pre, "infer": inf, "post": post,
            "e2e_active": e2e_active, "e2e_wall": e2e_wall,
            "wait": wait, "wait_ratio": wait_ratio
        },
        "dataset": {  # 해석용 보조 지표 (필요 시 참조)
            "throughput_wall_img_per_s": dataset_throughput_wall,
            "hardware_batch_throughput_img_per_s": hardware_batch_throughput
        }
    }

    print(json.dumps(summary, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # COCO mAP
    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    mAP = coco_eval.stats[0]
    target = TARGET_ACCURACY.get(model_name, 0.3) * 0.9
    if mAP >= target:
        print(f"{model_name} Accuracy check success! -> mAP: {mAP} [Target: {target}]")
    else:
        print(f"{model_name} Accuracy check failed! -> mAP: {mAP} [Target: {target}]")
