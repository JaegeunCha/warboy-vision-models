import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import json
from typing import List
import asyncio
import time
from statistics import mean, median

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


async def _inference_with_metrics(model, data_loader, preprocessor, postprocessor, worker_num=16, output_dir="outputs"):
    from furiosa.runtime import create_runner

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    saved_count = 0

    async def task(runner, data_loader, worker_id, worker_num):
        nonlocal saved_count
        e2e_ms, inf_ms, pre_ms, post_ms = [], [], [], []
        results = []

        for idx, (img_path, annotation) in enumerate(data_loader):
            if idx % worker_num != worker_id:
                continue
            img = cv2.imread(str(img_path))
            img0shape = img.shape[:2]

            # 전처리
            t0 = time.perf_counter()
            inp, ctx = preprocessor(img)
            t1 = time.perf_counter()

            # 추론
            t2 = time.perf_counter()
            preds = await runner.run([inp])
            t3 = time.perf_counter()

            # 후처리
            t4 = time.perf_counter()
            outputs = postprocessor(preds, ctx, img0shape)[0]
            t5 = time.perf_counter()

            # coco eval용 결과
            bboxes = xyxy2xywh(outputs[:, :4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            for output, bbox in zip(outputs, bboxes):
                results.append(
                    {
                        "image_id": annotation["id"],
                        "category_id": YOLO_CATEGORY_TO_COCO_CATEGORY[int(output[5])],
                        "bbox": [round(x, 3) for x in bbox],
                        "score": round(float(output[4]), 5),
                    }
                )

            # 10장 출력 저장
            if saved_count < 10:
                for det in outputs:
                    x1, y1, x2, y2, conf, cls = det[:6]
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"{int(cls)} {conf:.2f}", (int(x1), max(int(y1) - 5, 0)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                out_path = Path(output_dir) / f"{saved_count+1}.jpg"
                cv2.imwrite(str(out_path), img)
                saved_count += 1

            # 시간 측정
            pre_ms.append((t1 - t0) * 1000)
            inf_ms.append((t3 - t2) * 1000)
            post_ms.append((t5 - t4) * 1000)
            e2e_ms.append((t5 - t0) * 1000)

        return e2e_ms, inf_ms, pre_ms, post_ms, results

    async with create_runner(model, worker_num=worker_num, compiler_config={"use_program_loading": True}) as runner:
        parts = await asyncio.gather(*[task(runner, data_loader, i, worker_num) for i in range(worker_num)])

    # 합치기
    e2e_all, inf_all, pre_all, post_all, results_all = [], [], [], [], []
    for e2e_ms, inf_ms, pre_ms, post_ms, results in parts:
        e2e_all.extend(e2e_ms)
        inf_all.extend(inf_ms)
        pre_all.extend(pre_ms)
        post_all.extend(post_ms)
        results_all.extend(results)

    return {"e2e": e2e_all, "inf": inf_all, "pre": pre_all, "post": post_all}, results_all


def test_warboy_yolo_performance_det(config_file: str, image_dir: str, annotation_file: str, use_enf=True, batch_size: int=1):
    """COCO mAP + latency/throughput JSON + 10장 이미지 출력"""

    param = get_model_params_from_cfg(config_file)
    model_name = param["model_name"]
    input_shape = param["input_shape"]
    anchors = param["anchors"]

    # engin_configs 통해 YAML 기반 conf/iou 사용
    engin_configs = set_test_engin_configs(param, 1)
    conf_thres = engin_configs[0]["conf_thres"]
    iou_thres  = engin_configs[0]["iou_thres"]

    # 모델 경로
    if use_enf:
        # batch-size에 따라 ENF 파일명 결정
        if batch_size and batch_size > 1:
            enf_file = f"{model_name}_{batch_size}b.enf"
        else:
            enf_file = f"{model_name}.enf"

        enf_path = ENF_DIR / param["task"] / enf_file
        if enf_path.is_file():
            model_path = str(enf_path)
        else:
            raise FileNotFoundError(f"ENF file not found: {enf_path}")
    else:
        model_path = param.get("onnx_i8_path")
        if not model_path:
            model_path = os.path.join(
                QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"]
            )

    # 전/후처리
    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    postprocessor = ObjDetPostprocess(model_name, {"conf_thres": conf_thres, "iou_thres": iou_thres, "anchors": anchors}, None, False).postprocess_func

    # COCO 데이터 로더
    data_loader = MSCOCODataLoader(Path(image_dir), Path(annotation_file), preprocessor, input_shape)

    # 추론 실행
    start = time.time()
    metrics, results = asyncio.run(_inference_with_metrics(model_path, data_loader, preprocessor, postprocessor))
    print(f"Inference Done in {time.time() - start:.2f} sec")

    # 성능 요약
    def summarize(xs: List[float]):
        avg, p50, p90, p99 = quantiles(xs)
        return {"avg": avg, "p50": p50, "p90": p90, "p99": p99}

    e2e = summarize(metrics["e2e"])
    inf = summarize(metrics["inf"])
    pre = summarize(metrics["pre"])
    post = summarize(metrics["post"])

    ips_e2e = (1000.0 / e2e["avg"]) if e2e["avg"] else None
    ips_inf = (1000.0 / inf["avg"]) if inf["avg"] else None

    summary = {
        "model": model_path,
        "cfg": config_file,
        "images": len(metrics["e2e"]),
        "throughput_img_per_s": {"e2e": ips_e2e, "infer_only": ips_inf},
        "latency_ms": {"pre": pre, "infer": inf, "post": post, "e2e": e2e},
    }
    print(json.dumps(summary, indent=2))
    Path("outputs").mkdir(exist_ok=True)
    with open("outputs/results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 정확도 평가 (COCOEval)
    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    mAP = coco_eval.stats[0]
    target = TARGET_ACCURACY.get(model_name, 0.3) * 0.9   # 90% 허용
    if mAP >= target:
        print(
            f"{model_name} Accuracy check success! -> mAP: {mAP} [Target: {target}]"
        )
    else:
        print(
            f"{model_name} Accuracy check failed! -> mAP: {mAP} [Target: {target}]"
        )