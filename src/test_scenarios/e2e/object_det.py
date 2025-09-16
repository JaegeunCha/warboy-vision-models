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
from multiprocessing import Manager

from pycocotools.cocoeval import COCOeval

from ...warboy import get_model_params_from_cfg
from ...warboy.utils.process_pipeline import Engine, Image, ImageList, PipeLine
from ...warboy.yolo.preprocess import YoloPreProcessor

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


def test_warboy_yolo_performance_det(config_file: str, image_dir: str, annotation_file: str,
                                     use_enf=True, batch_size: int=1, save_samples: int=0, sample_start: int=1000):
    """성능/정확도 테스트: latency, throughput, mAP"""

    param = get_model_params_from_cfg(config_file)
    model_name = param["model_name"]
    input_shape = param["input_shape"] 

    # YAML 기반 conf/iou
    engin_configs = set_test_engin_configs(param, 1)

    # ENF 경로 확인
    if use_enf:
        enf_file = f"{model_name}_{batch_size}b.enf" if batch_size > 1 else f"{model_name}.enf"
        enf_path = ENF_DIR / param["task"] / enf_file
        if enf_path.is_file(): model_path = str(enf_path)
        else: raise FileNotFoundError(f"ENF file not found: {enf_path}")
    else:
        model_path = param.get("onnx_i8_path") or os.path.join(QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"])

    # 이미지 준비
    image_names = os.listdir(image_dir)
    images = [Image(image_info=os.path.join(image_dir, n)) for n in image_names]

    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    data_loader = MSCOCODataLoader(Path(image_dir), Path(annotation_file), preprocessor, input_shape)

    # shared timings
    manager = Manager()
    TIMINGS = manager.dict()

    # Pipeline
    #task = PipeLine(run_fast_api=False, run_e2e_test=True, num_channels=len(images), timings=TIMINGS)
    
    #outputs/<model_name> 초기화 (요청 시)
    import shutil
    save_dir = Path("outputs") / model_name
    if save_samples > 0:
        if save_dir.exists():
            shutil.rmtree(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Pipeline
    task = PipeLine(run_fast_api=False,
                    run_e2e_test=True,
                    num_channels=len(images),
                    timings=TIMINGS,
                    save_samples=save_samples,
                    sample_start=sample_start,
                    save_dir=str(save_dir))
    
    for idx, engin in enumerate(engin_configs):
        engin["model"] = model_path
        #print("[Engine Config]", json.dumps(engin, indent=2, default=str)) 
        task.add(Engine(**engin, batch_size=batch_size), postprocess_as_img=False)
        task.add(ImageList([image for image in images[idx::len(engin_configs)]]),
                 name=engin["name"], postprocess_as_img=False)

    wall_start = time.time()
    task.run()
    wall_elapsed = time.time() - wall_start

    print(f"Inference Done in {wall_elapsed:.2f} sec")

    # latency 요약
    #pre_list, infer_list, post_list, e2e_active_list, e2e_wall_list = [], [], [], [], []
    pre_list, infer_list, post_list, e2e_active_list = [], [], [], []
    for timings in TIMINGS.values():
        if "pre" in timings: pre_list.append(timings["pre"])
        if "infer" in timings: infer_list.append(timings["infer"])
        if "post" in timings: post_list.append(timings["post"])
        if "e2e_active" in timings: e2e_active_list.append(timings["e2e_active"])

    def summarize(xs): 
        return {"avg": float(np.mean(xs)), "p50": float(np.median(xs))} if xs else {}

    #num_imgs = len(pre_list)
    #e2e_wall_per_image = num_imgs / wall_elapsed if wall_elapsed > 0 else None

    summary = {
        "model": model_name,  # 파일 경로 대신 모델명만
        "cfg": os.path.basename(config_file),
        "images": len(pre_list),
        "throughput_img_per_s": {
            "e2e_active": 1000.0 / summarize(e2e_active_list).get("avg", np.nan),
            "infer_only": 1000.0 / summarize(infer_list).get("avg", np.nan),
            #"e2e_wall_per_image": 1000.0 / summarize(e2e_wall_list).get("avg", np.nan),
            #"e2e_wall_per_image": e2e_wall_per_image,
        },
        "latency_ms": {
            "pre": summarize(pre_list),
            "infer": summarize(infer_list),
            "post": summarize(post_list),
            "e2e_active": summarize(e2e_active_list),
            #"e2e_wall": summarize(e2e_wall_list),
        }
    }
    print(json.dumps(summary, indent=2))

    # COCO mAP
    results = _process_outputs(task.outputs, data_loader)
    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()

    mAP = coco_eval.stats[0]
    target = TARGET_ACCURACY.get(model_name, 0.3) * 0.9
    if mAP >= target:
        print(f"{model_name} Accuracy check success! -> mAP: {mAP} [Target: {target}]")
    else:
        print(f"{model_name} Accuracy check failed! -> mAP: {mAP} [Target: {target}]")
