import asyncio
import os
from pathlib import Path
from typing import List

import cv2
import pytest
from pycocotools.cocoeval import COCOeval

from src.warboy.cfg import get_model_params_from_cfg
from src.warboy.yolo.postprocess import ObjDetPostprocess
from src.warboy.yolo.preprocess import YoloPreProcessor
from tests.test_config import QUANTIZED_ONNX_DIR, TEST_MODEL_LIST
from tests.utils import (
    CONF_THRES,
    IOU_THRES,
    YOLO_CATEGORY_TO_COCO_CATEGORY,
    MSCOCODataLoader,
    xyxy2xywh,
)

CONFIG_PATH = "./tests/test_config/object_detection"
##-- jgcha
ENF_DIR = Path("../models/enf")  # models/enf/object_detection/<model_name>.enf
##-- True  -> ENF 파일을 직접 로드 (컴파일 없이 즉시 추론)
##-- False -> 양자화 ONNX(quantized_onnx)를 로드 (최초 1회 컴파일 후 캐시 재사용)
USE_ENF = True

YAML_PATH = [
    os.path.join(CONFIG_PATH, model_name + ".yaml")
    for model_name in TEST_MODEL_LIST["object_detection"]
]

PARAMS = [get_model_params_from_cfg(yaml) for yaml in YAML_PATH]

PARAMETERS = []
for param in PARAMS:
    task = param["task"]                    # 예: "object_detection"
    model_name = param["model_name"]        # 예: "yolov9t"
    if USE_ENF:
        # ENF 경로 직접 사용
        model_path = str(ENF_DIR / task / f"{model_name}.enf")
    else:
        # 양자화 ONNX 경로 사용 (최초 실행 시 컴파일 후 캐시 재사용)
        model_path = os.path.join(QUANTIZED_ONNX_DIR, task, param["onnx_i8_path"])

    PARAMETERS.append((
        model_name,
        model_path,
        param["input_shape"],
        param["anchors"],
        task,
    ))

# PARAMETERS = [
#     (
#         param["model_name"],
#         os.path.join(QUANTIZED_ONNX_DIR, param["task"], param["onnx_i8_path"]),
#         param["input_shape"],
#         param["anchors"],
#     )
#     for param in PARAMS
# ]

TARGET_ACCURACY = {
    "yolov5nu": 0.343,
    "yolov5su": 0.430,
    "yolov5mu": 0.490,
    "yolov5lu": 0.522,
    "yolov5xu": 0.532,
    "yolov5n": 0.280,
    "yolov5s": 0.374,
    "yolov5m": 0.454,
    "yolov5l": 0.490,
    "yolov5x": 0.507,
    "yolov7": 0.514,
    "yolov7x": 0.531,
    "yolov7-w6": 0.549,
    "yolov7-e6": 0.560,
    "yolov7-d6": 0.566,
    "yolov7-e6e": 0.568,
    "yolov8n": 0.373,
    "yolov8s": 0.449,
    "yolov8m": 0.502,
    "yolov8l": 0.529,
    "yolov8x": 0.539,
    "yolov9t": 0.383,
    "yolov9s": 0.468,
    "yolov9m": 0.514,
    "yolov9c": 0.530,
    "yolov9e": 0.556,
    "yolov5n6": 0.360,
    "yolov5n6u": 0.421,
    "yolov5s6": 0.448,
    "yolov5s6u": 0.486,
    "yolov5m6": 0.513,
    "yolov5m6u": 0.536,
    "yolov5l6": 0.537,
    "yolov5l6u": 0.557,
    "yolov5x6": 0.550,
    "yolov5x6u": 0.568,
}


async def warboy_inference(model, data_loader, preprocessor, postprocessor):
    async def task(
        runner, data_loader, preprocessor, postprocessor, worker_id, worker_num
    ):
        results = []
        for idx, (img_path, annotation) in enumerate(data_loader):
            if idx % worker_num != worker_id:
                continue

            img = cv2.imread(str(img_path))
            img0shape = img.shape[:2]
            input_, contexts = preprocessor(img)
            preds = await runner.run([input_])

            outputs = postprocessor(preds, contexts, img0shape)[0]

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

    from furiosa.runtime import create_runner

    worker_num = 16
    async with create_runner(
        model, worker_num=32, compiler_config={"use_program_loading": True}
    ) as runner:
        results = await asyncio.gather(
            *(
                task(
                    runner,
                    data_loader,
                    preprocessor,
                    postprocessor,
                    idx,
                    worker_num,
                )
                for idx in range(worker_num)
            )
        )
    return sum(results, [])


@pytest.mark.parametrize("model_name, model, input_shape, anchors", PARAMETERS)
def test_warboy_yolo_accuracy_det(
    model_name: str, model: str, input_shape: List[int], anchors
):

    ##-- ENF 사용 모드일 때는 ENF 파일 존재 검사
    if USE_ENF:
        if not Path(model).is_file():
            ##-- 안내: ENF 생성 방법 힌트 제공
            qonnx = os.path.join(QUANTIZED_ONNX_DIR, task, f"{model_name}_i8.onnx")
            enf_target = ENF_DIR / task / f"{model_name}.enf"
            raise FileNotFoundError(
                f"[ENF not found]\n"
                f"  - Expected: {model}\n\n"
                f"To create it once, run:\n"
                f"  furiosa-compiler {qonnx} -o {enf_target}\n"
                f"(Then rerun the test; it will load ENF without compilation.)"
            )

    model_cfg = {"conf_thres": CONF_THRES, "iou_thres": IOU_THRES, "anchors": anchors}

    preprocessor = YoloPreProcessor(new_shape=input_shape[2:])
    postprocessor = ObjDetPostprocess(
        model_name, model_cfg, None, False
    ).postprocess_func

    data_loader = MSCOCODataLoader(
        Path("../datasets/coco/val2017"),  # CHECK you may change this to your own path
        Path(
            "../datasets/coco/annotations/instances_val2017.json"
        ),  # CHECK you may change this to your own path
        preprocessor,
        input_shape,
    )

    results = asyncio.run(
        warboy_inference(model, data_loader, preprocessor, postprocessor)
    )
    coco_result = data_loader.coco.loadRes(results)
    coco_eval = COCOeval(data_loader.coco, coco_result, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    assert coco_eval.stats[0] >= (
        TARGET_ACCURACY[model_name] * 0.9
    ), f"{model_name} Accuracy check failed! -> mAP: {coco_eval.stats[0]} [Target: {TARGET_ACCURACY[model_name] * 0.9}]"
