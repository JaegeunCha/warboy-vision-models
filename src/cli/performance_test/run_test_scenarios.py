import click

from ...test_scenarios.e2e import instance_seg, npu_performance, object_det, pose_est
from ...warboy import get_model_params_from_cfg


@click.command(
    "model-performance",
    help="Run end-to-end performance test for object detection, pose estimation, or instance segmentation models.",
    short_help="Run end-to-end performance test.",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the model configuration file.",
)
@click.option(
    "--batch-size",
    type=int,
    default=1,
    show_default=True,
    help="Override batch size for inference (default: 1).",
)
@click.option(
    "--save-samples",
    "--save-sample",
    type=int,
    default=10,
    show_default=True,
    help="Save N sample prediction images with boxes+labels to outputs/ (0=disable).",
)
@click.option(
    "--sample-start",
    type=int,
    default=1000,
    show_default=True,
    help="Global index start (0-based). 1000 means 1001st image.",
)
def run_e2e_test(config_file: str, batch_size: int, save_samples: int, sample_start: int):
    ANNOTATION_DIR = (
        "../datasets/coco/annotations"  # CHECK you may change this to your own path
    )

    IMAGE_DIR = "../datasets/coco/val2017"  # CHECK you may change this to your own path

    param = get_model_params_from_cfg(config_file)

    if param["task"] == "object_detection":
        #func = object_det.test_warboy_yolo_accuracy_det
        func = object_det.test_warboy_yolo_performance_det
        annotation = f"{ANNOTATION_DIR}/instances_val2017.json"
        image_dir = IMAGE_DIR

    elif param["task"] == "pose_estimation":
        func = pose_est.test_warboy_yolo_accuracy_pose
        annotation = f"{ANNOTATION_DIR}/person_keypoints_val2017.json"
        image_dir = IMAGE_DIR

    elif param["task"] == "instance_segmentation":
        func = instance_seg.test_warboy_yolo_accuracy_seg
        annotation = f"{ANNOTATION_DIR}/instances_val2017.json"
        image_dir = IMAGE_DIR

    else:
        raise ValueError(
            "Invalid task type. Choose from 'object_detection', 'pose_estimation', or 'instance_segmentation'."
        )

    func(config_file, image_dir, annotation, batch_size=batch_size, save_samples=save_samples, sample_start=sample_start)


@click.command(
    "npu-performance",
    help="Run NPU performance test.",
    short_help="Run NPU performance test.",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True, dir_okay=False, readable=True),
    required=True,
    help="The path to the model configuration file.",
)
@click.option(
    "--num_device",
    type=int,
    default=1,
    help="Number of devices to use (1 or 2).",
)
def run_npu_performance_test(config_file: str, num_device: int):
    npu_performance.test_warboy_performance(config_file, num_device)
