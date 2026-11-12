# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import json
import os

import cv2

# --- hackathon demo patch: fiftyone is only needed by the coco variants ---
try:
    import fiftyone
except ImportError:
    fiftyone = None
import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0


# --- hackathon demo patch: removed from tt-metal; caching is internal now ---
def disable_persistent_kernel_cache():
    pass


from models.demos.utils.common_demo_utils import LoadImages, get_mesh_mappers, load_coco_class_names
from models.demos.yolov4.common import YOLOV4_L1_SMALL_SIZE, get_model_result
from models.demos.yolov4.post_processing import plot_boxes_cv2, post_processing
from models.demos.yolov4.runner.performant_runner_infra import YOLOv4PerformanceRunnerInfra
from models.demos.yolov4.runner.pipeline_runner import YoloV4PipelineRunner
from models.tt_cnn.tt.pipeline import PipelineConfig, create_pipeline_from_config


def preprocess_image(im, resolution):
    sized = cv2.resize(im, resolution)
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(sized.transpose(2, 0, 1)).float().div(255.0)
    return tensor


def run_yolov4(
    device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    batch_size = batch_size_per_device * device.get_num_devices()
    logger.info(f"Running with batch_size={batch_size} across {device.get_num_devices()} devices")

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    # Create the performance runner infrastructure
    test_infra = YOLOv4PerformanceRunnerInfra(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    # Get memory configs from the infrastructure
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        device, mesh_mapper=inputs_mesh_mapper, mesh_composer=output_mesh_composer
    )

    # Create pipeline configuration
    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        config,
        YoloV4PipelineRunner(test_infra),
        device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    # Compile pipeline
    pipeline.compile(tt_inputs_host)

    # --- hackathon demo patch 04: every image in the folder, in sorted order ---
    image_files = sorted(
        os.path.join(input_loc, f) for f in os.listdir(input_loc) if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    assert image_files, f"No .png, .jpg or .jpeg images in {input_loc}"
    logger.info(f"{len(image_files)} images in {input_loc}, {batch_size} per batch")

    conf_thresh = 0.3
    nms_thresh = 0.4
    class_names = load_coco_class_names()
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)

    for start in range(0, len(image_files), batch_size):
        chunk = image_files[start : start + batch_size]
        n_real = len(chunk)
        chunk = chunk + [chunk[-1]] * (batch_size - n_real)

        torch_images = []
        orig_images = []
        for image_path in chunk:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Image not found or unreadable: {image_path}")
            orig_images.append(image)
            torch_images.append(preprocess_image(image, resolution).unsqueeze(0))

        torch_input_tensor = torch.cat(torch_images, dim=0)
        tt_inputs_host, _ = test_infra._setup_l1_sharded_input(device, torch_input_tensor)
        outputs = pipeline.enqueue([tt_inputs_host]).pop_all()
        result_boxes, result_confs = get_model_result(outputs[0], resolution, mesh_composer=output_mesh_composer)
        boxes = post_processing(torch_images, conf_thresh, nms_thresh, [result_boxes, result_confs])

        for i in range(n_real):
            name = os.path.basename(chunk[i])
            img_base = os.path.splitext(name)[0]
            output_path = os.path.join(output_dir, f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg")
            plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)
            if len(boxes[i]) == 0:
                logger.warning(f"{name}: no detections above conf {conf_thresh}")
            else:
                logger.info(f"{name}: {len(boxes[i])} detections -> {output_path}")

    pipeline.cleanup()


def run_yolov4_coco(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    disable_persistent_kernel_cache()

    batch_size = batch_size_per_device * device.get_num_devices()
    logger.info(f"Running with batch_size={batch_size} across {device.get_num_devices()} devices")

    inputs_mesh_mapper, _, output_mesh_composer = get_mesh_mappers(device)

    # Create the performance runner infrastructure
    test_infra = YOLOv4PerformanceRunnerInfra(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator=model_location_generator,
        resolution=resolution,
        mesh_mapper=inputs_mesh_mapper,
        mesh_composer=output_mesh_composer,
    )

    # Get memory configs from the infrastructure
    tt_inputs_host, sharded_mem_config_DRAM, input_mem_config = test_infra.setup_dram_sharded_input(
        device, mesh_mapper=inputs_mesh_mapper, mesh_composer=output_mesh_composer
    )

    # Create pipeline configuration
    config = PipelineConfig(use_trace=True, num_command_queues=2, all_transfers_on_separate_command_queue=False)
    pipeline = create_pipeline_from_config(
        config,
        YoloV4PipelineRunner(test_infra),
        device,
        dram_input_memory_config=sharded_mem_config_DRAM,
        l1_input_memory_config=input_mem_config,
    )

    # Compile pipeline
    pipeline.compile(tt_inputs_host)

    dataset = fiftyone.zoo.load_zoo_dataset("coco-2017", split="validation", max_samples=batch_size)
    data_set = LoadImages([sample["filepath"] for sample in dataset])

    with open(os.path.expanduser("~") + "/fiftyone/coco-2017/info.json", "r") as file:
        coco_info = json.load(file)
        class_names = coco_info["classes"]

    # Preprocess batch of images to [BS, C, H, W]
    torch_images = []
    orig_images = []
    paths_images = []
    for batch in data_set:
        paths, im0s, _ = batch
        paths_images.append(paths[0])
        orig_images.append(im0s[0])
        tensor = preprocess_image(im0s[0], resolution)
        torch_images.append(tensor.unsqueeze(0))
        if len(torch_images) == batch_size:
            break

    torch_input_tensor = torch.cat(torch_images, dim=0)

    # Convert input to TTNN format using the infrastructure's method
    tt_inputs_host, _ = test_infra._setup_l1_sharded_input(device, torch_input_tensor)

    # Run inference
    outputs = pipeline.enqueue([tt_inputs_host]).pop_all()
    tt_output = outputs[0]

    # Convert TTNN output to PyTorch tensors using get_model_result
    result_boxes, result_confs = get_model_result(tt_output, resolution, mesh_composer=output_mesh_composer)

    conf_thresh = 0.3
    nms_thresh = 0.4
    boxes = post_processing(torch_images, conf_thresh, nms_thresh, [result_boxes, result_confs])

    class_names = load_coco_class_names()
    output_dir = "yolov4_predictions"
    os.makedirs(output_dir, exist_ok=True)
    for i in range(batch_size):
        img_base = os.path.splitext(os.path.basename(paths_images[i]))[0]
        output_filename = f"ttnn_yolov4_{img_base}_{resolution[0]}x{resolution[1]}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        plot_boxes_cv2(orig_images[i], boxes[i], output_path, class_names)

    pipeline.cleanup()


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc",
    [
        "models/demos/yolov4/resources",
    ],
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_yolov4(
    device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4(
        device,
        input_loc,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "input_loc",
    [
        "models/demos/yolov4/resources",
    ],
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_yolov4_dp(
    mesh_device,
    input_loc,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4(
        mesh_device,
        input_loc,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_yolov4_coco(
    device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4_coco(
        device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )


@run_for_wormhole_b0()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": YOLOV4_L1_SMALL_SIZE, "trace_region_size": 6434816, "num_command_queues": 2}],
    indirect=True,
)
@pytest.mark.parametrize(
    "batch_size_per_device, act_dtype, weight_dtype",
    ((1, ttnn.bfloat16, ttnn.bfloat16),),
)
@pytest.mark.parametrize(
    "resolution",
    [
        (320, 320),
        (640, 640),
    ],
)
def test_yolov4_coco_dp(
    mesh_device,
    batch_size_per_device,
    act_dtype,
    weight_dtype,
    model_location_generator,
    resolution,
):
    run_yolov4_coco(
        mesh_device,
        batch_size_per_device,
        act_dtype,
        weight_dtype,
        model_location_generator,
        resolution,
    )
