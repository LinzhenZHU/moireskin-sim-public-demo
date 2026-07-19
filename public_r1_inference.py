#!/usr/bin/env python3
"""Frozen R1 ONNX inference and five-pressure public-demo adapter."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from public_object_catalog import PROCEDURAL_OBJECTS, procedural_object_maps
from rigid_object_poc import simulate_rigid_object_poc
from rigid_texture_calibration import (
    load_calibration_bundle,
    simulation_kwargs_from_bundle,
)


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "r1_latest_best.onnx"
MODEL_METADATA_PATH = ROOT / "models" / "r1_latest_best.json"
CALIBRATION_PATH = ROOT / "configs" / "rigid_texture_calibration_v1.json"
PRESSURES_KPA = np.asarray((0.0, 2.0, 4.0, 6.0, 8.0), dtype=np.float32)
PRESSURE_TENSION_STIFFENING_PER_KPA = 0.04
PHYSICS_IMAGE_SIZE = 192
CAMERA_SUPERSAMPLE = 2
FRAME_COUNTS = (1, 2, 3, 5)
OUTPUT_NAMES = (
    "height_mm",
    "mask_logits",
    "log_variance",
    "image_height_mm",
    "physics_height_mm",
    "physics_gate",
    "appearance_gate",
    "safety_gate",
)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_model_metadata():
    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    if metadata["artifact_sha256"] != _sha256(MODEL_PATH):
        raise ValueError("R1 ONNX artifact SHA-256 does not match its release metadata")
    if metadata["pressures_kpa"] != PRESSURES_KPA.tolist():
        raise ValueError("R1 pressure schedule does not match its release metadata")
    return metadata


@lru_cache(maxsize=1)
def load_onnx_session():
    import onnxruntime as ort

    load_model_metadata()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=("CPUExecutionProvider",),
    )


def pressure_subset_indices(frame_count, geometry_key):
    frame_count = int(frame_count)
    if frame_count not in FRAME_COUNTS:
        raise ValueError(f"frame count must be one of {FRAME_COUNTS}")
    if frame_count == 5:
        return (0, 1, 2, 3, 4)
    if frame_count == 1:
        return (2,)
    if frame_count == 2:
        step = int.from_bytes(
            hashlib.sha256(str(geometry_key).encode("utf-8")).digest()[:8],
            "big",
        )
        return (0, 2) if step % 2 == 0 else (2, 4)
    return (0, 2, 4)


def _simulation_config(base_config, calibration, pattern):
    grating = calibration["grating"]
    camera = calibration["camera"]
    config = dict(
        base_config,
        image_size=PHYSICS_IMAGE_SIZE,
        pattern=str(pattern),
        grating_pitch_mm=float(grating["pitch_mm"]),
        grating_angle_a_deg=float(grating["angle_a_deg"]),
        grating_angle_b_deg=float(grating["angle_b_deg"]),
        brightness_gain=list(camera["brightness_gain"]),
        brightness_offset=list(camera["brightness_offset"]),
        radial_distortion_k1=float(camera["radial_distortion_k1"]),
    )
    return config


def simulate_pressure_series(
    base_config,
    object_name,
    pattern,
    rotation_deg,
    texture_frequency,
    visual_texture_frequency,
    relief_scale,
    indentation_mm,
    offset_x_mm,
    offset_y_mm,
    base_tension_n_per_mm,
    camera_psf_sigma,
    grating_open_fraction,
    grating_line_transmittance,
    noise_std,
    seed,
):
    """Generate the same five-pressure observation contract used by R1."""
    calibration = load_calibration_bundle(CALIBRATION_PATH)
    config = _simulation_config(base_config, calibration, pattern)
    calibrated = simulation_kwargs_from_bundle(calibration)
    if object_name in PROCEDURAL_OBJECTS:
        height_map, albedo_map, _ = procedural_object_maps(
            object_name,
            config,
            rotation_deg,
            texture_frequency,
            visual_texture_frequency,
            offset_x_mm,
            offset_y_mm,
            seed,
        )
        simulator_object = "screwdriver"
        simulator_rotation = 0.0
        simulator_offset_x = 0.0
        simulator_offset_y = 0.0
    else:
        height_map = None
        albedo_map = None
        simulator_object = object_name
        simulator_rotation = rotation_deg
        simulator_offset_x = offset_x_mm
        simulator_offset_y = offset_y_mm

    states = []
    for pressure_index, pressure_kpa in enumerate(PRESSURES_KPA):
        tension = float(base_tension_n_per_mm) * (
            1.0
            + PRESSURE_TENSION_STIFFENING_PER_KPA * float(pressure_kpa)
        )
        states.append(
            simulate_rigid_object_poc(
                config,
                object_type=simulator_object,
                pattern=pattern,
                rotation_deg=simulator_rotation,
                texture_frequency=texture_frequency,
                visual_texture_frequency=visual_texture_frequency,
                relief_scale=relief_scale,
                indentation_mm=indentation_mm,
                offset_x_mm=simulator_offset_x,
                offset_y_mm=simulator_offset_y,
                height_map_mm=height_map,
                albedo_map=albedo_map,
                membrane_tension_n_per_mm=tension,
                inflation_pressure_kpa=float(pressure_kpa),
                membrane_bending_stiffness_n_mm=calibrated[
                    "membrane_bending_stiffness_n_mm"
                ],
                cavity_depth_mm=calibrated["cavity_depth_mm"],
                sealed_air_coupling=calibrated["sealed_air_coupling"],
                camera_psf_sigma=float(camera_psf_sigma),
                camera_supersample=CAMERA_SUPERSAMPLE,
                grating_open_fraction=float(grating_open_fraction),
                grating_line_transmittance=float(
                    grating_line_transmittance
                ),
                noise_std=float(noise_std),
                slope_to_shift_mm=calibrated["slope_to_shift_mm"],
                raw_flow_to_displacement_scale=calibrated[
                    "raw_flow_to_displacement_scale"
                ],
                raw_flow_highpass_mm=calibrated[
                    "raw_flow_highpass_mm"
                ],
                mechanical_mtf_calibration=calibrated[
                    "mechanical_mtf_calibration"
                ],
                confidence_calibration=calibrated[
                    "confidence_calibration"
                ],
                abstention_threshold=calibrated["abstention_threshold"],
                seed=int(seed) + pressure_index,
            )
        )
    return states


def prepare_r1_inputs(states, frame_count, geometry_key):
    if len(states) != len(PRESSURES_KPA):
        raise ValueError("R1 requires the frozen five-pressure sequence")
    selected = pressure_subset_indices(frame_count, geometry_key)
    validity = np.zeros((1, len(PRESSURES_KPA)), dtype=np.float32)
    validity[0, list(selected)] = 1.0

    def stack(name, dtype=np.float32):
        return np.stack([state[name] for state in states]).astype(dtype)

    pre = stack("pre_image") / 255.0
    post = stack("post_image") / 255.0
    raw_images = np.concatenate((pre, post, post - pre), axis=0)[None]

    recovered = np.clip(
        np.rint(stack("recovered_appearance") * 255.0),
        0,
        255,
    ).astype(np.uint8).astype(np.float32) / 255.0
    physics_height = stack("carrier_only_height_mm")
    fused_height = stack("reconstructed_height_mm")
    reconstruction_confidence = stack("reconstruction_confidence")
    relief_confidence = stack("appearance_geometry_confidence")
    carrier_confidence = stack("carrier_confidence")
    expected_error = stack("expected_error_mm") / 0.12
    estimated_masks = stack("estimated_mask")
    pressure_maps = np.broadcast_to(
        (PRESSURES_KPA / float(np.max(PRESSURES_KPA)))[:, None, None],
        physics_height.shape,
    ).copy()
    physics_features = np.concatenate(
        (
            recovered,
            physics_height,
            fused_height,
            reconstruction_confidence,
            relief_confidence,
            carrier_confidence,
            expected_error,
            estimated_masks,
            pressure_maps,
        ),
        axis=0,
    )[None]

    spatial_validity = validity[0, :, None, None]
    weights = (
        (0.05 + 0.95 * reconstruction_confidence)
        * estimated_masks
        * spatial_validity
    )
    weight_sum = np.maximum(weights.sum(axis=0), 1e-6)
    base_height = (
        (weights * physics_height).sum(axis=0) / weight_sum
    )[None, None]
    relief_support = np.max(
        estimated_masks * relief_confidence * spatial_validity,
        axis=0,
    )[None, None]
    inputs = {
        "raw_images": np.ascontiguousarray(raw_images, dtype=np.float32),
        "physics_features": np.ascontiguousarray(
            physics_features,
            dtype=np.float32,
        ),
        "base_height_mm": np.ascontiguousarray(
            base_height,
            dtype=np.float32,
        ),
        "relief_support": np.ascontiguousarray(
            relief_support,
            dtype=np.float32,
        ),
        "pressure_validity": validity,
    }
    return inputs, selected


def _texture_correlation(target, prediction, mask):
    if np.count_nonzero(mask) < 2:
        return 0.0
    target_detail = target - cv2.blur(target.astype(np.float32), (5, 5))
    prediction_detail = prediction - cv2.blur(
        prediction.astype(np.float32),
        (5, 5),
    )
    if (
        np.std(target_detail[mask]) < 1e-8
        or np.std(prediction_detail[mask]) < 1e-8
    ):
        return 0.0
    return float(
        np.corrcoef(target_detail[mask], prediction_detail[mask])[0, 1]
    )


def compute_demo_metrics(
    reference,
    height,
    predicted_mask,
    uncertainty=None,
):
    """Apply the frozen evaluator's synthetic height/mask metric support."""
    target_height = reference["ground_truth_height_mm"].astype(np.float32)
    target_mask = reference["target_mask"].astype(bool)
    predicted_mask = np.asarray(predicted_mask, dtype=bool)
    prediction = np.asarray(height, dtype=np.float32) * predicted_mask
    texture_mask = cv2.erode(
        target_mask.astype(np.uint8),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    ).astype(bool)
    if np.count_nonzero(texture_mask) < 16:
        texture_mask = target_mask
    height_scale = max(float(np.max(target_height[target_mask])), 1e-8)
    height_rmse = float(
        np.sqrt(
            np.mean(
                (
                    prediction[target_mask]
                    - target_height[target_mask]
                )
                ** 2
            )
        )
    )
    union = target_mask | predicted_mask
    common = target_mask & predicted_mask
    mask_iou = float(
        np.count_nonzero(common) / max(1, np.count_nonzero(union))
    )
    spacing_mm = float(reference["axis_mm"][1] - reference["axis_mm"][0])
    prediction_dx = np.diff(prediction, axis=1) / spacing_mm
    prediction_dy = np.diff(prediction, axis=0) / spacing_mm
    target_dx = np.diff(target_height, axis=1) / spacing_mm
    target_dy = np.diff(target_height, axis=0) / spacing_mm
    normal_support = (
        target_mask[:-1, :-1]
        & target_mask[:-1, 1:]
        & target_mask[1:, :-1]
    )
    prediction_normals = np.stack(
        (
            -prediction_dx[:-1],
            -prediction_dy[:, :-1],
            np.ones_like(prediction_dx[:-1]),
        ),
        axis=-1,
    )[normal_support]
    target_normals = np.stack(
        (
            -target_dx[:-1],
            -target_dy[:, :-1],
            np.ones_like(target_dx[:-1]),
        ),
        axis=-1,
    )[normal_support]
    denominator = np.maximum(
        np.linalg.norm(prediction_normals, axis=-1)
        * np.linalg.norm(target_normals, axis=-1),
        1e-8,
    )
    cosine = np.clip(
        np.sum(prediction_normals * target_normals, axis=-1) / denominator,
        -1.0,
        1.0,
    )
    normal_error = (
        float(np.degrees(np.arccos(cosine)).mean())
        if len(cosine)
        else float("inf")
    )
    metrics = {
        "height_nrmse": height_rmse / height_scale,
        "mask_iou": mask_iou,
        "normal_error_deg": normal_error,
        "texture_correlation": _texture_correlation(
            target_height,
            prediction,
            texture_mask,
        ),
    }
    if uncertainty is not None:
        metrics["uncertainty_p95_mm"] = float(
            np.percentile(uncertainty[target_mask], 95)
        )
    return metrics


def run_r1_inference(states, frame_count, geometry_key, session=None):
    inputs, selected = prepare_r1_inputs(states, frame_count, geometry_key)
    if session is None:
        session = load_onnx_session()
    values = session.run(OUTPUT_NAMES, inputs)
    output = {
        name: value[0, 0].astype(np.float32)
        for name, value in zip(OUTPUT_NAMES, values)
    }
    output["mask_probability"] = (
        1.0 / (1.0 + np.exp(-output["mask_logits"]))
    ).astype(np.float32)
    output["predicted_mask"] = output["mask_probability"] >= 0.5
    output["uncertainty_mm"] = np.exp(
        0.5 * output["log_variance"]
    ).astype(np.float32)
    output["display_height_mm"] = np.where(
        output["predicted_mask"],
        output["height_mm"],
        0.0,
    ).astype(np.float32)
    output["selected_pressure_indices"] = selected
    output["selected_pressures_kpa"] = tuple(
        float(PRESSURES_KPA[index]) for index in selected
    )
    output["base_height_mm"] = inputs["base_height_mm"][0, 0]
    physics_mask = np.max(
        np.stack(
            [states[index]["estimated_mask"] for index in selected],
            axis=0,
        ),
        axis=0,
    ).astype(bool)
    output["physics_mask"] = physics_mask
    output["metrics"] = compute_demo_metrics(
        states[len(states) // 2],
        output["height_mm"],
        output["predicted_mask"],
        output["uncertainty_mm"],
    )
    output["physics_metrics"] = compute_demo_metrics(
        states[len(states) // 2],
        output["base_height_mm"],
        physics_mask,
    )
    return output
