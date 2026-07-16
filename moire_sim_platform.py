#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np


SCHEMA_VERSION = 1
TARGET_NAMES = ("contact_x_mm", "contact_y_mm", "normal_force_n")
CONDITION_NAMES = ("pressure_kpa",)
GRAM_FORCE_TO_NEWTON = 9.80665e-3
SAMPLE_PARAMETER_NAMES = (
    "contact_radius_mm",
    "brightness_gain",
    "brightness_offset",
    "noise_std",
)

DEFAULT_CONFIG = {
    "image_size": 64,
    "sensor_radius_mm": 15.0,
    "pattern": "parallel",
    "grating_pitch_mm": 0.33,
    "grating_angle_a_deg": 0.0,
    "grating_angle_b_deg": 6.31,
    "pressure_kpa": [0.0, 5.0],
    "normal_force_n": [0.1, 3.0],
    "contact_radius_mm": [1.5, 3.5],
    "membrane_stiffness_n_per_mm": 2.0,
    "pressure_stiffening_per_kpa": 0.15,
    "pressure_strain_per_kpa": 0.0005,
    "top_layer_motion_scale": 1.0,
    "bottom_layer_motion_scale": 0.5,
    "contrast": [0.55, 0.9],
    "brightness_gain": [0.9, 1.1],
    "brightness_offset": [-0.05, 0.05],
    "noise_std": [0.005, 0.03],
    "max_contact_center_radius_mm": 9.75,
    "carrier_contrast": 0.0,
    "radial_distortion_k1": 0.0,
    "outside_intensity": 0.12,
    "label_to_sensor_matrix": [[1.0, 0.0], [0.0, 1.0]],
    "boundary_clamp_power": 2.0,
    "moire_phase_offset_rad": 0.0,
}


def load_config(path=None):
    config = dict(DEFAULT_CONFIG)
    if path:
        config.update(json.loads(Path(path).read_text()))
    if int(config["image_size"]) < 16:
        raise ValueError("image_size must be at least 16")
    if float(config["sensor_radius_mm"]) <= 0:
        raise ValueError("sensor_radius_mm must be positive")
    if not 0.0 <= float(config["outside_intensity"]) <= 1.0:
        raise ValueError("outside_intensity must be between 0 and 1")
    if float(config["max_contact_center_radius_mm"]) <= 0:
        raise ValueError("max_contact_center_radius_mm must be positive")
    clamp_power = float(config["boundary_clamp_power"])
    if not np.isfinite(clamp_power) or clamp_power < 0:
        raise ValueError("boundary_clamp_power must be finite and non-negative")
    if not np.isfinite(float(config["moire_phase_offset_rad"])):
        raise ValueError("moire_phase_offset_rad must be finite")
    matrix = np.asarray(config["label_to_sensor_matrix"], dtype=np.float64)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("label_to_sensor_matrix must be a finite 2x2 matrix")
    if abs(float(np.linalg.det(matrix))) < 1e-8:
        raise ValueError("label_to_sensor_matrix must be invertible")
    _pattern_directions(config["pattern"])
    for name in (
        "pressure_kpa",
        "normal_force_n",
        "contact_radius_mm",
        "contrast",
        "brightness_gain",
        "brightness_offset",
        "noise_std",
    ):
        values = config[name]
        if len(values) != 2 or float(values[0]) > float(values[1]):
            raise ValueError(f"{name} must be [minimum, maximum]")
    return config


def _pattern_directions(pattern):
    patterns = {
        "parallel": (0.0,),
        "cross": (0.0, 90.0),
        "hexagonal": (0.0, 60.0, 120.0),
    }
    try:
        return patterns[str(pattern)]
    except KeyError as exc:
        raise ValueError(
            f"pattern must be one of {sorted(patterns)}, got {pattern!r}"
        ) from exc


def _sample_range(rng, config, name):
    low, high = config[name]
    return float(rng.uniform(low, high))


def _sensor_grid(config):
    size = int(config["image_size"])
    radius = float(config["sensor_radius_mm"])
    axis = np.linspace(-radius, radius, size, dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx, yy, xx * xx + yy * yy <= radius * radius


def _label_to_sensor(config, contact_x_mm, contact_y_mm):
    matrix = np.asarray(config["label_to_sensor_matrix"], dtype=np.float64)
    sensor_x, sensor_y = matrix @ np.asarray(
        (contact_x_mm, contact_y_mm), dtype=np.float64
    )
    return float(sensor_x), float(sensor_y)


def _contact_displacement(config, xx, yy, pressure_kpa, contact):
    contact_x, contact_y, force_n, contact_radius = contact
    dx = xx - contact_x
    dy = yy - contact_y
    gaussian = np.exp(
        -(dx * dx + dy * dy) / (2.0 * contact_radius * contact_radius)
    )
    compliance = force_n / (
        float(config["membrane_stiffness_n_per_mm"])
        * (1.0 + float(config["pressure_stiffening_per_kpa"]) * pressure_kpa)
    )
    ux = compliance * dx / contact_radius * gaussian
    uy = compliance * dy / contact_radius * gaussian

    clamp_power = float(config["boundary_clamp_power"])
    if clamp_power:
        radius = float(config["sensor_radius_mm"])
        edge = np.clip(1.0 - (xx * xx + yy * yy) / (radius * radius), 0.0, 1.0)
        ux *= edge**clamp_power
        uy *= edge**clamp_power
    return ux, uy


def _render_moire(
    config,
    xx,
    yy,
    mask,
    pressure_kpa,
    optical,
    rng,
    contact=None,
    displacement=None,
):
    radius = float(config["sensor_radius_mm"])
    radial_scale = 1.0 + float(config["radial_distortion_k1"]) * (
        xx * xx + yy * yy
    ) / (radius * radius)
    top_x = xx * radial_scale
    top_y = yy * radial_scale
    bottom_x = top_x.copy()
    bottom_y = top_y.copy()

    if contact is not None and displacement is not None:
        raise ValueError("contact and displacement are mutually exclusive")
    if contact is not None or displacement is not None:
        if displacement is None:
            ux, uy = _contact_displacement(
                config, xx, yy, pressure_kpa, contact
            )
        else:
            ux, uy = (np.asarray(value) for value in displacement)
            if ux.shape != xx.shape or uy.shape != yy.shape:
                raise ValueError("displacement arrays must match the sensor grid")
        top_scale = float(config["top_layer_motion_scale"])
        bottom_scale = float(config["bottom_layer_motion_scale"])
        top_x -= top_scale * ux
        top_y -= top_scale * uy
        bottom_x -= bottom_scale * ux
        bottom_y -= bottom_scale * uy

    pressure_strain = pressure_kpa * float(config["pressure_strain_per_kpa"])
    top_x *= 1.0 - pressure_strain
    top_y *= 1.0 - pressure_strain
    bottom_x *= 1.0 + pressure_strain
    bottom_y *= 1.0 + pressure_strain

    pitch = float(config["grating_pitch_mm"])
    components = []
    carrier_components = []
    phase_offset = float(config["moire_phase_offset_rad"])
    for direction in _pattern_directions(config["pattern"]):
        angle_a = math.radians(float(config["grating_angle_a_deg"]) + direction)
        angle_b = math.radians(float(config["grating_angle_b_deg"]) + direction)
        phase_a = (
            2.0
            * np.pi
            * (math.cos(angle_a) * top_x + math.sin(angle_a) * top_y)
            / pitch
        )
        phase_b = (
            2.0
            * np.pi
            * (math.cos(angle_b) * bottom_x + math.sin(angle_b) * bottom_y)
            / pitch
        )
        components.append(np.cos(phase_a - phase_b + phase_offset))
        carrier_components.append(0.5 * (np.cos(phase_a) + np.cos(phase_b)))

    vignette = np.clip(1.0 - 0.25 * (xx * xx + yy * yy) / (radius * radius), 0.65, 1.0)
    signal = optical["contrast"] * np.mean(components, axis=0)
    signal += float(config["carrier_contrast"]) * np.mean(
        carrier_components, axis=0
    )
    image = 0.5 + 0.5 * signal
    image = 0.5 + (image - 0.5) * vignette
    image = image * optical["gain"] + optical["offset"]
    image += rng.normal(0.0, optical["noise_std"], image.shape)
    image[~mask] = float(config["outside_intensity"])
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def simulate_contact_state(
    config,
    contact_x_mm,
    contact_y_mm,
    normal_force_n,
    pressure_kpa,
    contact_radius_mm,
    seed=0,
):
    """Render one deterministic pre/post contact state for interactive use."""
    rng = np.random.default_rng(seed)
    xx, yy, mask = _sensor_grid(config)
    optical = {
        "contrast": sum(config["contrast"]) / 2.0,
        "gain": sum(config["brightness_gain"]) / 2.0,
        "offset": sum(config["brightness_offset"]) / 2.0,
        "noise_std": sum(config["noise_std"]) / 2.0,
    }
    sensor_x, sensor_y = _label_to_sensor(
        config, contact_x_mm, contact_y_mm
    )
    pre_image = _render_moire(
        config, xx, yy, mask, pressure_kpa, optical, rng
    )
    post_image = _render_moire(
        config,
        xx,
        yy,
        mask,
        pressure_kpa,
        optical,
        rng,
        contact=(
            sensor_x,
            sensor_y,
            normal_force_n,
            contact_radius_mm,
        ),
    )
    return {
        "pre_image": pre_image,
        "post_image": post_image,
        "difference": post_image.astype(np.int16) - pre_image.astype(np.int16),
        "sensor_mask": mask,
        "contact_sensor_xy_mm": (sensor_x, sensor_y),
    }


def prepare_raster_display(state):
    """Return PNG/HTML-oriented arrays with physical +y displayed upward."""
    displayed = dict(state)
    for name in ("pre_image", "post_image", "difference", "sensor_mask"):
        displayed[name] = np.flipud(state[name])
    return displayed


def simulate_labeled_dataset(config, targets, conditions, seed):
    targets = np.asarray(targets, dtype=np.float32)
    conditions = np.asarray(conditions, dtype=np.float32)
    if targets.ndim != 2 or targets.shape[1] != len(TARGET_NAMES):
        raise ValueError(f"targets must have shape [N, {len(TARGET_NAMES)}]")
    if conditions.shape != (len(targets), len(CONDITION_NAMES)):
        raise ValueError(
            f"conditions must have shape [N, {len(CONDITION_NAMES)}]"
        )

    rng = np.random.default_rng(seed)
    xx, yy, mask = _sensor_grid(config)
    size = int(config["image_size"])
    images = np.empty((len(targets), 2, size, size), dtype=np.uint8)
    sample_parameters = np.empty(
        (len(targets), len(SAMPLE_PARAMETER_NAMES)), dtype=np.float32
    )
    for index, (contact_x, contact_y, force_n) in enumerate(targets):
        pressure_kpa = float(conditions[index, 0])
        contact_radius = _sample_range(rng, config, "contact_radius_mm")
        optical = {
            "contrast": _sample_range(rng, config, "contrast"),
            "gain": _sample_range(rng, config, "brightness_gain"),
            "offset": _sample_range(rng, config, "brightness_offset"),
            "noise_std": _sample_range(rng, config, "noise_std"),
        }
        sensor_x, sensor_y = _label_to_sensor(config, contact_x, contact_y)
        images[index, 0] = _render_moire(
            config, xx, yy, mask, pressure_kpa, optical, rng
        )
        images[index, 1] = _render_moire(
            config,
            xx,
            yy,
            mask,
            pressure_kpa,
            optical,
            rng,
            contact=(sensor_x, sensor_y, force_n, contact_radius),
        )
        sample_parameters[index] = (
            contact_radius,
            optical["gain"],
            optical["offset"],
            optical["noise_std"],
        )
    return {
        "images": images,
        "targets": targets,
        "conditions": conditions,
        "sample_parameters": sample_parameters,
    }


def simulate_dataset(config, sample_count, seed):
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    targets = np.empty((sample_count, len(TARGET_NAMES)), dtype=np.float32)
    conditions = np.empty((sample_count, len(CONDITION_NAMES)), dtype=np.float32)

    max_center_radius = float(config["max_contact_center_radius_mm"])
    for index in range(sample_count):
        center_radius = max_center_radius * math.sqrt(float(rng.random()))
        center_angle = float(rng.uniform(0.0, 2.0 * np.pi))
        contact_x = center_radius * math.cos(center_angle)
        contact_y = center_radius * math.sin(center_angle)
        pressure_kpa = _sample_range(rng, config, "pressure_kpa")
        force_n = _sample_range(rng, config, "normal_force_n")
        targets[index] = (contact_x, contact_y, force_n)
        conditions[index] = (pressure_kpa,)
    return simulate_labeled_dataset(config, targets, conditions, seed + 1)


def save_dataset(path, data, config, domain, sample_ids=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **data,
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "target_names": np.asarray(TARGET_NAMES),
        "condition_names": np.asarray(CONDITION_NAMES),
        "sample_parameter_names": np.asarray(SAMPLE_PARAMETER_NAMES),
        "domain": np.asarray(domain),
        "config_json": np.asarray(json.dumps(config, sort_keys=True)),
    }
    if sample_ids is not None:
        payload["sample_ids"] = np.asarray(sample_ids)
    np.savez_compressed(path, **payload)


def load_dataset(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {name: archive[name] for name in archive.files}
    if int(data.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported dataset schema in {path}")
    images = data.get("images")
    targets = data.get("targets")
    conditions = data.get("conditions")
    if images is None or images.ndim != 4 or images.shape[1] != 2:
        raise ValueError("images must have shape [N, 2, H, W]")
    if targets is None or targets.shape != (len(images), len(TARGET_NAMES)):
        raise ValueError(f"targets must have shape [N, {len(TARGET_NAMES)}]")
    if conditions is None or conditions.shape != (
        len(images),
        len(CONDITION_NAMES),
    ):
        raise ValueError(f"conditions must have shape [N, {len(CONDITION_NAMES)}]")
    if tuple(data["target_names"].tolist()) != TARGET_NAMES:
        raise ValueError("target_names do not match this platform")
    if tuple(data["condition_names"].tolist()) != CONDITION_NAMES:
        raise ValueError("condition_names do not match this platform")
    return data


def _calibration_indices(targets, sample_count, seed):
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    forces = targets[:, 2]
    radii = np.linalg.norm(targets[:, :2], axis=1)
    positive = forces > 0.03
    if np.count_nonzero(positive) < 3:
        raise ValueError("calibration needs at least three samples above 0.03 N")
    force_edges = np.quantile(forces[positive], (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0))
    rng = np.random.default_rng(seed)
    target_count = min(sample_count, len(targets))
    floor_candidates = np.flatnonzero(forces <= 0.03)
    floor_count = (
        min(len(floor_candidates), max(1, target_count // 10))
        if len(floor_candidates)
        else 0
    )
    positive_count = target_count - floor_count
    per_group, remainder = divmod(positive_count, 9)
    selected = []
    for group_index, (lower_radius, upper_radius) in enumerate(
        ((0.0, 8.0), (8.0, 10.4), (10.4, np.inf))
    ):
        radial = (radii >= lower_radius) & (radii < upper_radius)
        for index in range(3):
            upper_force = forces <= force_edges[index + 1]
            if index < 2:
                upper_force = forces < force_edges[index + 1]
            candidates = np.flatnonzero(
                positive & radial & (forces >= force_edges[index]) & upper_force
            )
            count = per_group + (group_index * 3 + index < remainder)
            if len(candidates) and count:
                selected.extend(
                    rng.choice(candidates, min(count, len(candidates)), replace=False)
                )
    if floor_count:
        selected.extend(
            rng.choice(floor_candidates, floor_count, replace=False)
        )
    selected = np.unique(selected)
    if len(selected) < target_count:
        remaining = np.setdiff1d(np.arange(len(targets)), selected, assume_unique=True)
        selected = np.concatenate(
            (selected, rng.choice(remaining, target_count - len(selected), replace=False))
        )
    return np.sort(selected), force_edges.astype(np.float32)


def _calibration_highpass(image):
    image = image.astype(np.float32) / 255.0
    return cv2.GaussianBlur(image, (0, 0), 1.5) - cv2.GaussianBlur(
        image, (0, 0), 12.0
    )


def _dominant_moire_peak(image, mask):
    size = image.shape[0]
    window = np.outer(np.hanning(size), np.hanning(size))
    spectrum = np.abs(
        np.fft.fftshift(np.fft.fft2(_calibration_highpass(image) * window * mask))
    ) ** 2
    yy, xx = np.ogrid[:size, :size]
    radial_frequency = np.hypot(yy - size // 2, xx - size // 2)
    spectrum[(radial_frequency < 2.0) | (radial_frequency > size / 4.0)] = 0.0
    peak_y, peak_x = np.unravel_index(np.argmax(spectrum), spectrum.shape)
    peak = (int(peak_x - size // 2), int(peak_y - size // 2))
    # A real-valued image has equal conjugate peaks. Use one orientation so
    # train and held-out phase maps share the same complex carrier.
    if peak[1] < 0 or (peak[1] == 0 and peak[0] < 0):
        return -peak[0], -peak[1]
    return peak


def _moire_component(image, peak, mask, sigma=None):
    size = image.shape[0]
    window = np.outer(np.hanning(size), np.hanning(size))
    spectrum = np.fft.fftshift(
        np.fft.fft2(image.astype(np.float32) / 255.0 * window * mask)
    )
    yy, xx = np.ogrid[:size, :size]
    sigma = max(1.0, size / 128.0) if sigma is None else float(sigma)
    if sigma <= 0.0:
        raise ValueError("moire component sigma must be positive")
    band = np.exp(
        -(
            (xx - size // 2 - peak[0]) ** 2
            + (yy - size // 2 - peak[1]) ** 2
        )
        / (2.0 * sigma * sigma)
    )
    return np.fft.ifft2(np.fft.ifftshift(spectrum * band))


def _remove_phase_background(phase, mask):
    sigma = max(4.0, phase.shape[0] / 12.0)
    denominator = np.maximum(
        cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigma), 1e-8
    )
    trend = np.arctan2(
        cv2.GaussianBlur(np.sin(phase) * mask, (0, 0), sigma) / denominator,
        cv2.GaussianBlur(np.cos(phase) * mask, (0, 0), sigma) / denominator,
    )
    # ponytail: removes unmodeled pairwise drift; replace with a calibrated mode if it becomes repeatable.
    return np.angle(np.exp(1j * (phase - trend)))


def _phase_delta_maps(images, peak, mask, shared_pre=False):
    phase_maps = np.empty(images.shape[:1] + images.shape[-2:], dtype=np.float32)
    weights = np.empty_like(phase_maps)
    reference = _moire_component(images[0, 0], peak, mask) if shared_pre else None
    for index, (pre_image, post_image) in enumerate(images):
        pre_component = (
            reference if reference is not None else _moire_component(pre_image, peak, mask)
        )
        post_component = _moire_component(post_image, peak, mask)
        phase = np.angle(post_component * np.conj(pre_component))
        weight = np.abs(post_component) * np.abs(pre_component) * mask
        global_phase = np.angle(np.sum(weight * np.exp(1j * phase)))
        phase_maps[index] = _remove_phase_background(phase - global_phase, mask)
        weights[index] = weight
    return phase_maps, weights


def _fixed_calibration_config(config):
    fixed = dict(config)
    for name in (
        "contact_radius_mm",
        "contrast",
        "brightness_gain",
        "brightness_offset",
    ):
        value = sum(config[name]) / 2.0
        fixed[name] = [value, value]
    fixed["noise_std"] = [0.0, 0.0]
    return fixed


def _masked_correlation(left, right, mask):
    left = left[mask].astype(np.float64)
    right = right[mask].astype(np.float64)
    left -= left.mean()
    right -= right.mean()
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    return 0.0 if denominator == 0.0 else float(left @ right / denominator)


def _fit_moire_phase_offset(mean_pre_image, config, mask, phase_steps=64):
    real_highpass = _calibration_highpass(mean_pre_image)
    base_offset = float(config["moire_phase_offset_rad"])
    best = None
    for delta in np.linspace(-np.pi, np.pi, phase_steps, endpoint=False):
        phase_offset = (base_offset + delta + np.pi) % (2.0 * np.pi) - np.pi
        candidate = _fixed_calibration_config(
            dict(config, moire_phase_offset_rad=float(phase_offset))
        )
        simulated = simulate_contact_state(
            candidate,
            0.0,
            0.0,
            0.0,
            0.0,
            sum(candidate["contact_radius_mm"]) / 2.0,
        )["pre_image"]
        correlation = _masked_correlation(
            real_highpass, _calibration_highpass(simulated), mask
        )
        if best is None or correlation > best[0]:
            best = (correlation, float(phase_offset))
    return best


def _fit_moire_optics(mean_pre_image, config, mask):
    initial_a = float(config["grating_angle_a_deg"])
    initial_relative = float(config["grating_angle_b_deg"]) - initial_a
    best = None
    for common_shift in np.linspace(-2.0, 2.0, 9):
        for relative_shift in np.linspace(-1.0, 1.0, 9):
            candidate = dict(
                config,
                grating_angle_a_deg=float(initial_a + common_shift),
                grating_angle_b_deg=float(
                    initial_a + common_shift + initial_relative + relative_shift
                ),
            )
            correlation, _ = _fit_moire_phase_offset(
                mean_pre_image, candidate, mask, phase_steps=8
            )
            if best is None or correlation > best[0]:
                best = (correlation, candidate)
    correlation, phase_offset = _fit_moire_phase_offset(
        mean_pre_image, best[1], mask
    )
    return dict(best[1], moire_phase_offset_rad=phase_offset), correlation


def _prepare_phase_target(data, config, sample_count, seed):
    if data["images"].shape[-1] != int(config["image_size"]):
        raise ValueError("calibration data image size must match the config")
    if "calibration_force_edges" in data:
        indices = np.arange(len(data["targets"]))
        force_edges = np.asarray(data["calibration_force_edges"], dtype=np.float32)
    else:
        indices, force_edges = _calibration_indices(data["targets"], sample_count, seed)
    images = data["images"][indices]
    targets = data["targets"][indices]
    conditions = data["conditions"][indices]
    xx, yy, mask = _sensor_grid(config)
    radial = np.hypot(xx, yy)
    mask &= radial < 13.5
    contact_xy = np.asarray(
        [
            _label_to_sensor(config, contact_x, contact_y)
            for contact_x, contact_y in targets[:, :2]
        ],
        dtype=np.float32,
    )
    contact_radius = sum(config["contact_radius_mm"]) / 2.0
    local_distance = np.hypot(
        xx[None, :, :] - contact_xy[:, 0, None, None],
        yy[None, :, :] - contact_xy[:, 1, None, None],
    )
    local_masks = (
        mask[None, :, :]
        & (local_distance >= 0.5 * contact_radius)
        & (local_distance <= 2.0 * contact_radius)
    )
    if not np.all(local_masks.any(axis=(1, 2))):
        raise ValueError("calibration contact ROI falls outside the sensor")
    real_peak = _dominant_moire_peak(images[:, 0].mean(axis=0), mask)
    real_phase, real_weights = _phase_delta_maps(images, real_peak, mask)
    return {
        "images": images,
        "targets": targets,
        "conditions": conditions,
        "force_edges": force_edges,
        "mask": mask,
        "contact_radius_mm": np.hypot(contact_xy[:, 0], contact_xy[:, 1]),
        "local_masks": local_masks,
        "contact_roi_annulus_mm": [
            float(0.5 * contact_radius),
            float(2.0 * contact_radius),
        ],
        "real_peak": real_peak,
        "real_phase": real_phase,
        "real_weights": real_weights,
    }


def _phase_groups(prepared):
    force = prepared["targets"][:, 2]
    contact_radius = prepared["contact_radius_mm"]
    for index in range(3):
        in_bin = force >= prepared["force_edges"][index]
        if index == 2:
            in_bin &= force <= prepared["force_edges"][index + 1]
        else:
            in_bin &= force < prepared["force_edges"][index + 1]
        for name, region in (
            ("contact_inner", contact_radius < 8.0),
            ("contact_edge", contact_radius >= 10.4),
        ):
            group = in_bin & region
            if np.any(group):
                yield f"force_bin_{index}_{name}", group


def _phase_score(prepared, simulated_phase):
    phase_error = 1.0 - np.cos(prepared["real_phase"] - simulated_phase)
    groups = {}
    scores = []
    for name, group in _phase_groups(prepared):
        sample_scores = []
        for index in np.flatnonzero(group):
            local_mask = prepared["local_masks"][index]
            weights = prepared["real_weights"][index, local_mask]
            if float(weights.sum()) != 0.0:
                sample_scores.append(
                    float(
                        (phase_error[index, local_mask] * weights).sum()
                        / weights.sum()
                    )
                )
        if sample_scores:
            value = float(np.median(sample_scores))
            groups[name] = value
            scores.append(value)
    if not scores:
        raise ValueError("calibration split has no positive-force phase bins")
    return float(np.mean(scores)), groups


def _local_phase_amplitudes(phase_maps, local_masks):
    return np.asarray(
        [
            np.median(np.abs(phase_map[local_mask]))
            for phase_map, local_mask in zip(phase_maps, local_masks)
        ],
        dtype=np.float32,
    )


def _phase_response_by_group(phase_maps, prepared):
    force = prepared["targets"][:, 2]
    amplitudes = _local_phase_amplitudes(phase_maps, prepared["local_masks"])
    floor_samples = force <= 0.03
    floor = (
        float(np.median(amplitudes[floor_samples]))
        if np.any(floor_samples)
        else 0.0
    )
    responses = {}
    for name, group in _phase_groups(prepared):
        response = float(np.median(amplitudes[group]))
        responses[name] = max(response - floor, 0.003)
    return floor, responses


def _phase_response_loss(prepared, simulated_phase):
    _, real_response = _phase_response_by_group(
        prepared["real_phase"], prepared
    )
    _, sim_response = _phase_response_by_group(
        simulated_phase, prepared
    )
    common = sorted(set(real_response) & set(sim_response))
    loss = float(
        np.mean(
            [
                abs(
                    math.log(
                        (sim_response[name] + 0.01) / (real_response[name] + 0.01)
                    )
                )
                for name in common
            ]
        )
    )
    return loss, {"real": real_response, "sim": sim_response}


def _phase_candidate(prepared, config):
    fixed_config = _fixed_calibration_config(config)
    simulated = simulate_labeled_dataset(
        fixed_config,
        prepared["targets"],
        prepared["conditions"],
        seed=0,
    )
    sim_peak = _dominant_moire_peak(simulated["images"][0, 0], prepared["mask"])
    sim_phase, _ = _phase_delta_maps(
        simulated["images"], sim_peak, prepared["mask"], shared_pre=True
    )
    score, groups = _phase_score(prepared, sim_phase)
    response_loss, response = _phase_response_loss(prepared, sim_phase)
    return {
        "phase_loss": score,
        "phase_loss_by_group": groups,
        "response_loss": response_loss,
        "phase_response_rad": response,
        "sim_peak_cycles_per_image": list(sim_peak),
    }


def evaluate_phase_alignment(data, config, sample_count=144, seed=7):
    prepared = _prepare_phase_target(data, config, sample_count, seed)
    report = _phase_candidate(prepared, config)
    report.update(
        {
            "samples": int(len(prepared["targets"])),
            "real_peak_cycles_per_image": list(prepared["real_peak"]),
        }
    )
    return report


def calibrate_pos_force(data, config, sample_count=144, seed=7):
    prepared = _prepare_phase_target(data, config, sample_count, seed)
    optical_config, correlation = _fit_moire_optics(
        prepared["images"][:, 0].mean(axis=0), config, prepared["mask"]
    )
    candidates = {
        (
            float(config["boundary_clamp_power"]),
            float(config["membrane_stiffness_n_per_mm"]),
        )
    }
    candidates.update(
        (clamp_power, stiffness)
        for clamp_power in (1.0, 2.0)
        for stiffness in (
            0.02,
            0.03,
            0.04,
            0.06,
            0.08,
            0.10,
            0.12,
            0.15,
            0.20,
            0.35,
        )
    )
    reports = []
    for clamp_power, stiffness in sorted(candidates):
        candidate = dict(
            optical_config,
            boundary_clamp_power=clamp_power,
            membrane_stiffness_n_per_mm=stiffness,
        )
        report = _phase_candidate(prepared, candidate)
        report.update(
            {
                "boundary_clamp_power": clamp_power,
                "membrane_stiffness_n_per_mm": stiffness,
            }
        )
        reports.append(report)
    selected = min(
        reports,
        key=lambda report: (report["phase_loss"], report["response_loss"]),
    )
    calibrated = dict(
        optical_config,
        boundary_clamp_power=selected["boundary_clamp_power"],
        membrane_stiffness_n_per_mm=selected["membrane_stiffness_n_per_mm"],
    )
    baseline = next(
        report
        for report in reports
        if report["boundary_clamp_power"] == float(config["boundary_clamp_power"])
        and report["membrane_stiffness_n_per_mm"]
        == float(config["membrane_stiffness_n_per_mm"])
    )
    best_by_boundary = {}
    for clamp_power in (1.0, 2.0):
        best = min(
            (
                report
                for report in reports
                if report["boundary_clamp_power"] == clamp_power
            ),
            key=lambda report: (report["phase_loss"], report["response_loss"]),
        )
        best_by_boundary[str(int(clamp_power))] = {
            "membrane_stiffness_n_per_mm": best["membrane_stiffness_n_per_mm"],
            "phase_loss": best["phase_loss"],
            "response_loss": best["response_loss"],
        }
    return calibrated, {
        "fit_samples": int(len(prepared["targets"])),
        "fit_uses_test_data": False,
        "selection_metric": "contact_roi_circular_phase_loss",
        "force_bin_edges_n": [float(value) for value in prepared["force_edges"]],
        "contact_roi_annulus_mm": prepared["contact_roi_annulus_mm"],
        "real_peak_cycles_per_image": list(prepared["real_peak"]),
        "grating_angle_a_deg": calibrated["grating_angle_a_deg"],
        "grating_angle_b_deg": calibrated["grating_angle_b_deg"],
        "moire_phase_offset_rad": calibrated["moire_phase_offset_rad"],
        "pre_phase_correlation": correlation,
        "baseline": baseline,
        "selected": selected,
        "best_candidate_by_boundary": best_by_boundary,
        "candidates": reports,
    }


def _features(images, conditions, feature_size=8):
    axis = np.linspace(-1.0, 1.0, images.shape[-1], dtype=np.float32)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    features = np.empty((len(images), feature_size * feature_size + 5), np.float32)
    scale = 255.0 if images.max(initial=0) > 1 else 1.0
    for index, pair in enumerate(images):
        energy = np.abs(
            pair[1].astype(np.float32) - pair[0].astype(np.float32)
        ) / scale
        small = cv2.resize(
            energy, (feature_size, feature_size), interpolation=cv2.INTER_AREA
        )
        mass = float(energy.sum()) + 1e-8
        center_x = float((energy * xx).sum() / mass)
        center_y = float((energy * yy).sum() / mass)
        squared_distance = (xx - center_x) ** 2 + (yy - center_y) ** 2
        spread = float((energy * squared_distance).sum() / mass)
        features[index, : feature_size * feature_size] = small.ravel()
        features[index, feature_size * feature_size : -1] = (
            mass / energy.size,
            center_x,
            center_y,
            spread,
        )
        features[index, -1] = conditions[index, 0]
    return features


def fit_model(datasets, ridge=1.0, feature_size=8):
    if len(datasets) == 1:
        images = datasets[0]["images"]
        targets = datasets[0]["targets"].astype(np.float64)
        conditions = datasets[0]["conditions"]
    else:
        images = np.concatenate([data["images"] for data in datasets])
        targets = np.concatenate([data["targets"] for data in datasets]).astype(
            np.float64
        )
        conditions = np.concatenate([data["conditions"] for data in datasets])
    features = _features(images, conditions, feature_size).astype(np.float64)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale < 1e-8] = 1.0
    normalized = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    system = normalized.T @ normalized + ridge * np.eye(normalized.shape[1])
    weights = np.linalg.solve(system, normalized.T @ (targets - target_mean))
    return {
        "weights": weights.astype(np.float32),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_scale": feature_scale.astype(np.float32),
        "target_mean": target_mean.astype(np.float32),
        "feature_size": np.asarray(feature_size, dtype=np.int32),
        "target_names": np.asarray(TARGET_NAMES),
        "condition_names": np.asarray(CONDITION_NAMES),
    }


def save_model(path, model):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **model)


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def predict(model, dataset):
    if tuple(model["target_names"].tolist()) != TARGET_NAMES:
        raise ValueError("model target_names do not match this platform")
    features = _features(
        dataset["images"],
        dataset["conditions"],
        int(model["feature_size"]),
    )
    normalized = (features - model["feature_mean"]) / model["feature_scale"]
    return normalized @ model["weights"] + model["target_mean"]


def evaluate(model, dataset):
    predictions = predict(model, dataset)
    error = predictions - dataset["targets"]
    return {
        "samples": int(len(predictions)),
        "mae": {
            name: float(value)
            for name, value in zip(TARGET_NAMES, np.mean(np.abs(error), axis=0))
        },
        "rmse": {
            name: float(value)
            for name, value in zip(
                TARGET_NAMES, np.sqrt(np.mean(error * error, axis=0))
            )
        },
    }


def pack_real_dataset(manifest_path, output_path, config):
    manifest_path = Path(manifest_path)
    required = {
        "sample_id",
        "pre_image",
        "post_image",
        *TARGET_NAMES,
        *CONDITION_NAMES,
    }
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"real manifest missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("real manifest is empty")

    size = int(config["image_size"])
    images = np.empty((len(rows), 2, size, size), dtype=np.uint8)
    targets = np.empty((len(rows), len(TARGET_NAMES)), dtype=np.float32)
    conditions = np.empty((len(rows), len(CONDITION_NAMES)), dtype=np.float32)
    sample_parameters = np.empty(
        (len(rows), len(SAMPLE_PARAMETER_NAMES)), dtype=np.float32
    )
    sample_ids = []
    for index, row in enumerate(rows):
        sample_ids.append(row["sample_id"])
        for channel, column in enumerate(("pre_image", "post_image")):
            image_path = (manifest_path.parent / row[column]).resolve()
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cannot read image: {image_path}")
            images[index, channel] = cv2.resize(
                image, (size, size), interpolation=cv2.INTER_AREA
            )
        targets[index] = tuple(float(row[name]) for name in TARGET_NAMES)
        conditions[index] = tuple(float(row[name]) for name in CONDITION_NAMES)
        sample_parameters[index] = tuple(
            float(row[name]) if row.get(name) not in (None, "") else float("nan")
            for name in SAMPLE_PARAMETER_NAMES
        )

    save_dataset(
        output_path,
        {
            "images": images,
            "targets": targets,
            "conditions": conditions,
            "sample_parameters": sample_parameters,
        },
        config,
        domain="real",
        sample_ids=sample_ids,
    )


def load_pos_force_labels(dataset_dir):
    dataset_dir = Path(dataset_dir)
    csv_paths = sorted(dataset_dir.glob("ground_truth_Pos_Force_0.4N_*.csv"))
    if len(csv_paths) != 1:
        raise ValueError(
            f"expected one Pos_Force ground-truth CSV in {dataset_dir}, "
            f"found {len(csv_paths)}"
        )
    with csv_paths[0].open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"Session", "X", "Y", "Force"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Pos_Force CSV missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("Pos_Force CSV is empty")

    sample_ids = [row["Session"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Pos_Force Session values must be unique")
    targets = np.asarray(
        [
            (
                float(row["X"]) - 26.0,
                float(row["Y"]) + 26.0,
                max(0.0, float(row["Force"])) * GRAM_FORCE_TO_NEWTON,
            )
            for row in rows
        ],
        dtype=np.float32,
    )
    conditions = np.zeros((len(rows), len(CONDITION_NAMES)), dtype=np.float32)
    return sample_ids, targets, conditions


def _load_pos_force_images(dataset_dir, sample_ids, size, indices):
    dataset_dir = Path(dataset_dir)
    indices = np.asarray(indices, dtype=np.int64)
    images = np.empty((len(indices), 2, size, size), dtype=np.uint8)
    for output_index, source_index in enumerate(indices):
        sample_id = sample_ids[source_index]
        for channel, suffix in enumerate(("_initial.jpg", ".jpg")):
            image_path = dataset_dir / f"{sample_id}{suffix}"
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"cannot read image: {image_path}")
            images[output_index, channel] = cv2.resize(
                image, (size, size), interpolation=cv2.INTER_AREA
            )
    return images


def load_pos_force_calibration_dataset(dataset_dir, config, sample_count, seed):
    sample_ids, targets, conditions = load_pos_force_labels(dataset_dir)
    indices, force_edges = _calibration_indices(targets, sample_count, seed)
    return {
        "images": _load_pos_force_images(
            dataset_dir, sample_ids, int(config["image_size"]), indices
        ),
        "targets": targets[indices],
        "conditions": conditions[indices],
        "calibration_force_edges": force_edges,
    }


def pack_pos_force_dataset(dataset_dir, output_path, config):
    dataset_dir = Path(dataset_dir)
    sample_ids, targets, conditions = load_pos_force_labels(dataset_dir)
    size = int(config["image_size"])
    images = _load_pos_force_images(
        dataset_dir, sample_ids, size, np.arange(len(sample_ids))
    )
    sample_parameters = np.full(
        (len(sample_ids), len(SAMPLE_PARAMETER_NAMES)), np.nan, dtype=np.float32
    )
    save_dataset(
        output_path,
        {
            "images": images,
            "targets": targets,
            "conditions": conditions,
            "sample_parameters": sample_parameters,
        },
        config,
        domain="real",
        sample_ids=sample_ids,
    )
    return sample_ids, targets, conditions


def _command_simulate(args):
    config = load_config(args.config)
    output = Path(args.output_dir)
    save_dataset(
        output / "train.npz",
        simulate_dataset(config, args.train_samples, args.seed),
        config,
        domain="sim",
    )
    save_dataset(
        output / "test.npz",
        simulate_dataset(config, args.test_samples, args.seed + 1),
        config,
        domain="sim",
    )
    print(output.resolve())


def _command_train(args):
    datasets = [load_dataset(path) for path in args.data]
    model = fit_model(datasets, ridge=args.ridge)
    save_model(args.model, model)
    print(Path(args.model).resolve())


def _command_test(args):
    metrics = evaluate(load_model(args.model), load_dataset(args.data))
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    if args.metrics:
        metrics_path = Path(args.metrics)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(rendered + "\n")
    print(rendered)


def _command_pack_real(args):
    pack_real_dataset(args.manifest, args.output, load_config(args.config))
    print(Path(args.output).resolve())


def _command_align_pos_force(args):
    config = load_config(args.config)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for offset, (name, dataset_dir) in enumerate(
        (("train", args.train_dir), ("test", args.test_dir))
    ):
        sample_ids, targets, conditions = pack_pos_force_dataset(
            dataset_dir, output / f"real_{name}.npz", config
        )
        save_dataset(
            output / f"sim_{name}.npz",
            simulate_labeled_dataset(
                config, targets, conditions, seed=args.seed + offset
            ),
            config,
            domain="sim",
            sample_ids=sample_ids,
        )
        radii = np.linalg.norm(targets[:, :2], axis=1)
        summaries[name] = {
            "samples": len(sample_ids),
            "position_radius_mm": [float(radii.min()), float(radii.max())],
            "force_n": [float(targets[:, 2].min()), float(targets[:, 2].max())],
        }
    summary = {
        "force_source_unit": "gram-force",
        "force_conversion_n_per_g": GRAM_FORCE_TO_NEWTON,
        "position_centering": {"x_mm": "X - 26", "y_mm": "Y + 26"},
        "pressure_kpa": 0.0,
        "splits": summaries,
    }
    (output / "alignment_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(output.resolve())


def _command_calibrate_pos_force(args):
    config = load_config(args.config)
    if args.dataset_dir:
        calibration_data = load_pos_force_calibration_dataset(
            args.dataset_dir, config, args.samples, args.seed
        )
        calibration_samples = len(calibration_data["images"])
    else:
        calibration_data = load_dataset(args.data)
        if str(calibration_data["domain"].item()) != "real":
            raise ValueError("calibrate-pos-force requires a packed real dataset")
        calibration_samples = args.samples
    calibrated, report = calibrate_pos_force(
        calibration_data, config, sample_count=calibration_samples, seed=args.seed
    )
    output_config = Path(args.output_config)
    output_config.parent.mkdir(parents=True, exist_ok=True)
    output_config.write_text(json.dumps(calibrated, indent=2, sort_keys=True) + "\n")
    if args.evaluation_data or args.evaluation_dir:
        del calibration_data
        if args.evaluation_dir:
            evaluation_data = load_pos_force_calibration_dataset(
                args.evaluation_dir, calibrated, args.samples, args.seed
            )
            evaluation_samples = len(evaluation_data["images"])
        else:
            evaluation_data = load_dataset(args.evaluation_data)
            if str(evaluation_data["domain"].item()) != "real":
                raise ValueError("evaluation data must be a packed real dataset")
            evaluation_samples = args.samples
        report["held_out"] = evaluate_phase_alignment(
            evaluation_data,
            calibrated,
            sample_count=evaluation_samples,
            seed=args.seed,
        )
        report["held_out"]["used_for_fit"] = False
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"config": str(output_config.resolve()), "report": str(report_path.resolve())}))


def build_parser():
    parser = argparse.ArgumentParser(
        description="Minimal MoireSkin simulate -> train -> test platform"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    simulate = subparsers.add_parser("simulate", help="generate train/test data")
    simulate.add_argument("--output-dir", required=True)
    simulate.add_argument("--config")
    simulate.add_argument("--train-samples", type=int, default=1000)
    simulate.add_argument("--test-samples", type=int, default=200)
    simulate.add_argument("--seed", type=int, default=7)
    simulate.set_defaults(run=_command_simulate)

    train = subparsers.add_parser("train", help="train the baseline regressor")
    train.add_argument("--data", nargs="+", required=True)
    train.add_argument("--model", required=True)
    train.add_argument("--ridge", type=float, default=1.0)
    train.set_defaults(run=_command_train)

    test = subparsers.add_parser("test", help="evaluate a saved model")
    test.add_argument("--data", required=True)
    test.add_argument("--model", required=True)
    test.add_argument("--metrics")
    test.set_defaults(run=_command_test)

    pack_real = subparsers.add_parser(
        "pack-real", help="convert paired real images into the same dataset schema"
    )
    pack_real.add_argument("--manifest", required=True)
    pack_real.add_argument("--output", required=True)
    pack_real.add_argument("--config")
    pack_real.set_defaults(run=_command_pack_real)

    align = subparsers.add_parser(
        "align-pos-force",
        help="pack the real 0.4 N splits and simulate their exact labels",
    )
    align.add_argument("--train-dir", required=True)
    align.add_argument("--test-dir", required=True)
    align.add_argument("--output-dir", required=True)
    align.add_argument("--config", required=True)
    align.add_argument("--seed", type=int, default=7)
    align.set_defaults(run=_command_align_pos_force)

    calibrate = subparsers.add_parser(
        "calibrate-pos-force",
        help="fit phase-alignment boundary parameters on real Pos_Force training data",
    )
    source = calibrate.add_mutually_exclusive_group(required=True)
    source.add_argument("--data")
    source.add_argument("--dataset-dir")
    calibrate.add_argument("--config", required=True)
    calibrate.add_argument("--output-config", required=True)
    calibrate.add_argument("--report", required=True)
    evaluation = calibrate.add_mutually_exclusive_group()
    evaluation.add_argument("--evaluation-data")
    evaluation.add_argument("--evaluation-dir")
    calibrate.add_argument("--samples", type=int, default=144)
    calibrate.add_argument("--seed", type=int, default=7)
    calibrate.set_defaults(run=_command_calibrate_pos_force)
    return parser


def main():
    args = build_parser().parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
