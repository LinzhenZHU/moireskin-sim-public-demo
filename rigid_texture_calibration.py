#!/usr/bin/env python3
"""Confidence and sim-to-real calibration contracts for rigid texture POC v8."""

import hashlib
import json
from pathlib import Path

import numpy as np

from rigid_object_poc import DEFAULT_MECHANICAL_MTF_CALIBRATION


CALIBRATION_SCHEMA_VERSION = 1


def _pool_adjacent_violators(values, weights, increasing):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sign = 1.0 if increasing else -1.0
    blocks = []
    for index, (value, weight) in enumerate(zip(sign * values, weights)):
        blocks.append([index, index + 1, max(float(weight), 1e-8), float(value)])
        while len(blocks) >= 2 and blocks[-2][3] > blocks[-1][3]:
            right = blocks.pop()
            left = blocks.pop()
            total_weight = left[2] + right[2]
            blocks.append(
                [
                    left[0],
                    right[1],
                    total_weight,
                    (left[2] * left[3] + right[2] * right[3]) / total_weight,
                ]
            )
    fitted = np.empty_like(values)
    for start, end, _, value in blocks:
        fitted[start:end] = sign * value
    return fitted


def _fit_binned_monotonic(raw_confidence, target, bins, increasing):
    raw_confidence = np.asarray(raw_confidence, dtype=np.float64).ravel()
    target = np.asarray(target, dtype=np.float64).ravel()
    valid = np.isfinite(raw_confidence) & np.isfinite(target)
    raw_confidence = np.clip(raw_confidence[valid], 0.0, 1.0)
    target = target[valid]
    if len(raw_confidence) < bins:
        raise ValueError("confidence calibration needs at least one sample per bin")
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    indices = np.minimum(
        np.searchsorted(edges, raw_confidence, side="right") - 1,
        int(bins) - 1,
    )
    counts = np.bincount(indices, minlength=int(bins)).astype(np.float64)
    sums = np.bincount(indices, weights=target, minlength=int(bins))
    means = np.divide(sums, counts, out=np.full(int(bins), np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    populated = np.flatnonzero(np.isfinite(means))
    if not len(populated):
        raise ValueError("confidence calibration has no finite bins")
    means = np.interp(
        np.arange(int(bins)), populated, means[populated]
    )
    fitted = _pool_adjacent_violators(
        means, np.maximum(counts, 1e-6), increasing=increasing
    )
    axis = np.concatenate(([0.0], centers, [1.0]))
    outputs = np.concatenate(([fitted[0]], fitted, [fitted[-1]]))
    return axis, outputs, counts.astype(np.int64)


def fit_confidence_calibration(
    appearance_raw_confidence,
    appearance_helpful,
    reconstruction_raw_confidence,
    absolute_error_mm,
    bins=8,
):
    """Fit monotone helpfulness and expected-error lookup tables."""
    helpful_axis, helpful_probability, helpful_counts = _fit_binned_monotonic(
        appearance_raw_confidence,
        np.asarray(appearance_helpful, dtype=np.float32),
        bins,
        increasing=True,
    )
    error_axis, expected_error, error_counts = _fit_binned_monotonic(
        reconstruction_raw_confidence,
        absolute_error_mm,
        bins,
        increasing=False,
    )
    errors = np.asarray(absolute_error_mm, dtype=np.float64)
    error_scale = max(float(np.quantile(errors[np.isfinite(errors)], 0.95)), 1e-6)
    calibrated_confidence = np.clip(1.0 - expected_error / error_scale, 0.0, 1.0)
    calibrated_confidence = np.maximum.accumulate(calibrated_confidence)
    return {
        "appearance_helpfulness": {
            "raw_confidence": helpful_axis.tolist(),
            "calibrated_probability": np.clip(
                helpful_probability, 0.0, 1.0
            ).tolist(),
            "bin_counts": helpful_counts.tolist(),
        },
        "reconstruction_error": {
            "raw_confidence": error_axis.tolist(),
            "expected_error_mm": np.maximum(expected_error, 0.0).tolist(),
            "calibrated_confidence": calibrated_confidence.tolist(),
            "bin_counts": error_counts.tolist(),
            "error_scale_mm": error_scale,
        },
    }


def evaluate_confidence_calibration(
    calibration,
    appearance_raw_confidence,
    appearance_helpful,
    reconstruction_raw_confidence,
    absolute_error_mm,
):
    appearance_raw_confidence = np.asarray(
        appearance_raw_confidence, dtype=np.float64
    )
    appearance_helpful = np.asarray(appearance_helpful, dtype=np.float64)
    reconstruction_raw_confidence = np.asarray(
        reconstruction_raw_confidence, dtype=np.float64
    )
    absolute_error_mm = np.asarray(absolute_error_mm, dtype=np.float64)
    appearance = calibration["appearance_helpfulness"]
    reconstruction = calibration["reconstruction_error"]
    calibrated_probability = np.interp(
        appearance_raw_confidence,
        appearance["raw_confidence"],
        appearance["calibrated_probability"],
    )
    calibrated_error = np.interp(
        reconstruction_raw_confidence,
        reconstruction["raw_confidence"],
        reconstruction["expected_error_mm"],
    )
    raw_error_guess = 0.12 * (1.0 - reconstruction_raw_confidence)
    return {
        "appearance_brier_raw": float(
            np.mean((appearance_raw_confidence - appearance_helpful) ** 2)
        ),
        "appearance_brier_calibrated": float(
            np.mean((calibrated_probability - appearance_helpful) ** 2)
        ),
        "error_mae_raw_mm": float(
            np.mean(np.abs(raw_error_guess - absolute_error_mm))
        ),
        "error_mae_calibrated_mm": float(
            np.mean(np.abs(calibrated_error - absolute_error_mm))
        ),
    }


def default_calibration_bundle(config, confidence_calibration=None):
    """Return a versioned synthetic bundle that real calibration can replace."""
    config = dict(config, pattern="cross")
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    if confidence_calibration is None:
        confidence_calibration = {
            "appearance_helpfulness": {
                "raw_confidence": [0.0, 1.0],
                "calibrated_probability": [0.0, 1.0],
                "bin_counts": [0, 0],
            },
            "reconstruction_error": {
                "raw_confidence": [0.0, 1.0],
                "expected_error_mm": [0.12, 0.0],
                "calibrated_confidence": [0.0, 1.0],
                "bin_counts": [0, 0],
                "error_scale_mm": 0.12,
            },
        }
    radius_mm = float(config["sensor_radius_mm"])
    image_size = int(config["image_size"])
    bundle = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "bundle_type": "moireskin_rigid_texture_calibration",
        "calibration_id": f"synthetic-v8-{config_hash}",
        "compatible_model_version": 8,
        "source_domain": "synthetic",
        "units": {
            "length": "mm",
            "spatial_frequency": "cycles/mm",
            "angle": "degree",
            "pressure": "kPa",
            "force": "N",
            "intensity": "normalized_0_1",
        },
        "coordinates": {
            "origin": "sensor_center",
            "x_axis": "positive_image_column_right",
            "y_axis": "positive_physical_up",
            "z_axis": "positive_toward_contact_object",
            "array_row_zero": "y=-sensor_radius_mm",
            "display_transform": "flipud_once",
        },
        "grating": {
            "pattern": "cross",
            "directions_deg": [0.0, 90.0],
            "angle_a_deg": float(config["grating_angle_a_deg"]),
            "angle_b_deg": float(config["grating_angle_b_deg"]),
            "pitch_mm": float(config["grating_pitch_mm"]),
            "open_fraction": 0.82,
            "line_transmittance": 0.10,
            "top_layer_motion_scale": float(config["top_layer_motion_scale"]),
            "bottom_layer_motion_scale": float(
                config["bottom_layer_motion_scale"]
            ),
        },
        "camera": {
            "physics_image_size_px": image_size,
            "raw_supersample": 2,
            "physics_pixels_per_mm": (image_size - 1) / (2.0 * radius_mm),
            "raw_pixels_per_mm": (2 * image_size - 1) / (2.0 * radius_mm),
            "psf_sigma_px": 0.45,
            "read_noise_std": 0.004,
            "exposure_scale": 1.0,
            "response_gamma": 1.0,
            "brightness_gain": list(config["brightness_gain"]),
            "brightness_offset": list(config["brightness_offset"]),
            "radial_distortion_k1": float(config["radial_distortion_k1"]),
        },
        "membrane": {
            "tension_n_per_mm": 0.08,
            "bending_stiffness_n_mm": 1e-4,
            "cavity_depth_mm": 8.0,
            "sealed_air_coupling": True,
            "reference_inflation_pressure_kpa": 4.0,
            "pressure_volume_curve": {
                "model": "isothermal_ideal_gas",
                "ambient_pressure_kpa": 101.325,
                "maximum_volume_change_fraction": 0.35,
            },
            "mechanical_mtf": {
                key: list(value) if isinstance(value, tuple) else value
                for key, value in DEFAULT_MECHANICAL_MTF_CALIBRATION.items()
            },
        },
        "inverse": {
            "slope_to_shift_mm": 0.12,
            "carrier_to_displacement_scale": 1.0,
            "carrier_highpass_mm": 1.2,
            "abstention_threshold": 0.20,
            "confidence_calibration": confidence_calibration,
        },
        "calibration_confidence": {
            "grating": 1.0,
            "camera": 1.0,
            "membrane": 0.75,
            "inverse": 0.75,
        },
        "parameter_snapshot": dict(config),
    }
    validate_calibration_bundle(bundle)
    return bundle


def validate_calibration_bundle(bundle):
    required = {
        "schema_version",
        "bundle_type",
        "compatible_model_version",
        "source_domain",
        "units",
        "coordinates",
        "grating",
        "camera",
        "membrane",
        "inverse",
        "calibration_confidence",
    }
    missing = required - set(bundle)
    if missing:
        raise ValueError(f"calibration bundle missing keys: {sorted(missing)}")
    if int(bundle["schema_version"]) != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported calibration schema version")
    if int(bundle["compatible_model_version"]) != 8:
        raise ValueError("calibration bundle is not compatible with model v8")
    if bundle["units"].get("length") != "mm":
        raise ValueError("calibration length unit must be mm")
    if bundle["coordinates"].get("array_row_zero") != "y=-sensor_radius_mm":
        raise ValueError("calibration coordinate convention does not match simulator")
    for section in ("grating", "camera", "membrane", "inverse"):
        confidence = float(bundle["calibration_confidence"].get(section, -1.0))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"invalid calibration confidence for {section}")
    if float(bundle["grating"]["pitch_mm"]) <= 0.0:
        raise ValueError("grating pitch must be positive")
    if float(bundle["camera"]["psf_sigma_px"]) < 0.0:
        raise ValueError("camera PSF must be non-negative")
    if float(bundle["membrane"]["tension_n_per_mm"]) <= 0.0:
        raise ValueError("membrane tension must be positive")
    mtf = bundle["membrane"].get("mechanical_mtf", {})
    mtf_frequency = np.asarray(
        mtf.get("frequencies_cycles_per_mm", ()), dtype=np.float64
    )
    mtf_gain = np.asarray(mtf.get("amplitude_gains", ()), dtype=np.float64)
    if (
        mtf_frequency.ndim != 1
        or len(mtf_frequency) < 2
        or mtf_gain.shape != mtf_frequency.shape
        or mtf_frequency[0] != 0.0
        or np.any(np.diff(mtf_frequency) <= 0.0)
        or np.any(mtf_gain <= 0.0)
    ):
        raise ValueError("mechanical MTF table is invalid")
    threshold = float(bundle["inverse"]["abstention_threshold"])
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("abstention threshold must be between zero and one")
    confidence = bundle["inverse"].get("confidence_calibration", {})
    for section, output_names in (
        ("appearance_helpfulness", ("calibrated_probability",)),
        (
            "reconstruction_error",
            ("calibrated_confidence", "expected_error_mm"),
        ),
    ):
        table = confidence.get(section, {})
        raw_axis = np.asarray(table.get("raw_confidence", ()), dtype=np.float64)
        if (
            raw_axis.ndim != 1
            or len(raw_axis) < 2
            or not np.isfinite(raw_axis).all()
            or np.any(np.diff(raw_axis) <= 0.0)
        ):
            raise ValueError(f"invalid confidence axis for {section}")
        for output_name in output_names:
            output = np.asarray(table.get(output_name, ()), dtype=np.float64)
            if output.shape != raw_axis.shape or not np.isfinite(output).all():
                raise ValueError(f"invalid {output_name} table for {section}")
            if "confidence" in output_name or "probability" in output_name:
                if np.any((output < 0.0) | (output > 1.0)):
                    raise ValueError(f"{output_name} must stay between zero and one")
            elif np.any(output < 0.0):
                raise ValueError(f"{output_name} must be non-negative")
    return bundle


def simulation_kwargs_from_bundle(bundle):
    validate_calibration_bundle(bundle)
    camera = bundle["camera"]
    membrane = bundle["membrane"]
    grating = bundle["grating"]
    inverse = bundle["inverse"]
    return {
        "membrane_tension_n_per_mm": float(membrane["tension_n_per_mm"]),
        "inflation_pressure_kpa": float(
            membrane["reference_inflation_pressure_kpa"]
        ),
        "membrane_bending_stiffness_n_mm": float(
            membrane["bending_stiffness_n_mm"]
        ),
        "cavity_depth_mm": float(membrane["cavity_depth_mm"]),
        "sealed_air_coupling": bool(membrane["sealed_air_coupling"]),
        "camera_psf_sigma": float(camera["psf_sigma_px"]),
        "camera_supersample": int(camera["raw_supersample"]),
        "noise_std": float(camera["read_noise_std"]),
        "grating_open_fraction": float(grating["open_fraction"]),
        "grating_line_transmittance": float(grating["line_transmittance"]),
        "slope_to_shift_mm": float(inverse["slope_to_shift_mm"]),
        "raw_flow_to_displacement_scale": float(
            inverse["carrier_to_displacement_scale"]
        ),
        "raw_flow_highpass_mm": float(inverse["carrier_highpass_mm"]),
        "mechanical_mtf_calibration": membrane["mechanical_mtf"],
        "confidence_calibration": inverse["confidence_calibration"],
        "abstention_threshold": float(inverse["abstention_threshold"]),
    }


def save_calibration_bundle(path, bundle):
    validate_calibration_bundle(bundle)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")


def load_calibration_bundle(path):
    bundle = json.loads(Path(path).read_text())
    return validate_calibration_bundle(bundle)
