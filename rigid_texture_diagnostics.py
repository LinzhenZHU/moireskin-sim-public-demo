#!/usr/bin/env python3
"""Local-only diagnostics for the rigid texture reconstruction POC."""

import cv2
import numpy as np

from moire_sim_platform import _sensor_grid
from rigid_object_poc import (
    _integrate_gradient,
    _masked_gaussian,
    _texture_metrics,
    simulate_rigid_object_poc,
)
from rigid_texture_calibration import (
    evaluate_confidence_calibration,
    fit_confidence_calibration,
)


DEFAULT_FREQUENCIES_CYCLES_PER_MM = (0.25, 0.50, 0.75, 1.00, 1.25)


def _interior_support(mask, radius_pixels=5):
    kernel_size = 2 * int(radius_pixels) + 1
    return cv2.erode(
        mask.astype(np.uint8), np.ones((kernel_size, kernel_size), np.uint8)
    ).astype(bool)


def _surface_from_exact_shift(config, shift_mm, slope_to_shift_mm):
    """Integrate exact simulated displacement for an oracle-only upper bound."""
    xx, yy, sensor_mask = _sensor_grid(config)
    spacing_mm = 2.0 * float(config["sensor_radius_mm"]) / (xx.shape[0] - 1)
    height = _integrate_gradient(
        shift_mm[0] / slope_to_shift_mm,
        shift_mm[1] / slope_to_shift_mm,
        spacing_mm,
    )
    boundary = sensor_mask & (
        np.hypot(xx, yy) > 0.86 * float(config["sensor_radius_mm"])
    )
    if height[sensor_mask].mean() < height[boundary].mean():
        height *= -1.0
    height -= float(np.median(height[boundary]))
    return (np.clip(height, 0.0, None) * sensor_mask).astype(np.float32)


def build_oracle_diagnostics(state, config, slope_to_shift_mm=0.12):
    """Measure each stage without allowing GT into the production inverse."""
    oracle_height = _surface_from_exact_shift(
        config,
        state["ground_truth_apparent_shift_mm"],
        float(slope_to_shift_mm),
    )
    spacing_mm = 2.0 * float(config["sensor_radius_mm"]) / (
        oracle_height.shape[0] - 1
    )
    support = _interior_support(state["target_mask"] & state["estimated_mask"])
    if not np.any(support):
        support = state["target_mask"] & state["estimated_mask"]
    target = state["ground_truth_height_mm"]
    membrane = state["membrane_height_mm"]
    stages = {
        "object": target,
        "membrane": membrane,
        "oracle_integrated": oracle_height,
        "coarse_inverse": state["coarse_reconstructed_height_mm"],
        "carrier_fused": state["reconstructed_height_mm"],
    }
    target_rms = _texture_metrics(
        target, target, support, spacing_mm
    )["texture_target_rms_mm"]
    stage_metrics = {}
    for name, height in stages.items():
        metrics = _texture_metrics(target, height, support, spacing_mm)
        stage_metrics[name] = {
            "texture_correlation_to_object": metrics["texture_correlation"],
            "amplitude_gain_from_object": metrics[
                "texture_reconstructed_rms_mm"
            ]
            / max(target_rms, 1e-8),
            "texture_nrmse_to_object": metrics["texture_nrmse"],
        }
    membrane_oracle = _texture_metrics(
        membrane, oracle_height, support, spacing_mm
    )
    return {
        "oracle_height_mm": oracle_height,
        "support": support,
        "stages": stage_metrics,
        "oracle_texture_correlation_to_membrane": membrane_oracle[
            "texture_correlation"
        ],
        "oracle_texture_nrmse_to_membrane": membrane_oracle["texture_nrmse"],
        "note": "Diagnostic only: exact simulated shift is never used by the inverse.",
    }


def _sinusoidal_coupon(config, frequency, angle_deg=23.0, amplitude_mm=0.10):
    xx, yy, sensor_mask = _sensor_grid(config)
    angle = np.deg2rad(float(angle_deg))
    coordinate = np.cos(angle) * xx + np.sin(angle) * yy
    object_mask = sensor_mask & (
        np.hypot(xx, yy) < 0.47 * 2.0 * float(config["sensor_radius_mm"])
    )
    height = np.zeros_like(xx, dtype=np.float32)
    height[object_mask] = 0.58 + float(amplitude_mm) * np.sin(
        2.0 * np.pi * float(frequency) * coordinate[object_mask]
    )
    return height, object_mask, coordinate


def run_frequency_response_benchmark(
    config,
    frequencies=DEFAULT_FREQUENCIES_CYCLES_PER_MM,
    slope_to_shift_mm=0.12,
    noise_std=0.0,
    simulation_parameters=None,
):
    """Return cumulative texture transfer versus spatial frequency."""
    rows = []
    for index, frequency in enumerate(frequencies):
        height, _, _ = _sinusoidal_coupon(config, frequency)
        albedo = np.where(height > 0.0, 0.66, 0.20).astype(np.float32)
        parameters = dict(simulation_parameters or {})
        parameters.pop("object_type", None)
        parameters.update(
            {
                "height_map_mm": height,
                "albedo_map": albedo,
                "slope_to_shift_mm": float(slope_to_shift_mm),
                "noise_std": float(noise_std),
                "seed": 101 + index,
            }
        )
        state = simulate_rigid_object_poc(
            config,
            **parameters,
        )
        oracle = build_oracle_diagnostics(
            state, config, slope_to_shift_mm=float(slope_to_shift_mm)
        )
        row = {"frequency_cycles_per_mm": float(frequency)}
        for stage, metrics in oracle["stages"].items():
            row[f"{stage}_amplitude_gain"] = metrics[
                "amplitude_gain_from_object"
            ]
            row[f"{stage}_correlation"] = metrics[
                "texture_correlation_to_object"
            ]
        row["oracle_correlation_to_membrane"] = oracle[
            "oracle_texture_correlation_to_membrane"
        ]
        row["carrier_confidence"] = state["high_frequency_diagnostics"][
            "carrier_confidence_mean"
        ]
        rows.append(row)
    return {
        "frequencies_cycles_per_mm": [
            row["frequency_cycles_per_mm"] for row in rows
        ],
        "rows": rows,
        "note": "All gains are cumulative relative to object texture amplitude.",
    }


def run_appearance_control_benchmark(
    config,
    frequency_cycles_per_mm=0.85,
    slope_to_shift_mm=0.12,
    noise_std=0.0,
    simulation_parameters=None,
):
    """Separate printed appearance, geometric relief, and coupled evidence."""
    relief_height, object_mask, coordinate = _sinusoidal_coupon(
        config, frequency_cycles_per_mm
    )
    flat_height = np.where(object_mask, 0.58, 0.0).astype(np.float32)
    visual_texture = 0.66 + 0.18 * np.sin(
        2.0 * np.pi * float(frequency_cycles_per_mm) * coordinate
    )
    textured_albedo = np.where(object_mask, visual_texture, 0.20).astype(
        np.float32
    )
    neutral_albedo = np.where(object_mask, 0.66, 0.20).astype(np.float32)
    cases = {
        "flat_print": (flat_height, textured_albedo),
        "neutral_relief": (relief_height, neutral_albedo),
        "coupled_relief": (relief_height, textured_albedo),
    }
    summaries = {}
    for index, (name, (height, albedo)) in enumerate(cases.items()):
        parameters = dict(simulation_parameters or {})
        parameters.pop("object_type", None)
        parameters.update(
            {
                "height_map_mm": height,
                "albedo_map": albedo,
                "slope_to_shift_mm": float(slope_to_shift_mm),
                "noise_std": float(noise_std),
                "seed": 211 + index,
            }
        )
        state = simulate_rigid_object_poc(
            config,
            **parameters,
        )
        support = _interior_support(
            state["target_mask"] & state["estimated_mask"]
        )
        detail = state["high_frequency_detail_mm"]
        false_relief_rms = (
            float(np.sqrt(np.mean(detail[support] ** 2)))
            if np.any(support)
            else float("nan")
        )
        summaries[name] = {
            "texture_correlation": state["metrics"]["texture_correlation"],
            "texture_amplitude_gain": state["metrics"][
                "texture_amplitude_gain"
            ],
            "high_frequency_detail_rms_mm": false_relief_rms,
            "appearance_geometry_confidence": state[
                "high_frequency_diagnostics"
            ]["appearance_geometry_confidence_mean"],
            "carrier_confidence": state["high_frequency_diagnostics"][
                "carrier_confidence_mean"
            ],
        }
    relief_reference_rms = summaries["neutral_relief"][
        "high_frequency_detail_rms_mm"
    ]
    summaries["flat_print"]["false_relief_ratio_vs_neutral"] = (
        summaries["flat_print"]["high_frequency_detail_rms_mm"]
        / max(relief_reference_rms, 1e-8)
    )
    return {
        "frequency_cycles_per_mm": float(frequency_cycles_per_mm),
        "cases": summaries,
        "note": "Appearance can refine geometry only where carrier geometry agrees.",
    }


def run_appearance_counterfactual_benchmark(
    config,
    frequencies=(0.50, 1.00),
    geometry_amplitudes_mm=(0.0, 0.10),
    contrasts=(0.0, 0.18),
    orientation_offsets_deg=(0.0, 60.0),
    phase_offsets_rad=(0.0, np.pi),
    simulation_parameters=None,
):
    """Calibrate fusion confidence on matched and conflicting appearance."""
    xx, yy, sensor_mask = _sensor_grid(config)
    base_angle_deg = 23.0
    base_angle = np.deg2rad(base_angle_deg)
    geometry_coordinate = np.cos(base_angle) * xx + np.sin(base_angle) * yy
    object_mask = sensor_mask & (
        np.hypot(xx, yy) < 0.47 * 2.0 * float(config["sensor_radius_mm"])
    )
    records = []
    arrays = []
    index = 0
    for frequency in frequencies:
        for geometry_amplitude in geometry_amplitudes_mm:
            for contrast in contrasts:
                for orientation_offset in orientation_offsets_deg:
                    for phase_offset in phase_offsets_rad:
                        height = np.zeros_like(xx, dtype=np.float32)
                        height[object_mask] = 0.58 + float(
                            geometry_amplitude
                        ) * np.sin(
                            2.0
                            * np.pi
                            * float(frequency)
                            * geometry_coordinate[object_mask]
                        )
                        appearance_angle = np.deg2rad(
                            base_angle_deg + float(orientation_offset)
                        )
                        appearance_coordinate = (
                            np.cos(appearance_angle) * xx
                            + np.sin(appearance_angle) * yy
                        )
                        albedo = np.full_like(xx, 0.20, dtype=np.float32)
                        albedo[object_mask] = 0.66 + float(contrast) * np.sin(
                            2.0
                            * np.pi
                            * float(frequency)
                            * appearance_coordinate[object_mask]
                            + float(phase_offset)
                        )
                        parameters = dict(simulation_parameters or {})
                        parameters.pop("object_type", None)
                        parameters.update(
                            {
                                "height_map_mm": height,
                                "albedo_map": albedo,
                                "noise_std": 0.0,
                                "seed": 401 + index,
                            }
                        )
                        state = simulate_rigid_object_poc(config, **parameters)
                        support = _interior_support(
                            state["target_mask"] & state["estimated_mask"]
                        )
                        if not np.any(support):
                            support = state["target_mask"] & state["estimated_mask"]
                        if not np.any(support):
                            raise ValueError("counterfactual case has no shared support")
                        sampled = support.copy()
                        sampled[1::2, :] = False
                        sampled[:, 1::2] = False
                        spacing_mm = 2.0 * float(config["sensor_radius_mm"]) / (
                            state["ground_truth_height_mm"].shape[0] - 1
                        )
                        sigma_pixels = max(0.8, 1.0 / spacing_mm)

                        def texture_detail(field):
                            return field - _masked_gaussian(
                                field, support, sigma_pixels
                            )

                        target_detail = texture_detail(
                            state["ground_truth_height_mm"]
                        )
                        fused_error = np.abs(
                            texture_detail(state["reconstructed_height_mm"])
                            - target_detail
                        )
                        carrier_error = np.abs(
                            texture_detail(state["carrier_only_height_mm"])
                            - target_detail
                        )
                        helpful = fused_error + 1e-5 < carrier_error
                        raw_appearance = state[
                            "raw_appearance_geometry_confidence"
                        ]
                        raw_reconstruction = state["reconstruction_confidence"]
                        arrays.append(
                            {
                                "fit": index % 2 == 0,
                                "appearance": raw_appearance[sampled],
                                "helpful": helpful[sampled],
                                "reconstruction": raw_reconstruction[sampled],
                                "error": fused_error[sampled],
                            }
                        )
                        records.append(
                            {
                                "frequency_cycles_per_mm": float(frequency),
                                "geometry_amplitude_mm": float(geometry_amplitude),
                                "appearance_contrast": float(contrast),
                                "orientation_offset_deg": float(orientation_offset),
                                "phase_offset_rad": float(phase_offset),
                                "is_conflict": bool(
                                    (geometry_amplitude == 0.0 and contrast > 0.0)
                                    or orientation_offset != 0.0
                                    or phase_offset != 0.0
                                ),
                                "fit_partition": (
                                    "fit" if index % 2 == 0 else "validation"
                                ),
                                "raw_appearance_confidence_mean": float(
                                    np.mean(raw_appearance[support])
                                ),
                                "fusion_helpful_fraction": float(
                                    np.mean(helpful[support])
                                ),
                                "carrier_mae_mm": float(
                                    np.mean(carrier_error[support])
                                ),
                                "fused_mae_mm": float(
                                    np.mean(fused_error[support])
                                ),
                            }
                        )
                        index += 1

    def concatenate(name, fit):
        return np.concatenate(
            [value[name] for value in arrays if value["fit"] == fit]
        )

    split_calibration = fit_confidence_calibration(
        concatenate("appearance", True),
        concatenate("helpful", True),
        concatenate("reconstruction", True),
        concatenate("error", True),
    )
    validation = evaluate_confidence_calibration(
        split_calibration,
        concatenate("appearance", False),
        concatenate("helpful", False),
        concatenate("reconstruction", False),
        concatenate("error", False),
    )
    final_calibration = fit_confidence_calibration(
        np.concatenate([value["appearance"] for value in arrays]),
        np.concatenate([value["helpful"] for value in arrays]),
        np.concatenate([value["reconstruction"] for value in arrays]),
        np.concatenate([value["error"] for value in arrays]),
    )
    return {
        "factor_grid": {
            "frequencies_cycles_per_mm": list(map(float, frequencies)),
            "geometry_amplitudes_mm": list(map(float, geometry_amplitudes_mm)),
            "appearance_contrasts": list(map(float, contrasts)),
            "orientation_offsets_deg": list(map(float, orientation_offsets_deg)),
            "phase_offsets_rad": list(map(float, phase_offsets_rad)),
        },
        "case_count": len(records),
        "cases": records,
        "validation": validation,
        "calibration": final_calibration,
        "note": (
            "The fit/validation split is by complete counterfactual case; the final "
            "lookup is refit on the full synthetic calibration grid."
        ),
    }
