#!/usr/bin/env python3
import math

import cv2
import numpy as np
from scipy.spatial import cKDTree

from moire_sim_platform import (
    _moire_component,
    _pattern_directions,
    _sensor_grid,
)


DEVELOPMENT_TEXTURE_OBJECT_TYPES = (
    "thread_array",
    "bump_array",
    "coin_relief",
    "knurled_ring",
)
HELD_OUT_TEXTURE_OBJECT_TYPES = (
    "phillips_head",
    "gear_face",
    "herringbone_plate",
    "microgroove_plate",
)
EXPLORATORY_TEXTURE_OBJECT_TYPES = (
    "hex_socket_bolt",
    "spiral_groove_disk",
    "woven_grid_plate",
    "serrated_key",
)
TEXTURE_OBJECT_TYPES = (
    DEVELOPMENT_TEXTURE_OBJECT_TYPES
    + HELD_OUT_TEXTURE_OBJECT_TYPES
    + EXPLORATORY_TEXTURE_OBJECT_TYPES
)
BASE_OBJECT_TYPES = ("screwdriver", "satin", "coin")
OBJECT_TYPES = (*BASE_OBJECT_TYPES, *TEXTURE_OBJECT_TYPES)
SHARED_AMBIENT_INTENSITY = 0.06
MIN_RAW_PIXELS_PER_GRATING_PITCH = 4.0
DEFAULT_MECHANICAL_MTF_CALIBRATION = {
    "frequencies_cycles_per_mm": (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
    "amplitude_gains": (1.0, 0.96, 0.38, 0.18, 0.10, 0.066, 0.045),
    "wiener_regularization": 0.04,
    "max_inverse_gain": 2.5,
}


def _object_height_field(
    object_type,
    xx,
    yy,
    rotation_deg,
    texture_frequency,
    visual_texture_frequency,
    offset_x_mm,
    offset_y_mm,
):
    if object_type not in OBJECT_TYPES:
        raise ValueError(f"object_type must be one of {OBJECT_TYPES}")
    angle = math.radians(rotation_deg)
    shifted_x = xx - offset_x_mm
    shifted_y = yy - offset_y_mm
    x = math.cos(angle) * shifted_x + math.sin(angle) * shifted_y
    y = -math.sin(angle) * shifted_x + math.cos(angle) * shifted_y
    height = np.zeros_like(xx, dtype=np.float32)
    albedo = np.full_like(xx, 0.20, dtype=np.float32)

    if object_type == "screwdriver":
        handle = ((x + 3.0) / 6.5) ** 2 + (y / 3.3) ** 2 <= 1.0
        shaft = (x >= 1.8) & (x <= 10.0) & (np.abs(y) <= 0.75)
        tip = (x > 9.0) & (x <= 11.0) & (np.abs(y) <= (11.0 - x) * 0.75)
        object_mask = handle | shaft | tip
        handle_profile = np.sqrt(
            np.clip(1.0 - (y[handle] / 3.3) ** 2, 0.0, 1.0)
        )
        height[handle] = (
            0.75
            + 0.15 * handle_profile
            + 0.05 * np.sin(2.0 * np.pi * texture_frequency * y[handle])
        )
        height[shaft] = 0.42 + 0.04 * np.cos(
            2.0 * np.pi * texture_frequency * x[shaft]
        )
        height[tip] = 0.25
        albedo[object_mask] = 0.66 + 0.14 * np.sin(
            2.0 * np.pi * visual_texture_frequency * y[object_mask]
        )
    elif object_type == "satin":
        object_mask = (np.abs(x) < 9.0) & (np.abs(y) < 6.0)
        height[object_mask] = (
            0.52
            + 0.16
            * np.sin(
                2.0 * np.pi * 0.12 * x[object_mask]
                + 0.7 * np.sin(0.4 * y[object_mask])
            )
            + 0.05
            * np.sin(2.0 * np.pi * texture_frequency * x[object_mask])
            * np.sin(2.0 * np.pi * texture_frequency * y[object_mask])
        )
        albedo[object_mask] = (
            0.66
            + 0.12
            * np.sin(2.0 * np.pi * visual_texture_frequency * x[object_mask])
            + 0.06
            * np.cos(2.0 * np.pi * visual_texture_frequency * y[object_mask])
        )
    elif object_type == "coin":
        radial = np.hypot(x, y)
        object_mask = radial < 6.5
        height[object_mask] = (
            0.55
            + 0.18
            * np.sqrt(np.clip(1.0 - (radial[object_mask] / 6.5) ** 2, 0.0, 1.0))
            + 0.06
            * np.cos(2.0 * np.pi * texture_frequency * radial[object_mask])
        )
        albedo[object_mask] = 0.65 + 0.12 * np.cos(
            2.0 * np.pi * visual_texture_frequency * radial[object_mask]
        )
    elif object_type == "thread_array":
        object_mask = np.zeros_like(xx, dtype=bool)
        centers = (-4.8, 0.0, 4.8)
        half_widths = (0.85, 1.10, 1.35)
        frequency_scales = (0.55, 0.75, 0.95)
        for center, half_width, frequency_scale in zip(
            centers, half_widths, frequency_scales
        ):
            local_x = x - center
            body = (np.abs(y) <= 4.0) & (np.abs(local_x) <= half_width)
            caps = (
                local_x * local_x + (np.abs(y) - 4.0) ** 2
                <= half_width * half_width
            ) & (np.abs(y) > 4.0)
            stud = body | caps
            crown = np.sqrt(
                np.clip(1.0 - (local_x[stud] / half_width) ** 2, 0.0, 1.0)
            )
            thread = (
                0.5
                + 0.5
                * np.cos(
                    2.0
                    * np.pi
                    * texture_frequency
                    * frequency_scale
                    * y[stud]
                )
            ) ** 2
            height[stud] = 0.46 + 0.16 * crown + 0.12 * thread
            albedo[stud] = 0.58 + 0.18 * thread + 0.05 * crown
            object_mask |= stud
    elif object_type == "bump_array":
        object_mask = np.zeros_like(xx, dtype=bool)
        for center_x in (-4.5, 0.0, 4.5):
            for center_y in (-4.5, 0.0, 4.5):
                radius_squared = (
                    ((x - center_x) / 1.45) ** 2
                    + ((y - center_y) / 1.45) ** 2
                )
                bump = radius_squared <= 1.0
                dome = np.sqrt(np.clip(1.0 - radius_squared[bump], 0.0, 1.0))
                height[bump] = 0.42 + 0.34 * dome
                albedo[bump] = 0.62 + 0.16 * dome
                object_mask |= bump
    elif object_type == "coin_relief":
        radial = np.hypot(x, y)
        theta = np.arctan2(y, x)
        object_mask = radial < 6.8
        rim = np.exp(-((radial - 5.95) / 0.38) ** 2)
        inner_ring = np.exp(-((radial - 4.75) / 0.28) ** 2)
        star_radius = 2.15 + 0.58 * np.cos(5.0 * theta)
        star = np.clip((star_radius - radial) / 0.28, 0.0, 1.0)
        beads = (
            np.exp(-((radial - 4.1) / 0.42) ** 2)
            * (0.5 + 0.5 * np.cos(12.0 * theta)) ** 6
        )
        relief = 0.14 * rim + 0.05 * inner_ring + 0.12 * star + 0.08 * beads
        height[object_mask] = (
            0.46
            + 0.04
            * np.sqrt(
                np.clip(1.0 - (radial[object_mask] / 6.8) ** 2, 0.0, 1.0)
            )
            + relief[object_mask]
        )
        albedo[object_mask] = 0.57 + 0.22 * np.clip(
            relief[object_mask] / 0.18, 0.0, 1.0
        )
    elif object_type == "knurled_ring":
        radial = np.hypot(x, y)
        object_mask = (radial < 7.2) & (radial > 2.6)
        knurl_frequency = 0.55 * texture_frequency
        diagonal_a = (
            0.5
            + 0.5
            * np.cos(
                2.0
                * np.pi
                * knurl_frequency
                * (x + y)
                / math.sqrt(2.0)
            )
        ) ** 6
        diagonal_b = (
            0.5
            + 0.5
            * np.cos(
                2.0
                * np.pi
                * knurl_frequency
                * (x - y)
                / math.sqrt(2.0)
            )
        ) ** 6
        knurl = np.maximum(diagonal_a, diagonal_b)
        edge = np.exp(-((radial - 6.75) / 0.30) ** 2) + np.exp(
            -((radial - 3.05) / 0.30) ** 2
        )
        height[object_mask] = (
            0.44 + 0.17 * knurl[object_mask] + 0.05 * edge[object_mask]
        )
        albedo[object_mask] = 0.56 + 0.20 * knurl[object_mask]
    elif object_type == "phillips_head":
        radial = np.hypot(x, y)
        object_mask = radial < 6.6
        dome = np.sqrt(np.clip(1.0 - (radial / 6.6) ** 2, 0.0, 1.0))
        rim = np.exp(-((radial - 5.9) / 0.32) ** 2)
        horizontal_slot = (np.abs(y) < 0.42) & (np.abs(x) < 3.1)
        vertical_slot = (np.abs(x) < 0.42) & (np.abs(y) < 3.1)
        slot = horizontal_slot | vertical_slot
        fine_rings = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * radial)
        ) ** 8
        height[object_mask] = (
            0.48
            + 0.08 * dome[object_mask]
            + 0.10 * rim[object_mask]
            + 0.04 * fine_rings[object_mask]
            - 0.14 * slot[object_mask]
        )
        albedo[object_mask] = (
            0.61
            + 0.10 * rim[object_mask]
            - 0.15 * slot[object_mask]
        )
    elif object_type == "gear_face":
        radial = np.hypot(x, y)
        theta = np.arctan2(y, x)
        teeth = (0.5 + 0.5 * np.cos(12.0 * theta)) ** 5
        outer_radius = 6.25 + 0.65 * teeth
        object_mask = radial < outer_radius
        hub = np.exp(-(radial / 1.45) ** 6)
        rim = np.exp(-((radial - 5.45) / 0.34) ** 2)
        spokes = (
            (0.5 + 0.5 * np.cos(8.0 * theta)) ** 7
            * np.exp(-((radial - 3.45) / 1.25) ** 4)
        )
        circular_ridges = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * radial)
        ) ** 8
        height[object_mask] = (
            0.44
            + 0.13 * hub[object_mask]
            + 0.10 * rim[object_mask]
            + 0.10 * spokes[object_mask]
            + 0.04 * circular_ridges[object_mask]
        )
        albedo[object_mask] = (
            0.56
            + 0.13 * spokes[object_mask]
            + 0.11 * rim[object_mask]
        )
    elif object_type == "herringbone_plate":
        object_mask = (np.abs(x) / 8.0) ** 6 + (np.abs(y) / 5.5) ** 6 <= 1.0
        chevron_coordinate = y - 0.72 * np.abs(x)
        ribs = (
            0.5
            + 0.5
            * np.cos(
                2.0
                * np.pi
                * 0.62
                * texture_frequency
                * chevron_coordinate
            )
        ) ** 8
        center_seam = np.exp(-(x / 0.28) ** 2)
        low_form = 0.5 + 0.5 * np.cos(2.0 * np.pi * 0.10 * y)
        height[object_mask] = (
            0.43
            + 0.15 * ribs[object_mask]
            + 0.05 * center_seam[object_mask]
            + 0.04 * low_form[object_mask]
        )
        albedo[object_mask] = (
            0.55
            + 0.20 * ribs[object_mask]
            + 0.05 * center_seam[object_mask]
        )
    elif object_type == "microgroove_plate":
        object_mask = (np.abs(x) / 8.2) ** 8 + (np.abs(y) / 5.6) ** 8 <= 1.0
        scale = np.where(y > 1.8, 0.65, np.where(y < -1.8, 1.45, 1.0))
        direction = np.where(y < -1.8, 0.30 * y, 0.0)
        grooves = (
            0.5
            + 0.5
            * np.cos(
                2.0
                * np.pi
                * texture_frequency
                * scale
                * (x + direction)
            )
        ) ** 8
        separators = np.exp(-((np.abs(y) - 1.8) / 0.22) ** 2)
        low_form = 0.5 + 0.5 * np.cos(2.0 * np.pi * 0.12 * y)
        height[object_mask] = (
            0.44
            + 0.13 * grooves[object_mask]
            + 0.05 * separators[object_mask]
            + 0.03 * low_form[object_mask]
        )
        albedo[object_mask] = (
            0.57
            + 0.18 * grooves[object_mask]
            + 0.05 * separators[object_mask]
        )
    elif object_type == "hex_socket_bolt":
        radial = np.hypot(x, y)
        hex_radius = np.maximum(
            np.abs(x), 0.5 * np.abs(x) + 0.5 * math.sqrt(3.0) * np.abs(y)
        )
        object_mask = hex_radius < 6.4
        socket = hex_radius < 1.75
        bevel = np.exp(-((hex_radius - 5.75) / 0.35) ** 2)
        machining = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * radial)
        ) ** 8
        height[object_mask] = (
            0.48
            + 0.09 * bevel[object_mask]
            + 0.04 * machining[object_mask]
            - 0.16 * socket[object_mask]
        )
        optical_rings = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * visual_texture_frequency * radial)
        ) ** 6
        albedo[object_mask] = (
            0.60
            + 0.10 * optical_rings[object_mask]
            - 0.18 * socket[object_mask]
        )
    elif object_type == "spiral_groove_disk":
        radial = np.hypot(x, y)
        theta = np.arctan2(y, x)
        object_mask = radial < 6.8
        spiral = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * texture_frequency * radial - theta)
        ) ** 8
        rim = np.exp(-((radial - 6.05) / 0.32) ** 2)
        hub = np.exp(-(radial / 1.25) ** 4)
        height[object_mask] = (
            0.45
            + 0.11 * spiral[object_mask]
            + 0.08 * rim[object_mask]
            + 0.05 * hub[object_mask]
        )
        optical_spiral = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * visual_texture_frequency * radial - theta)
        ) ** 6
        albedo[object_mask] = (
            0.56
            + 0.18 * optical_spiral[object_mask]
            + 0.06 * rim[object_mask]
        )
    elif object_type == "woven_grid_plate":
        object_mask = (np.abs(x) / 8.0) ** 8 + (np.abs(y) / 5.5) ** 8 <= 1.0
        warp = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * x)
        ) ** 7
        weft = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * y)
        ) ** 7
        over_under = np.sin(np.pi * texture_frequency * x) * np.sin(
            np.pi * texture_frequency * y
        )
        weave = np.where(over_under >= 0.0, warp, weft)
        low_form = 0.5 + 0.5 * np.cos(2.0 * np.pi * 0.10 * (x + y))
        height[object_mask] = (
            0.43
            + 0.15 * weave[object_mask]
            + 0.04 * low_form[object_mask]
        )
        optical_warp = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * visual_texture_frequency * x)
        ) ** 5
        optical_weft = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * visual_texture_frequency * y)
        ) ** 5
        albedo[object_mask] = (
            0.54
            + 0.11 * optical_warp[object_mask]
            + 0.11 * optical_weft[object_mask]
        )
    elif object_type == "serrated_key":
        bow_outer = (x + 5.2) ** 2 + y * y < 3.4 ** 2
        bow_hole = (x + 5.2) ** 2 + y * y < 1.35 ** 2
        tooth_profile = 1.35 + 0.70 * (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * 0.42 * texture_frequency * (x - 0.2))
        ) ** 5
        shaft = (
            (x >= -3.0)
            & (x <= 7.8)
            & (y >= -1.35)
            & (y <= tooth_profile)
        )
        tip = (x > 7.8) & (x <= 9.2) & (np.abs(y) <= 1.35)
        object_mask = (bow_outer & ~bow_hole) | shaft | tip
        longitudinal_groove = np.exp(-(y / 0.30) ** 2)
        cross_ridges = (
            0.5 + 0.5 * np.cos(2.0 * np.pi * texture_frequency * x)
        ) ** 8
        height[object_mask] = (
            0.46
            + 0.07 * longitudinal_groove[object_mask]
            + 0.06 * cross_ridges[object_mask]
        )
        optical_ridges = (
            0.5
            + 0.5
            * np.cos(2.0 * np.pi * visual_texture_frequency * x)
        ) ** 6
        albedo[object_mask] = (
            0.59
            + 0.13 * optical_ridges[object_mask]
            + 0.06 * longitudinal_groove[object_mask]
        )
    return height, np.clip(albedo, 0.0, 1.0), object_mask


def _carrier_peak_and_row(config, direction_deg):
    angle_a = math.radians(float(config["grating_angle_a_deg"]) + direction_deg)
    angle_b = math.radians(float(config["grating_angle_b_deg"]) + direction_deg)
    diameter = 2.0 * float(config["sensor_radius_mm"])
    pitch = float(config["grating_pitch_mm"])
    peak = (
        int(round((math.cos(angle_a) - math.cos(angle_b)) * diameter / pitch)),
        int(round((math.sin(angle_a) - math.sin(angle_b)) * diameter / pitch)),
    )
    orientation = 1.0
    if peak[1] < 0 or (peak[1] == 0 and peak[0] < 0):
        peak = (-peak[0], -peak[1])
        orientation = -1.0
    row = orientation * (-2.0 * np.pi / pitch) * np.asarray(
        (
            float(config["top_layer_motion_scale"]) * math.cos(angle_a)
            - float(config["bottom_layer_motion_scale"]) * math.cos(angle_b),
            float(config["top_layer_motion_scale"]) * math.sin(angle_a)
            - float(config["bottom_layer_motion_scale"]) * math.sin(angle_b),
        )
    )
    return peak, row


def _recover_displacement(config, pre_image, post_image, sensor_mask):
    phase_maps = []
    rows = []
    sigma = min(4.0, max(2.0, pre_image.shape[0] / 32.0))
    for direction in _pattern_directions(config["pattern"]):
        peak, row = _carrier_peak_and_row(config, direction)
        pre_component = _moire_component(
            pre_image, peak, sensor_mask, sigma=sigma
        )
        post_component = _moire_component(
            post_image, peak, sensor_mask, sigma=sigma
        )
        phase_maps.append(np.angle(post_component * np.conj(pre_component)))
        rows.append(row)
    if len(rows) != 2:
        raise ValueError("rigid reconstruction needs a cross pattern with two carriers")
    recovered = np.einsum(
        "ij,jhw->ihw", np.linalg.inv(np.asarray(rows)), np.asarray(phase_maps)
    )
    recovered[:, ~sensor_mask] = 0.0
    return recovered


def _integrate_gradient(gradient_x, gradient_y, spacing_mm):
    angular_frequency = 2.0 * np.pi * np.fft.fftfreq(
        gradient_x.shape[0], d=spacing_mm
    )
    ky, kx = np.meshgrid(angular_frequency, angular_frequency, indexing="ij")
    denominator = kx * kx + ky * ky
    denominator[0, 0] = 1.0
    height_spectrum = (
        -1j * kx * np.fft.fft2(gradient_x)
        - 1j * ky * np.fft.fft2(gradient_y)
    ) / denominator
    height_spectrum[0, 0] = 0.0
    return np.fft.ifft2(height_spectrum).real


def _laplacian(field, spacing_mm):
    return (
        np.roll(field, 1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, 1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4.0 * field
    ) / (spacing_mm * spacing_mm)


def _biharmonic(field, spacing_mm):
    return _laplacian(_laplacian(field, spacing_mm), spacing_mm)


def _sealed_cavity_pressure(
    reference_pressure_kpa,
    displacement_mm,
    interior_mask,
    spacing_mm,
    cavity_depth_mm,
):
    """Return ideal-gas pressure after the membrane displaces cavity volume."""
    if cavity_depth_mm <= 0.0:
        raise ValueError("cavity depth must be positive")
    reference_volume_mm3 = (
        np.count_nonzero(interior_mask)
        * spacing_mm
        * spacing_mm
        * cavity_depth_mm
    )
    displaced_volume_mm3 = float(
        np.sum(np.maximum(displacement_mm[interior_mask], 0.0))
        * spacing_mm
        * spacing_mm
    )
    volume_fraction = float(
        np.clip(
            displaced_volume_mm3 / max(reference_volume_mm3, 1e-8),
            0.0,
            0.35,
        )
    )
    ambient_kpa = 101.325
    effective_pressure_kpa = (
        (ambient_kpa + float(reference_pressure_kpa))
        / (1.0 - volume_fraction)
        - ambient_kpa
    )
    return effective_pressure_kpa, volume_fraction


def _solve_membrane_contact(
    obstacle_height_mm,
    sensor_mask,
    spacing_mm,
    membrane_tension_n_per_mm,
    inflation_pressure_kpa,
    membrane_bending_stiffness_n_mm=0.0,
    cavity_depth_mm=8.0,
    sealed_air_coupling=True,
):
    """Solve local displacement from an inflated reference toward an obstacle."""
    if membrane_tension_n_per_mm <= 0.0:
        raise ValueError("membrane tension must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
    if membrane_bending_stiffness_n_mm < 0.0:
        raise ValueError("membrane bending stiffness must be non-negative")
    if cavity_depth_mm <= 0.0:
        raise ValueError("cavity depth must be positive")
    interior = cv2.erode(
        sensor_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    boundary = sensor_mask & ~interior
    obstacle = np.maximum(np.asarray(obstacle_height_mm, dtype=np.float64), 0.0)
    obstacle[~interior] = 0.0
    displacement = obstacle.copy()
    yy, xx = np.indices(displacement.shape)
    checkerboard = (xx + yy) & 1
    relaxation = 1.82
    tolerance_mm = 1e-6
    update_mm = math.inf
    # w is measured inward from the inflated reference, so preload lowers the
    # free membrane toward the obstacle in this local coordinate system.

    for iteration in range(1, 1201):
        previous = displacement.copy()
        if sealed_air_coupling:
            effective_pressure_kpa, volume_fraction = _sealed_cavity_pressure(
                inflation_pressure_kpa,
                displacement,
                interior,
                spacing_mm,
                cavity_depth_mm,
            )
        else:
            effective_pressure_kpa = float(inflation_pressure_kpa)
            volume_fraction = 0.0
        pressure_n_per_mm2 = effective_pressure_kpa * 1e-3
        pressure_step_mm = (
            pressure_n_per_mm2
            * spacing_mm
            * spacing_mm
            / (4.0 * membrane_tension_n_per_mm)
        )
        for color in (0, 1):
            neighbors = (
                np.roll(displacement, 1, axis=0)
                + np.roll(displacement, -1, axis=0)
                + np.roll(displacement, 1, axis=1)
                + np.roll(displacement, -1, axis=1)
            )
            selected = interior & (checkerboard == color)
            candidate = (
                (1.0 - relaxation) * displacement
                + relaxation * (neighbors / 4.0 - pressure_step_mm)
            )
            displacement[selected] = np.maximum(
                obstacle[selected], candidate[selected]
            )
        displacement[~interior] = 0.0
        update_mm = float(np.max(np.abs(displacement - previous)))
        if update_mm < tolerance_mm:
            break
    else:
        raise RuntimeError("membrane contact solver did not converge")

    bending_iterations = 0
    bending_converged = True
    if membrane_bending_stiffness_n_mm > 0.0:
        operator_lipschitz = (
            8.0 * membrane_tension_n_per_mm / (spacing_mm * spacing_mm)
            + 64.0
            * membrane_bending_stiffness_n_mm
            / (spacing_mm**4)
        )
        step = 0.82 / operator_lipschitz
        for bending_iterations in range(1, 401):
            previous = displacement.copy()
            if sealed_air_coupling:
                effective_pressure_kpa, volume_fraction = (
                    _sealed_cavity_pressure(
                        inflation_pressure_kpa,
                        displacement,
                        interior,
                        spacing_mm,
                        cavity_depth_mm,
                    )
                )
            pressure_n_per_mm2 = effective_pressure_kpa * 1e-3
            energy_gradient = (
                -membrane_tension_n_per_mm
                * _laplacian(displacement, spacing_mm)
                + membrane_bending_stiffness_n_mm
                * _biharmonic(displacement, spacing_mm)
                + pressure_n_per_mm2
            )
            candidate = displacement - step * energy_gradient
            displacement[interior] = np.maximum(
                obstacle[interior], candidate[interior]
            )
            displacement[~interior] = 0.0
            update_mm = float(np.max(np.abs(displacement - previous)))
            if update_mm < tolerance_mm:
                break
        bending_converged = update_mm < 1e-5

    if sealed_air_coupling:
        effective_pressure_kpa, volume_fraction = _sealed_cavity_pressure(
            inflation_pressure_kpa,
            displacement,
            interior,
            spacing_mm,
            cavity_depth_mm,
        )
    else:
        effective_pressure_kpa = float(inflation_pressure_kpa)
        volume_fraction = 0.0
    pressure_n_per_mm2 = effective_pressure_kpa * 1e-3
    pressure = np.maximum(
        -membrane_tension_n_per_mm
        * _laplacian(displacement, spacing_mm)
        + membrane_bending_stiffness_n_mm
        * _biharmonic(displacement, spacing_mm)
        + pressure_n_per_mm2,
        0.0,
    )
    contact_mask = (
        (obstacle > 0.0)
        & (np.abs(displacement - obstacle) < 1e-4)
        & (pressure > 0.0)
    )
    pressure *= contact_mask
    return {
        "obstacle_height_mm": obstacle.astype(np.float32),
        "membrane_height_mm": displacement.astype(np.float32),
        "contact_pressure_n_per_mm2": pressure.astype(np.float32),
        "contact_mask": contact_mask,
        "boundary_mask": boundary,
        "iterations": iteration,
        "bending_iterations": bending_iterations,
        "bending_converged": bending_converged,
        "update_mm": update_mm,
        "normal_force_n": float(np.sum(pressure) * spacing_mm * spacing_mm),
        "inflation_pressure_kpa": float(inflation_pressure_kpa),
        "effective_pressure_kpa": float(effective_pressure_kpa),
        "sealed_air_volume_change_fraction": volume_fraction,
        "membrane_bending_stiffness_n_mm": float(
            membrane_bending_stiffness_n_mm
        ),
        "cavity_depth_mm": float(cavity_depth_mm),
        "contact_fraction": float(
            np.count_nonzero(contact_mask) / max(1, np.count_nonzero(obstacle))
        ),
        "max_penetration_mm": float(
            np.max(np.maximum(obstacle - displacement, 0.0))
        ),
        "boundary_displacement_mm": float(np.max(np.abs(displacement[boundary]))),
    }


def _gaussian_blur(image, sigma):
    return image if sigma <= 0.0 else cv2.GaussianBlur(image, (0, 0), sigma)


def _render_grating_transmission(
    config,
    displacement,
    open_fraction,
    line_transmittance,
    supersample=4,
):
    if not 0.0 < open_fraction < 1.0:
        raise ValueError("grating open fraction must be between zero and one")
    if not 0.0 <= line_transmittance <= 1.0:
        raise ValueError("line transmittance must be between zero and one")
    size = int(config["image_size"])
    high_size = supersample * size
    radius = float(config["sensor_radius_mm"])
    axis = np.linspace(-radius, radius, high_size, dtype=np.float32)
    high_x, high_y = np.meshgrid(axis, axis)
    high_mask = high_x * high_x + high_y * high_y <= radius * radius
    radial_scale = 1.0 + float(config["radial_distortion_k1"]) * (
        high_x * high_x + high_y * high_y
    ) / (radius * radius)
    top_x = high_x * radial_scale
    top_y = high_y * radial_scale
    bottom_x = top_x.copy()
    bottom_y = top_y.copy()

    if displacement is not None:
        ux, uy = (np.asarray(value, dtype=np.float32) for value in displacement)
        if ux.shape != (size, size) or uy.shape != (size, size):
            raise ValueError("displacement arrays must match the sensor grid")
        ux = cv2.resize(ux, (high_size, high_size), interpolation=cv2.INTER_CUBIC)
        uy = cv2.resize(uy, (high_size, high_size), interpolation=cv2.INTER_CUBIC)
        top_x -= float(config["top_layer_motion_scale"]) * ux
        top_y -= float(config["top_layer_motion_scale"]) * uy
        bottom_x -= float(config["bottom_layer_motion_scale"]) * ux
        bottom_y -= float(config["bottom_layer_motion_scale"]) * uy

    top = np.ones_like(high_x, dtype=np.float32)
    bottom = np.ones_like(high_x, dtype=np.float32)
    line_half_width = 0.5 * (1.0 - float(open_fraction))
    pitch = float(config["grating_pitch_mm"])
    phase_offset = float(config["moire_phase_offset_rad"])
    for direction in _pattern_directions(config["pattern"]):
        angle_a = math.radians(float(config["grating_angle_a_deg"]) + direction)
        angle_b = math.radians(float(config["grating_angle_b_deg"]) + direction)
        cycles_a = (
            math.cos(angle_a) * top_x + math.sin(angle_a) * top_y
        ) / pitch
        cycles_b = (
            math.cos(angle_b) * bottom_x + math.sin(angle_b) * bottom_y
        ) / pitch - phase_offset / (2.0 * np.pi)
        distance_a = np.abs(cycles_a - np.rint(cycles_a))
        distance_b = np.abs(cycles_b - np.rint(cycles_b))
        top *= np.where(
            distance_a <= line_half_width, line_transmittance, 1.0
        )
        bottom *= np.where(
            distance_b <= line_half_width, line_transmittance, 1.0
        )

    transmission = top * bottom * high_mask
    transmission = cv2.resize(
        transmission, (size, size), interpolation=cv2.INTER_AREA
    )
    _, _, low_mask = _sensor_grid(config)
    transmission[~low_mask] = 0.0
    return np.clip(transmission, 0.0, 1.0).astype(np.float32)


def _render_moire_envelope(config, xx, yy, displacement):
    radius = float(config["sensor_radius_mm"])
    radial_scale = 1.0 + float(config["radial_distortion_k1"]) * (
        xx * xx + yy * yy
    ) / (radius * radius)
    top_x = xx * radial_scale
    top_y = yy * radial_scale
    bottom_x = top_x.copy()
    bottom_y = top_y.copy()
    if displacement is not None:
        ux, uy = (np.asarray(value) for value in displacement)
        top_x -= float(config["top_layer_motion_scale"]) * ux
        top_y -= float(config["top_layer_motion_scale"]) * uy
        bottom_x -= float(config["bottom_layer_motion_scale"]) * ux
        bottom_y -= float(config["bottom_layer_motion_scale"]) * uy

    pitch = float(config["grating_pitch_mm"])
    phase_offset = float(config["moire_phase_offset_rad"])
    components = []
    for direction in _pattern_directions(config["pattern"]):
        angle_a = math.radians(float(config["grating_angle_a_deg"]) + direction)
        angle_b = math.radians(float(config["grating_angle_b_deg"]) + direction)
        phase_a = 2.0 * np.pi * (
            math.cos(angle_a) * top_x + math.sin(angle_a) * top_y
        ) / pitch
        phase_b = 2.0 * np.pi * (
            math.cos(angle_b) * bottom_x + math.sin(angle_b) * bottom_y
        ) / pitch
        components.append(np.cos(phase_a - phase_b + phase_offset))
    contrast = sum(config["contrast"]) / 2.0
    return np.clip(
        1.0 + contrast * np.mean(components, axis=0), 0.5, 1.5
    ).astype(np.float32)


def _render_shared_observation(
    config,
    xx,
    yy,
    optical_albedo,
    transmission,
    sensor_mask,
    optical,
    noise_std,
    rng,
):
    radius = float(config["sensor_radius_mm"])
    vignette = np.clip(
        1.0 - 0.16 * (xx * xx + yy * yy) / (radius * radius), 0.78, 1.0
    )
    ambient = SHARED_AMBIENT_INTENSITY
    image = ambient + (1.0 - ambient) * optical_albedo * transmission
    image *= vignette
    image = image * float(optical["gain"]) + float(optical["offset"])
    image += rng.normal(0.0, noise_std, image.shape)
    image[~sensor_mask] = float(config["outside_intensity"])
    return np.clip(np.rint(image * 255.0), 0, 255).astype(np.uint8)


def _recover_object_appearance(
    config,
    pre_image,
    post_image,
    apparent_shift,
    sensor_mask,
    grating_open_fraction,
    grating_line_transmittance,
):
    """Invert calibrated clear apertures without treating the grid as blur."""
    xx, yy, _ = _sensor_grid(config)
    radius = float(config["sensor_radius_mm"])
    vignette = np.clip(
        1.0 - 0.16 * (xx * xx + yy * yy) / (radius * radius), 0.78, 1.0
    )
    gain = sum(config["brightness_gain"]) / 2.0
    offset = sum(config["brightness_offset"]) / 2.0
    observations = []
    for image, displacement in (
        (pre_image, None),
        (post_image, apparent_shift),
    ):
        transfer = _render_grating_transmission(
            config,
            displacement,
            grating_open_fraction,
            grating_line_transmittance,
        )
        transfer *= _render_moire_envelope(config, xx, yy, displacement)
        corrected = (
            (image.astype(np.float32) / 255.0 - offset) / gain
        ) / vignette
        observations.append(
            (transfer, corrected - SHARED_AMBIENT_INTENSITY)
        )

    pre_transfer, pre_signal = observations[0]
    post_transfer, post_signal = observations[1]
    pre_valid = (pre_transfer > 0.35) & sensor_mask
    post_valid = (~pre_valid) & (post_transfer > 0.62) & sensor_mask
    sampled = pre_valid | post_valid
    appearance = np.zeros_like(pre_transfer, dtype=np.float32)
    scale = 1.0 - SHARED_AMBIENT_INTENSITY
    appearance[pre_valid] = pre_signal[pre_valid] / (
        scale * pre_transfer[pre_valid]
    )
    appearance[post_valid] = post_signal[post_valid] / (
        scale * post_transfer[post_valid]
    )

    weights = sampled.astype(np.float32)
    numerator = cv2.GaussianBlur(
        np.clip(appearance, 0.0, 1.0) * weights, (0, 0), 0.75
    )
    denominator = cv2.GaussianBlur(weights, (0, 0), 0.75)
    recovered = numerator / np.maximum(denominator, 1e-4)
    recovered[~sensor_mask] = 0.20
    return (
        np.clip(recovered, 0.0, 1.0).astype(np.float32),
        np.clip(denominator, 0.0, 1.0).astype(np.float32),
    )


def _optical_texture_metrics(
    albedo,
    optical_albedo,
    object_mask,
    transmission,
    spacing_mm,
):
    interior = cv2.erode(
        object_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
    ).astype(bool)
    clear = object_mask & (transmission >= 0.5)
    valid = interior & clear

    target_gradients = []
    observed_gradients = []
    for axis in (0, 1):
        target_gradient = np.diff(albedo, axis=axis)
        observed_gradient = np.diff(optical_albedo, axis=axis)
        if axis == 0:
            gradient_mask = valid[1:, :] & valid[:-1, :]
        else:
            gradient_mask = valid[:, 1:] & valid[:, :-1]
        target_gradients.append(target_gradient[gradient_mask])
        observed_gradients.append(observed_gradient[gradient_mask])
    target_gradient = np.concatenate(target_gradients)
    observed_gradient = np.concatenate(observed_gradients)
    correlation = 0.0
    retention = 0.0
    if len(target_gradient) > 1 and np.std(target_gradient) > 1e-8:
        correlation = float(np.corrcoef(target_gradient, observed_gradient)[0, 1])
        retention = float(
            np.sqrt(np.mean(observed_gradient * observed_gradient))
            / max(np.sqrt(np.mean(target_gradient * target_gradient)), 1e-8)
        )
    texture = _texture_metrics(
        albedo, optical_albedo, object_mask, spacing_mm
    )
    return {
        "clear_aperture_fraction": float(
            np.count_nonzero(clear) / max(1, np.count_nonzero(object_mask))
        ),
        "mean_grating_transmission": float(np.mean(transmission[object_mask])),
        "see_through_gradient_correlation": correlation,
        "see_through_gradient_retention": retention,
        "see_through_texture_correlation": texture["texture_correlation"],
        "see_through_texture_nrmse": texture["texture_nrmse"],
    }


def _estimate_object_mask(see_through, xx, yy, sensor_mask):
    smooth = cv2.GaussianBlur(see_through.astype(np.float32), (0, 0), 2.0)
    radius = np.hypot(xx, yy)
    sensor_radius = float(np.max(radius[sensor_mask]))
    background_ring = (radius > 0.83 * sensor_radius) & (
        radius < 0.93 * sensor_radius
    )
    background = float(np.median(smooth[background_ring]))
    bright = float(np.quantile(smooth[sensor_mask], 0.96))
    estimated = (smooth > background + 0.50 * (bright - background)) & sensor_mask
    kernel_size = max(3, int(round(see_through.shape[0] / 32.0)) | 1)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(
        estimated.astype(np.uint8), cv2.MORPH_CLOSE, kernel
    ).astype(bool)


def _complex_gaussian_blur(field, sigma_pixels):
    return cv2.GaussianBlur(
        field.real.astype(np.float32), (0, 0), sigma_pixels
    ) + 1j * cv2.GaussianBlur(
        field.imag.astype(np.float32), (0, 0), sigma_pixels
    )


def _local_carrier_component(image, carrier_x, carrier_y, sigma_pixels):
    yy, xx = np.indices(image.shape, dtype=np.float32)
    carrier = np.exp(-1j * (carrier_x * xx + carrier_y * yy))
    signal = image.astype(np.float32) / 255.0
    return _complex_gaussian_blur(signal * carrier, sigma_pixels)


def _solve_weighted_carrier_displacement(rows, phase_maps, weights):
    row_x = rows[:, 0, None, None]
    row_y = rows[:, 1, None, None]
    normal_xx = np.sum(weights * row_x * row_x, axis=0)
    normal_xy = np.sum(weights * row_x * row_y, axis=0)
    normal_yy = np.sum(weights * row_y * row_y, axis=0)
    right_x = np.sum(weights * row_x * phase_maps, axis=0)
    right_y = np.sum(weights * row_y * phase_maps, axis=0)
    determinant = normal_xx * normal_yy - normal_xy * normal_xy
    valid = determinant > 1e-8
    displacement_x = np.zeros_like(determinant, dtype=np.float32)
    displacement_y = np.zeros_like(determinant, dtype=np.float32)
    displacement_x[valid] = (
        right_x[valid] * normal_yy[valid]
        - right_y[valid] * normal_xy[valid]
    ) / determinant[valid]
    displacement_y[valid] = (
        right_y[valid] * normal_xx[valid]
        - right_x[valid] * normal_xy[valid]
    ) / determinant[valid]
    predicted = (
        row_x * displacement_x[None, ...]
        + row_y * displacement_y[None, ...]
    )
    residual = np.angle(np.exp(1j * (phase_maps - predicted)))
    residual_rms = np.sqrt(
        np.sum(weights * residual * residual, axis=0)
        / np.maximum(np.sum(weights, axis=0), 1e-8)
    )
    confidence = (
        np.mean(weights, axis=0)
        * np.exp(-0.5 * (residual_rms / 0.45) ** 2)
        * valid
    )
    return displacement_x, displacement_y, confidence, residual_rms


def _joint_wrapped_carrier_refinement(
    rows,
    phase_maps,
    weights,
    initial_x,
    initial_y,
    iterations=4,
):
    """Refine an LK initialization against every wrapped carrier phase."""
    displacement_x = np.asarray(initial_x, dtype=np.float32).copy()
    displacement_y = np.asarray(initial_y, dtype=np.float32).copy()
    row_x = rows[:, 0, None, None]
    row_y = rows[:, 1, None, None]
    weights = np.asarray(weights, dtype=np.float32)
    for _ in range(int(iterations)):
        predicted = (
            row_x * displacement_x[None, ...]
            + row_y * displacement_y[None, ...]
        )
        residual = np.angle(np.exp(1j * (phase_maps - predicted)))
        robust = weights * np.minimum(
            1.0, 0.45 / np.maximum(np.abs(residual), 1e-5)
        )
        delta_x, delta_y, _, _ = _solve_weighted_carrier_displacement(
            rows, residual, robust
        )
        displacement_x += delta_x
        displacement_y += delta_y

    predicted = (
        row_x * displacement_x[None, ...]
        + row_y * displacement_y[None, ...]
    )
    residual = np.angle(np.exp(1j * (phase_maps - predicted)))
    residual_rms = np.sqrt(
        np.sum(weights * residual * residual, axis=0)
        / np.maximum(np.sum(weights, axis=0), 1e-8)
    )
    normal_xx = np.sum(weights * row_x * row_x, axis=0)
    normal_xy = np.sum(weights * row_x * row_y, axis=0)
    normal_yy = np.sum(weights * row_y * row_y, axis=0)
    valid = normal_xx * normal_yy - normal_xy * normal_xy > 1e-8
    confidence = (
        np.mean(weights, axis=0)
        * np.exp(-0.5 * (residual_rms / 0.45) ** 2)
        * valid
    )
    return (
        displacement_x.astype(np.float32),
        displacement_y.astype(np.float32),
        np.clip(confidence, 0.0, 1.0).astype(np.float32),
        residual_rms.astype(np.float32),
    )


def _calibrate_confidence(raw_confidence, calibration, output_key):
    if not calibration:
        return np.asarray(raw_confidence, dtype=np.float32)
    raw_axis = np.asarray(calibration.get("raw_confidence", ()), dtype=np.float32)
    output_axis = np.asarray(calibration.get(output_key, ()), dtype=np.float32)
    if (
        raw_axis.ndim != 1
        or len(raw_axis) < 2
        or output_axis.shape != raw_axis.shape
        or not np.isfinite(raw_axis).all()
        or not np.isfinite(output_axis).all()
        or not np.all(np.diff(raw_axis) > 0.0)
    ):
        raise ValueError(
            "confidence calibration axes must be finite increasing vectors"
        )
    return np.interp(
        np.clip(raw_confidence, 0.0, 1.0),
        raw_axis,
        output_axis,
    ).astype(np.float32)


def _mechanical_mtf_deconvolution(
    detail,
    support,
    spacing_mm,
    calibration,
):
    calibration = calibration or DEFAULT_MECHANICAL_MTF_CALIBRATION
    frequencies = np.asarray(
        calibration["frequencies_cycles_per_mm"], dtype=np.float32
    )
    gains = np.asarray(calibration["amplitude_gains"], dtype=np.float32)
    regularization = float(calibration["wiener_regularization"])
    max_inverse_gain = float(calibration["max_inverse_gain"])
    if (
        frequencies.ndim != 1
        or len(frequencies) < 2
        or gains.shape != frequencies.shape
        or frequencies[0] != 0.0
        or np.any(np.diff(frequencies) <= 0.0)
        or np.any(gains <= 0.0)
        or regularization <= 0.0
        or max_inverse_gain < 1.0
    ):
        raise ValueError("invalid mechanical MTF calibration")

    edge_width = max(1.0, 0.35 / spacing_mm)
    taper = np.minimum(
        cv2.distanceTransform(support.astype(np.uint8), cv2.DIST_L2, 3)
        / edge_width,
        1.0,
    )
    centered = np.asarray(detail, dtype=np.float32).copy()
    if np.any(support):
        centered -= float(np.mean(centered[support]))
    centered *= taper
    frequency_y = np.fft.fftfreq(centered.shape[0], d=spacing_mm)
    frequency_x = np.fft.fftfreq(centered.shape[1], d=spacing_mm)
    radial_frequency = np.hypot(
        frequency_y[:, None], frequency_x[None, :]
    )
    transfer = np.interp(
        radial_frequency, frequencies, gains, left=gains[0], right=gains[-1]
    )
    inverse = (1.0 + regularization) * transfer / (
        transfer * transfer + regularization
    )
    inverse = np.clip(inverse, 1.0, max_inverse_gain)
    spectrum = np.fft.fft2(centered)
    corrected = np.fft.ifft2(spectrum * inverse).real.astype(np.float32)
    corrected *= support
    if np.any(support):
        corrected -= float(np.mean(corrected[support])) * support
    power = np.abs(spectrum) ** 2
    power_weighted_gain = float(
        np.sum(power * inverse) / max(float(np.sum(power)), 1e-8)
    )
    return corrected, {
        "mechanical_mtf_regularization": regularization,
        "mechanical_mtf_max_inverse_gain": max_inverse_gain,
        "mechanical_mtf_power_weighted_inverse_gain": power_weighted_gain,
    }


def _masked_gaussian(field, support, sigma_pixels):
    weights = support.astype(np.float32)
    return cv2.GaussianBlur(
        field.astype(np.float32) * weights, (0, 0), sigma_pixels
    ) / np.maximum(cv2.GaussianBlur(weights, (0, 0), sigma_pixels), 1e-5)


def _appearance_guided_geometry(
    geometry_detail,
    recovered_appearance,
    appearance_confidence,
    carrier_confidence,
    fused_mask,
    sigma_pixels,
    confidence_calibration=None,
):
    empty = np.zeros_like(geometry_detail, dtype=np.float32)
    interior = cv2.erode(
        fused_mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    if np.count_nonzero(interior) < 16:
        return geometry_detail, empty, empty, 0.0, 0.0

    appearance_local = _masked_gaussian(
        recovered_appearance, fused_mask, sigma_pixels
    )
    geometry_local = _masked_gaussian(
        geometry_detail, fused_mask, sigma_pixels
    )
    appearance_detail = recovered_appearance - appearance_local
    centered_geometry = geometry_detail - geometry_local
    covariance = _masked_gaussian(
        appearance_detail * centered_geometry, fused_mask, sigma_pixels
    )
    appearance_variance = _masked_gaussian(
        appearance_detail * appearance_detail, fused_mask, sigma_pixels
    )
    geometry_variance = _masked_gaussian(
        centered_geometry * centered_geometry, fused_mask, sigma_pixels
    )
    correlation = covariance / np.sqrt(
        np.maximum(appearance_variance * geometry_variance, 1e-12)
    )
    appearance_gradient_y, appearance_gradient_x = np.gradient(
        appearance_detail
    )
    geometry_gradient_y, geometry_gradient_x = np.gradient(centered_geometry)

    def orientation_tensor(gradient_x, gradient_y):
        xx = _masked_gaussian(
            gradient_x * gradient_x, fused_mask, sigma_pixels
        )
        xy = _masked_gaussian(
            gradient_x * gradient_y, fused_mask, sigma_pixels
        )
        yy = _masked_gaussian(
            gradient_y * gradient_y, fused_mask, sigma_pixels
        )
        angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
        anisotropy = np.sqrt((xx - yy) ** 2 + 4.0 * xy * xy) / np.maximum(
            xx + yy, 1e-10
        )
        return angle, np.clip(anisotropy, 0.0, 1.0)

    appearance_angle, appearance_anisotropy = orientation_tensor(
        appearance_gradient_x, appearance_gradient_y
    )
    geometry_angle, geometry_anisotropy = orientation_tensor(
        geometry_gradient_x, geometry_gradient_y
    )
    orientation_agreement = (
        np.abs(np.cos(appearance_angle - geometry_angle))
        * np.sqrt(appearance_anisotropy * geometry_anisotropy)
    )
    geometry_rms = float(
        np.sqrt(np.mean(centered_geometry[interior] ** 2))
    )
    geometry_evidence = np.clip(
        np.sqrt(np.maximum(geometry_variance, 0.0))
        / max(0.35 * geometry_rms, 2e-4),
        0.0,
        1.0,
    )
    global_alignment = 0.0
    if (
        np.std(appearance_detail[interior]) > 1e-6
        and np.std(centered_geometry[interior]) > 1e-6
    ):
        global_alignment = float(
            np.corrcoef(
                appearance_detail[interior], centered_geometry[interior]
            )[0, 1]
        )
    agreement_evidence = np.maximum(
        np.abs(correlation), orientation_agreement
    )
    agreement = np.clip((agreement_evidence - 0.18) / 0.55, 0.0, 1.0)
    raw_confidence = (
        agreement
        * geometry_evidence
        * np.clip(appearance_confidence, 0.0, 1.0)
        * np.clip(carrier_confidence, 0.0, 1.0)
        * fused_mask
    )
    confidence = _calibrate_confidence(
        raw_confidence,
        confidence_calibration,
        "calibrated_probability",
    ) * (raw_confidence > 1e-6) * fused_mask
    gain_sign = np.where(
        np.abs(correlation) > 0.18,
        np.sign(correlation),
        1.0 if global_alignment >= 0.0 else -1.0,
    )
    local_gain = 4.0 * gain_sign * np.sqrt(
        np.maximum(geometry_variance, 0.0)
        / np.maximum(appearance_variance, 1e-6)
    )
    gain_limit = max(2.5 * geometry_rms, 1e-3)
    guided_detail = np.clip(
        local_gain * appearance_detail, -gain_limit, gain_limit
    )
    fused_detail = geometry_detail + confidence * (
        guided_detail - centered_geometry
    )
    return (
        fused_detail.astype(np.float32),
        confidence.astype(np.float32),
        raw_confidence.astype(np.float32),
        global_alignment,
        float(np.mean(confidence[interior])),
    )


def _multiscale_carrier_lk(
    pre_image,
    post_image,
    sensor_mask,
    pixels_per_pitch,
    spacing_mm,
):
    sigma = max(1.0, 1.5 * pixels_per_pitch)

    def carrier_band(image):
        image_float = image.astype(np.float32)
        highpass = image_float - cv2.GaussianBlur(
            image_float, (0, 0), sigma
        )
        scale = float(np.percentile(np.abs(highpass[sensor_mask]), 95))
        return np.clip(128.0 + 96.0 * highpass / max(scale, 1.0), 0, 255).astype(
            np.uint8
        )

    pre_carrier = carrier_band(pre_image)
    post_carrier = carrier_band(post_image)
    stride = 2
    grid_x = np.arange(0, pre_image.shape[1], stride, dtype=np.float32)
    grid_y = np.arange(0, pre_image.shape[0], stride, dtype=np.float32)
    points_x, points_y = np.meshgrid(grid_x, grid_y)
    points = np.column_stack((points_x.ravel(), points_y.ravel())).astype(
        np.float32
    )[:, None, :]
    displacement_x = []
    displacement_y = []
    confidences = []
    for multiplier in (2.5, 4.5):
        window = max(9, int(round(multiplier * pixels_per_pitch)) | 1)
        next_points, forward_status, eigenvalue = cv2.calcOpticalFlowPyrLK(
            pre_carrier,
            post_carrier,
            points,
            None,
            winSize=(window, window),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                24,
                0.01,
            ),
            flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS,
            minEigThreshold=1e-5,
        )
        back_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
            post_carrier,
            pre_carrier,
            next_points,
            None,
            winSize=(window, window),
            maxLevel=2,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                24,
                0.01,
            ),
        )
        forward_status = forward_status.ravel().astype(bool)
        backward_status = backward_status.ravel().astype(bool)
        valid = forward_status & backward_status
        forward_backward_error = np.linalg.norm(
            back_points[:, 0, :] - points[:, 0, :], axis=1
        )
        eigenvalue = eigenvalue.ravel()
        eigen_scale = (
            float(np.percentile(eigenvalue[valid], 85))
            if np.any(valid)
            else 1.0
        )
        confidence = (
            valid
            * np.clip(eigenvalue / max(eigen_scale, 1e-8), 0.0, 1.0)
            * np.exp(-0.5 * (forward_backward_error / 0.35) ** 2)
        )
        motion = (next_points[:, 0, :] - points[:, 0, :]) * spacing_mm
        coarse_shape = points_x.shape
        displacement_x.append(
            cv2.resize(
                motion[:, 0].reshape(coarse_shape),
                pre_image.shape[::-1],
                interpolation=cv2.INTER_CUBIC,
            )
        )
        displacement_y.append(
            cv2.resize(
                motion[:, 1].reshape(coarse_shape),
                pre_image.shape[::-1],
                interpolation=cv2.INTER_CUBIC,
            )
        )
        confidences.append(
            cv2.resize(
                confidence.astype(np.float32).reshape(coarse_shape),
                pre_image.shape[::-1],
                interpolation=cv2.INTER_LINEAR,
            )
            * sensor_mask
        )
    confidence_stack = np.clip(np.asarray(confidences), 0.0, 1.0)
    confidence_sum = np.sum(confidence_stack, axis=0)
    return (
        np.sum(confidence_stack * np.asarray(displacement_x), axis=0)
        / np.maximum(confidence_sum, 1e-6),
        np.sum(confidence_stack * np.asarray(displacement_y), axis=0)
        / np.maximum(confidence_sum, 1e-6),
        np.clip(confidence_sum / len(confidences), 0.0, 1.0),
    )


def _recover_carrier_phase_detail(
    config,
    pre_image,
    post_image,
    recovered_appearance,
    appearance_confidence,
    fused_mask,
    spacing_mm,
    slope_to_shift_mm,
    carrier_to_displacement_scale,
    carrier_highpass_mm,
    mechanical_mtf_calibration=None,
    confidence_calibration=None,
    abstention_threshold=0.20,
):
    pixels_per_pitch = (
        (pre_image.shape[0] - 1)
        * float(config["grating_pitch_mm"])
        / (2.0 * float(config["sensor_radius_mm"]))
    )
    enabled = pixels_per_pitch >= MIN_RAW_PIXELS_PER_GRATING_PITCH
    empty = np.zeros_like(recovered_appearance, dtype=np.float32)
    diagnostics = {
        "enabled": enabled,
        "method": "joint_wrapped_multicarrier_phase_lk_init",
        "pixels_per_grating_pitch": float(pixels_per_pitch),
        "unique_carrier_count": 0,
        "joint_observation_count": 0,
        "carrier_confidence_mean": 0.0,
        "carrier_phase_residual_rad": 0.0,
        "appearance_geometry_alignment": 0.0,
        "appearance_geometry_confidence_mean": 0.0,
        "raw_appearance_geometry_confidence_mean": 0.0,
        "reconstruction_confidence_mean": 0.0,
        "abstention_coverage": 0.0,
        "appearance_flow_alignment": 0.0,
        "appearance_guide_weight": 0.0,
    }
    if not enabled or not np.any(fused_mask):
        return {
            "detail_mm": empty,
            "carrier_only_detail_mm": empty,
            "appearance_confidence": empty,
            "raw_appearance_confidence": empty,
            "diagnostics": diagnostics,
            "displacement_mm": np.stack((empty, empty)),
            "carrier_confidence": empty,
            "reconstruction_confidence": empty,
            "uncertainty": np.ones_like(empty),
            "expected_error_mm": np.full_like(empty, 0.12),
            "abstention_mask": np.zeros_like(fused_mask),
        }

    pitch = float(config["grating_pitch_mm"])
    carriers = []
    rows = []
    carrier_names = []
    for direction in _pattern_directions(config["pattern"]):
        angle_a = math.radians(float(config["grating_angle_a_deg"]) + direction)
        angle_b = math.radians(float(config["grating_angle_b_deg"]) + direction)
        top_scale = float(config["top_layer_motion_scale"])
        bottom_scale = float(config["bottom_layer_motion_scale"])
        top_carrier = 2.0 * np.pi * spacing_mm * np.asarray(
            (math.cos(angle_a), math.sin(angle_a)), dtype=np.float32
        ) / pitch
        bottom_carrier = 2.0 * np.pi * spacing_mm * np.asarray(
            (math.cos(angle_b), math.sin(angle_b)), dtype=np.float32
        ) / pitch
        top_row = -2.0 * np.pi * top_scale * np.asarray(
            (math.cos(angle_a), math.sin(angle_a)), dtype=np.float32
        ) / pitch
        bottom_row = -2.0 * np.pi * bottom_scale * np.asarray(
            (math.cos(angle_b), math.sin(angle_b)), dtype=np.float32
        ) / pitch
        for name, carrier, row in (
            ("top", top_carrier, top_row),
            ("bottom", bottom_carrier, bottom_row),
            ("sum", top_carrier + bottom_carrier, top_row + bottom_row),
            (
                "difference",
                top_carrier - bottom_carrier,
                top_row - bottom_row,
            ),
        ):
            carriers.append(tuple(carrier))
            rows.append(tuple(row))
            carrier_names.append(f"{direction:g}:{name}")
    rows = np.asarray(rows, dtype=np.float32)
    scale_sigmas = pixels_per_pitch * np.asarray((0.65, 1.05, 1.70))
    scale_priors = (1.0, 0.55, 0.30)
    phase_observations = []
    weight_observations = []
    row_observations = []
    _, _, sensor_mask = _sensor_grid(config)
    for sigma_pixels, scale_prior in zip(scale_sigmas, scale_priors):
        amplitudes = []
        phases = []
        for carrier_x, carrier_y in carriers:
            pre_component = _local_carrier_component(
                pre_image, carrier_x, carrier_y, float(sigma_pixels)
            )
            post_component = _local_carrier_component(
                post_image, carrier_x, carrier_y, float(sigma_pixels)
            )
            phases.append(np.angle(post_component * np.conj(pre_component)))
            amplitudes.append(
                np.sqrt(np.abs(pre_component) * np.abs(post_component))
            )
        amplitude_scales = np.asarray(
            [float(np.percentile(value[sensor_mask], 85)) for value in amplitudes]
        )
        strength_reference = max(float(np.median(amplitude_scales)), 1e-6)
        for row, phase, amplitude, amplitude_scale in zip(
            rows, phases, amplitudes, amplitude_scales
        ):
            strength = np.clip(amplitude_scale / strength_reference, 0.5, 1.0)
            phase_observations.append(phase.astype(np.float32))
            weight_observations.append(
                np.clip(amplitude / max(float(amplitude_scale), 1e-6), 0.0, 1.0)
                * float(scale_prior)
                * float(strength)
                * sensor_mask
            )
            row_observations.append(row)
    phase_observations = np.asarray(phase_observations, dtype=np.float32)
    weight_observations = np.asarray(weight_observations, dtype=np.float32)
    row_observations = np.asarray(row_observations, dtype=np.float32)
    lk_x, lk_y, lk_confidence = _multiscale_carrier_lk(
        pre_image,
        post_image,
        sensor_mask,
        pixels_per_pitch,
        spacing_mm,
    )
    displacement_x, displacement_y, carrier_confidence, residual_rms = (
        _joint_wrapped_carrier_refinement(
            row_observations,
            phase_observations,
            weight_observations,
            lk_x,
            lk_y,
        )
    )
    background = sensor_mask & ~cv2.dilate(
        fused_mask.astype(np.uint8), np.ones((7, 7), np.uint8)
    ).astype(bool)
    if np.count_nonzero(background) > 16:
        displacement_x -= float(np.median(displacement_x[background]))
        displacement_y -= float(np.median(displacement_y[background]))
    sigma_pixels = max(1.0, carrier_highpass_mm / spacing_mm)
    local_x = _masked_gaussian(displacement_x, fused_mask, sigma_pixels)
    local_y = _masked_gaussian(displacement_y, fused_mask, sigma_pixels)
    high_x = displacement_x - local_x
    high_y = displacement_y - local_y
    geometry_detail = _integrate_gradient(
        high_x * carrier_to_displacement_scale / slope_to_shift_mm,
        high_y * carrier_to_displacement_scale / slope_to_shift_mm,
        spacing_mm,
    )
    geometry_detail -= float(np.mean(geometry_detail[fused_mask]))
    carrier_only_detail, mtf_diagnostics = _mechanical_mtf_deconvolution(
        geometry_detail,
        fused_mask,
        spacing_mm,
        mechanical_mtf_calibration,
    )
    carrier_only_detail = np.clip(carrier_only_detail, -0.20, 0.20)
    appearance_calibration = (
        confidence_calibration.get("appearance_helpfulness")
        if confidence_calibration
        else None
    )
    (
        fused_detail,
        appearance_geometry_confidence,
        raw_appearance_geometry_confidence,
        alignment,
        guide_weight,
    ) = (
        _appearance_guided_geometry(
            carrier_only_detail,
            recovered_appearance,
            appearance_confidence,
            carrier_confidence,
            fused_mask,
            sigma_pixels,
            appearance_calibration,
        )
    )

    edge_width = max(1.0, 0.20 / spacing_mm)
    edge_weight = np.minimum(
        cv2.distanceTransform(
            fused_mask.astype(np.uint8), cv2.DIST_L2, 3
        )
        / edge_width,
        1.0,
    )
    raw_reconstruction_confidence = (
        carrier_confidence
        * (0.90 + 0.10 * appearance_geometry_confidence)
        * fused_mask
    )
    reconstruction_calibration = (
        confidence_calibration.get("reconstruction_error")
        if confidence_calibration
        else None
    )
    reconstruction_confidence = _calibrate_confidence(
        raw_reconstruction_confidence,
        reconstruction_calibration,
        "calibrated_confidence",
    ) * fused_mask
    if reconstruction_calibration and "expected_error_mm" in reconstruction_calibration:
        expected_error_mm = _calibrate_confidence(
            raw_reconstruction_confidence,
            reconstruction_calibration,
            "expected_error_mm",
        )
    else:
        expected_error_mm = 0.12 * (1.0 - reconstruction_confidence)
    expected_error_mm = expected_error_mm.astype(np.float32)
    expected_error_mm[~fused_mask] = 0.12
    abstention_mask = (
        (reconstruction_confidence >= float(abstention_threshold)) & fused_mask
    )
    detail_weight = edge_weight * abstention_mask
    fused_detail = (fused_detail * detail_weight).astype(np.float32)
    carrier_only_detail = (
        carrier_only_detail * detail_weight
    ).astype(np.float32)
    diagnostics.update(
        {
            "unique_carrier_count": len(carrier_names),
            "joint_observation_count": len(row_observations),
            "lk_initialization_confidence_mean": float(
                np.mean(lk_confidence[fused_mask])
            ),
            "carrier_confidence_mean": float(
                np.mean(carrier_confidence[fused_mask])
            ),
            "carrier_phase_residual_rad": float(np.mean(residual_rms[fused_mask])),
            "appearance_geometry_alignment": alignment,
            "appearance_geometry_confidence_mean": guide_weight,
            "raw_appearance_geometry_confidence_mean": float(
                np.mean(raw_appearance_geometry_confidence[fused_mask])
            ),
            "reconstruction_confidence_mean": float(
                np.mean(reconstruction_confidence[fused_mask])
            ),
            "abstention_coverage": float(
                np.count_nonzero(abstention_mask)
                / max(1, np.count_nonzero(fused_mask))
            ),
            "appearance_flow_alignment": alignment,
            "appearance_guide_weight": guide_weight,
            **mtf_diagnostics,
        }
    )
    return {
        "detail_mm": fused_detail,
        "carrier_only_detail_mm": carrier_only_detail,
        "appearance_confidence": appearance_geometry_confidence,
        "raw_appearance_confidence": raw_appearance_geometry_confidence,
        "diagnostics": diagnostics,
        "displacement_mm": np.stack((displacement_x, displacement_y)).astype(
            np.float32
        ),
        "carrier_confidence": carrier_confidence.astype(np.float32),
        "reconstruction_confidence": reconstruction_confidence.astype(np.float32),
        "uncertainty": (1.0 - reconstruction_confidence).astype(np.float32),
        "expected_error_mm": expected_error_mm,
        "abstention_mask": abstention_mask,
    }


def reconstruct_rigid_contact(
    config,
    pre_image,
    post_image,
    see_through_image,
    slope_to_shift_mm,
    membrane_tension_n_per_mm,
    inflation_pressure_kpa,
    membrane_bending_stiffness_n_mm,
    cavity_depth_mm,
    sealed_air_coupling,
    grating_open_fraction,
    grating_line_transmittance,
    raw_flow_to_displacement_scale,
    raw_flow_highpass_mm,
    mechanical_mtf_calibration=None,
    confidence_calibration=None,
    abstention_threshold=0.20,
):
    """Recover the local interface using observations and calibrated parameters only."""
    if slope_to_shift_mm <= 0.0:
        raise ValueError("slope_to_shift_mm must be positive")
    if membrane_tension_n_per_mm <= 0.0:
        raise ValueError("membrane tension must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
    if membrane_bending_stiffness_n_mm < 0.0:
        raise ValueError("membrane bending stiffness must be non-negative")
    if cavity_depth_mm <= 0.0:
        raise ValueError("cavity depth must be positive")
    if raw_flow_to_displacement_scale <= 0.0 or raw_flow_highpass_mm <= 0.0:
        raise ValueError("carrier calibration parameters must be positive")
    xx, yy, sensor_mask = _sensor_grid(config)
    expected_shape = xx.shape
    for image in (pre_image, post_image, see_through_image):
        if np.asarray(image).shape != expected_shape:
            raise ValueError("observation arrays must match the configured image size")
    spacing_mm = 2.0 * float(config["sensor_radius_mm"]) / (xx.shape[0] - 1)
    apparent_shift = _recover_displacement(
        config, pre_image, post_image, sensor_mask
    )
    recovered_appearance, appearance_confidence = _recover_object_appearance(
        config,
        pre_image,
        see_through_image,
        apparent_shift,
        sensor_mask,
        grating_open_fraction,
        grating_line_transmittance,
    )
    estimated_object_mask = _estimate_object_mask(
        recovered_appearance, xx, yy, sensor_mask
    )
    if not np.any(estimated_object_mask):
        raise ValueError("see-through silhouette could not be recovered")

    membrane_height = _integrate_gradient(
        cv2.GaussianBlur(apparent_shift[0], (0, 0), 0.7)
        / slope_to_shift_mm,
        cv2.GaussianBlur(apparent_shift[1], (0, 0), 0.7)
        / slope_to_shift_mm,
        spacing_mm,
    )
    radius = np.hypot(xx, yy)
    boundary_ring = sensor_mask & (
        radius > 0.86 * float(config["sensor_radius_mm"])
    )
    if membrane_height[estimated_object_mask].mean() < membrane_height[
        boundary_ring
    ].mean():
        membrane_height *= -1.0
    membrane_height -= np.median(membrane_height[boundary_ring])
    membrane_height = np.clip(membrane_height, 0.0, None)

    support_threshold = max(
        0.015,
        0.05 * float(np.percentile(membrane_height[estimated_object_mask], 99)),
    )
    moire_mask = (membrane_height > support_threshold) & sensor_mask
    fused_mask = moire_mask & estimated_object_mask
    fused_mask = cv2.morphologyEx(
        fused_mask.astype(np.uint8),
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
    ).astype(bool)
    membrane_height *= sensor_mask
    moire_height = membrane_height * moire_mask
    coarse_fused_height = membrane_height * fused_mask
    carrier_reconstruction = _recover_carrier_phase_detail(
        config,
        pre_image,
        post_image,
        recovered_appearance,
        appearance_confidence,
        fused_mask,
        spacing_mm,
        slope_to_shift_mm,
        raw_flow_to_displacement_scale,
        raw_flow_highpass_mm,
        mechanical_mtf_calibration,
        confidence_calibration,
        abstention_threshold,
    )
    high_frequency_detail = carrier_reconstruction["detail_mm"]
    fused_height = np.clip(
        membrane_height + high_frequency_detail, 0.0, None
    ) * fused_mask
    carrier_only_height = np.clip(
        membrane_height + carrier_reconstruction["carrier_only_detail_mm"],
        0.0,
        None,
    ) * fused_mask
    see_through_height = 0.5 * estimated_object_mask
    if sealed_air_coupling:
        cavity_interior = cv2.erode(
            sensor_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        recovered_pressure_kpa, _ = _sealed_cavity_pressure(
            inflation_pressure_kpa,
            membrane_height,
            cavity_interior,
            spacing_mm,
            cavity_depth_mm,
        )
    else:
        recovered_pressure_kpa = float(inflation_pressure_kpa)
    pressure_surface = cv2.GaussianBlur(membrane_height, (0, 0), 1.5)
    recovered_pressure = np.maximum(
        -membrane_tension_n_per_mm
        * _laplacian(pressure_surface, spacing_mm)
        + membrane_bending_stiffness_n_mm
        * _biharmonic(pressure_surface, spacing_mm)
        + recovered_pressure_kpa * 1e-3,
        0.0,
    ) * fused_mask
    return {
        "estimated_object_mask": estimated_object_mask,
        "moire_only_mask": moire_mask,
        "fused_mask": fused_mask,
        "reconstructed_membrane_height_mm": membrane_height.astype(np.float32),
        "coarse_fused_height_mm": coarse_fused_height.astype(np.float32),
        "high_frequency_detail_mm": high_frequency_detail.astype(np.float32),
        "carrier_only_detail_mm": carrier_reconstruction[
            "carrier_only_detail_mm"
        ].astype(np.float32),
        "carrier_only_height_mm": carrier_only_height.astype(np.float32),
        "see_through_only_height_mm": see_through_height.astype(np.float32),
        "moire_only_height_mm": moire_height.astype(np.float32),
        "fused_height_mm": fused_height.astype(np.float32),
        "reconstructed_pressure_n_per_mm2": recovered_pressure.astype(np.float32),
        "recovered_appearance": recovered_appearance,
        "appearance_confidence": appearance_confidence,
        "appearance_geometry_confidence": carrier_reconstruction[
            "appearance_confidence"
        ],
        "raw_appearance_geometry_confidence": carrier_reconstruction[
            "raw_appearance_confidence"
        ],
        "carrier_confidence": carrier_reconstruction["carrier_confidence"],
        "reconstruction_confidence": carrier_reconstruction[
            "reconstruction_confidence"
        ],
        "reconstruction_uncertainty": carrier_reconstruction["uncertainty"],
        "expected_error_mm": carrier_reconstruction["expected_error_mm"],
        "abstention_mask": carrier_reconstruction["abstention_mask"],
        "carrier_displacement_mm": carrier_reconstruction["displacement_mm"],
        "high_frequency": carrier_reconstruction["diagnostics"],
    }


def _point_cloud(xx, yy, height, mask):
    return np.column_stack((xx[mask], yy[mask], height[mask])).astype(np.float32)


def _surface_normals(height, spacing_mm):
    gradient_y, gradient_x = np.gradient(height, spacing_mm)
    normals = np.stack((-gradient_x, -gradient_y, np.ones_like(height)), axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _texture_metrics(target_height, reconstructed_height, support, spacing_mm):
    if not np.any(support):
        return {
            "texture_target_rms_mm": 0.0,
            "texture_reconstructed_rms_mm": 0.0,
            "texture_amplitude_gain": 0.0,
            "texture_rmse_mm": math.inf,
            "texture_nrmse": math.inf,
            "texture_correlation": 0.0,
        }
    sigma_pixels = max(0.8, 1.0 / spacing_mm)
    support_float = support.astype(np.float32)
    blurred_support = cv2.GaussianBlur(
        support_float, (0, 0), sigma_pixels
    )

    def residual(height):
        local_form = cv2.GaussianBlur(
            height.astype(np.float32) * support_float,
            (0, 0),
            sigma_pixels,
        ) / np.maximum(blurred_support, 1e-6)
        return height - local_form

    target_texture = residual(target_height)[support]
    reconstructed_texture = residual(reconstructed_height)[support]
    target_rms = float(np.sqrt(np.mean(target_texture * target_texture)))
    reconstructed_rms = float(
        np.sqrt(np.mean(reconstructed_texture * reconstructed_texture))
    )
    texture_rmse = float(
        np.sqrt(np.mean((reconstructed_texture - target_texture) ** 2))
    )
    correlation = 0.0
    if (
        len(target_texture) > 1
        and np.std(target_texture) > 1e-6
        and np.std(reconstructed_texture) > 1e-6
    ):
        correlation = float(
            np.corrcoef(target_texture, reconstructed_texture)[0, 1]
        )
    return {
        "texture_target_rms_mm": target_rms,
        "texture_reconstructed_rms_mm": reconstructed_rms,
        "texture_amplitude_gain": (
            reconstructed_rms / target_rms if target_rms > 1e-6 else 0.0
        ),
        "texture_rmse_mm": texture_rmse,
        "texture_nrmse": texture_rmse / max(target_rms, 1e-8),
        "texture_correlation": correlation,
    }


def _chamfer_distance_mm(first, second):
    if not len(first) or not len(second):
        return math.inf
    return float(
        0.5
        * (
            cKDTree(first).query(second)[0].mean()
            + cKDTree(second).query(first)[0].mean()
        )
    )


def simulate_rigid_object_poc(
    config,
    object_type="screwdriver",
    rotation_deg=20.0,
    texture_frequency=1.2,
    visual_texture_frequency=0.7,
    relief_scale=1.0,
    indentation_mm=0.85,
    offset_x_mm=0.0,
    offset_y_mm=0.0,
    membrane_tension_n_per_mm=0.08,
    inflation_pressure_kpa=4.0,
    membrane_bending_stiffness_n_mm=1e-4,
    cavity_depth_mm=8.0,
    sealed_air_coupling=True,
    camera_psf_sigma=0.45,
    camera_supersample=2,
    grating_open_fraction=0.82,
    grating_line_transmittance=0.10,
    noise_std=0.004,
    slope_to_shift_mm=0.12,
    raw_flow_to_displacement_scale=1.0,
    raw_flow_highpass_mm=1.2,
    seed=7,
    height_map_mm=None,
    albedo_map=None,
    mechanical_mtf_calibration=None,
    confidence_calibration=None,
    abstention_threshold=0.20,
):
    if texture_frequency <= 0.0 or visual_texture_frequency <= 0.0:
        raise ValueError("texture frequencies must be positive")
    if relief_scale <= 0.0 or indentation_mm <= 0.0:
        raise ValueError("relief scale and indentation must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
    if membrane_bending_stiffness_n_mm < 0.0:
        raise ValueError("membrane bending stiffness must be non-negative")
    if cavity_depth_mm <= 0.0:
        raise ValueError("cavity depth must be positive")
    if camera_psf_sigma < 0.0 or noise_std < 0.0:
        raise ValueError("camera PSF and noise must be non-negative")
    if int(camera_supersample) != camera_supersample or camera_supersample < 1:
        raise ValueError("camera supersample must be a positive integer")
    if raw_flow_to_displacement_scale <= 0.0 or raw_flow_highpass_mm <= 0.0:
        raise ValueError("carrier calibration parameters must be positive")
    if not 0.0 <= float(abstention_threshold) <= 1.0:
        raise ValueError("abstention threshold must be between zero and one")
    config = dict(config, pattern="cross")
    xx, yy, sensor_mask = _sensor_grid(config)
    spacing_mm = 2.0 * float(config["sensor_radius_mm"]) / (xx.shape[0] - 1)

    if height_map_mm is None:
        if albedo_map is not None:
            raise ValueError("albedo_map requires height_map_mm")
        raw_height, albedo, object_mask = _object_height_field(
            object_type,
            xx,
            yy,
            rotation_deg,
            texture_frequency,
            visual_texture_frequency,
            offset_x_mm,
            offset_y_mm,
        )
    else:
        raw_height = np.asarray(height_map_mm, dtype=np.float32)
        if raw_height.shape != xx.shape or not np.isfinite(raw_height).all():
            raise ValueError("height_map_mm must be finite and match the sensor grid")
        object_mask = raw_height > 0.0
        if albedo_map is None:
            albedo = np.where(object_mask, 0.66, 0.20).astype(np.float32)
        else:
            albedo = np.asarray(albedo_map, dtype=np.float32)
            if albedo.shape != xx.shape or not np.isfinite(albedo).all():
                raise ValueError("albedo_map must be finite and match the sensor grid")
            albedo = np.clip(albedo, 0.0, 1.0)
        object_type = "height_map"
    object_mask &= sensor_mask
    if not np.any(object_mask):
        raise ValueError("object does not overlap the sensor")
    raw_height = raw_height * float(relief_scale)
    peak_height = float(np.max(raw_height[object_mask]))
    relief_depth = np.zeros_like(raw_height)
    relief_depth[object_mask] = peak_height - raw_height[object_mask]
    rigid_surface = float(indentation_mm) - relief_depth
    obstacle = np.maximum(rigid_surface, 0.0) * object_mask

    physics = _solve_membrane_contact(
        obstacle,
        sensor_mask,
        spacing_mm,
        float(membrane_tension_n_per_mm),
        float(inflation_pressure_kpa),
        float(membrane_bending_stiffness_n_mm),
        float(cavity_depth_mm),
        bool(sealed_air_coupling),
    )
    membrane_height = physics["membrane_height_mm"]
    gradient_y, gradient_x = np.gradient(membrane_height, spacing_mm)
    apparent_shift = (
        float(slope_to_shift_mm) * gradient_x,
        float(slope_to_shift_mm) * gradient_y,
    )

    camera_size = xx.shape[0] * int(camera_supersample)
    camera_config = dict(config, image_size=camera_size)
    camera_xx, camera_yy, camera_mask = _sensor_grid(camera_config)
    camera_albedo = cv2.resize(
        albedo, (camera_size, camera_size), interpolation=cv2.INTER_CUBIC
    )
    camera_shift = tuple(
        cv2.resize(
            component,
            (camera_size, camera_size),
            interpolation=cv2.INTER_CUBIC,
        )
        for component in apparent_shift
    )
    optical = {
        "gain": sum(camera_config["brightness_gain"]) / 2.0,
        "offset": sum(camera_config["brightness_offset"]) / 2.0,
    }
    rng = np.random.default_rng(seed)
    optical_albedo = _gaussian_blur(camera_albedo, float(camera_psf_sigma))
    pre_transmission = _render_grating_transmission(
        camera_config,
        None,
        float(grating_open_fraction),
        float(grating_line_transmittance),
    )
    post_transmission = _render_grating_transmission(
        camera_config,
        camera_shift,
        float(grating_open_fraction),
        float(grating_line_transmittance),
    )
    pre_envelope = _render_moire_envelope(
        camera_config, camera_xx, camera_yy, None
    )
    post_envelope = _render_moire_envelope(
        camera_config, camera_xx, camera_yy, camera_shift
    )
    pre_image = _render_shared_observation(
        camera_config,
        camera_xx,
        camera_yy,
        optical_albedo,
        pre_transmission * pre_envelope,
        camera_mask,
        optical,
        noise_std,
        rng,
    )
    post_image = _render_shared_observation(
        camera_config,
        camera_xx,
        camera_yy,
        optical_albedo,
        post_transmission * post_envelope,
        camera_mask,
        optical,
        noise_std,
        rng,
    )
    see_through_image = post_image.copy()

    reconstruction = reconstruct_rigid_contact(
        camera_config,
        pre_image,
        post_image,
        see_through_image,
        float(slope_to_shift_mm),
        float(membrane_tension_n_per_mm),
        float(inflation_pressure_kpa),
        float(membrane_bending_stiffness_n_mm),
        float(cavity_depth_mm),
        bool(sealed_air_coupling),
        float(grating_open_fraction),
        float(grating_line_transmittance),
        float(raw_flow_to_displacement_scale),
        float(raw_flow_highpass_mm),
        mechanical_mtf_calibration,
        confidence_calibration,
        float(abstention_threshold),
    )

    physical_size = xx.shape[0]

    def resize_field(field):
        return cv2.resize(
            np.asarray(field, dtype=np.float32),
            (physical_size, physical_size),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

    def resize_mask(mask):
        return resize_field(mask.astype(np.float32)) >= 0.5

    reconstructed_membrane_height = resize_field(
        reconstruction["reconstructed_membrane_height_mm"]
    )
    coarse_fused_height = resize_field(reconstruction["coarse_fused_height_mm"])
    high_frequency_detail = resize_field(
        reconstruction["high_frequency_detail_mm"]
    )
    carrier_only_detail = resize_field(
        reconstruction["carrier_only_detail_mm"]
    )
    carrier_only_height = resize_field(
        reconstruction["carrier_only_height_mm"]
    )
    reconstructed_apparent_shift = np.stack(
        [resize_field(field) for field in reconstruction["carrier_displacement_mm"]]
    )
    see_through_height = resize_field(
        reconstruction["see_through_only_height_mm"]
    )
    moire_only_height = resize_field(reconstruction["moire_only_height_mm"])
    fused_height = resize_field(reconstruction["fused_height_mm"])
    reconstructed_pressure = resize_field(
        reconstruction["reconstructed_pressure_n_per_mm2"]
    )
    recovered_appearance_low = resize_field(
        reconstruction["recovered_appearance"]
    )
    appearance_geometry_confidence_low = resize_field(
        reconstruction["appearance_geometry_confidence"]
    )
    raw_appearance_geometry_confidence_low = resize_field(
        reconstruction["raw_appearance_geometry_confidence"]
    )
    carrier_confidence_low = resize_field(
        reconstruction["carrier_confidence"]
    )
    reconstruction_confidence_low = resize_field(
        reconstruction["reconstruction_confidence"]
    )
    reconstruction_uncertainty_low = resize_field(
        reconstruction["reconstruction_uncertainty"]
    )
    expected_error_low = resize_field(reconstruction["expected_error_mm"])
    abstention_mask_low = resize_mask(reconstruction["abstention_mask"])
    optical_albedo_low = resize_field(optical_albedo)
    post_transmission_low = resize_field(post_transmission)
    estimated_object_mask = resize_mask(reconstruction["estimated_object_mask"])
    fused_mask = resize_mask(reconstruction["fused_mask"])
    target_mask = physics["obstacle_height_mm"] > 0.01
    target_height = physics["obstacle_height_mm"] * target_mask
    height_scale = max(float(np.max(target_height)), 1e-8)

    def nrmse(prediction):
        return float(
            np.sqrt(
                np.mean(
                    (prediction[sensor_mask] - target_height[sensor_mask]) ** 2
                )
            )
            / height_scale
        )

    see_nrmse = nrmse(see_through_height)
    moire_nrmse = nrmse(moire_only_height)
    coarse_nrmse = nrmse(coarse_fused_height)
    fusion_nrmse = nrmse(fused_height)
    union = target_mask | fused_mask
    common = target_mask & fused_mask
    mask_iou = float(np.count_nonzero(common) / max(1, np.count_nonzero(union)))
    common_fused = fused_height[common]
    common_target = target_height[common]
    correlation = 0.0
    if (
        len(common_fused) > 1
        and np.std(common_fused) > 1e-6
        and np.std(common_target) > 1e-6
    ):
        correlation = float(np.corrcoef(common_fused, common_target)[0, 1])
    target_normals = _surface_normals(target_height, spacing_mm)
    fused_normals = _surface_normals(fused_height, spacing_mm)
    normal_cosine = np.clip(
        np.sum(target_normals[common] * fused_normals[common], axis=-1),
        -1.0,
        1.0,
    )
    normal_error_deg = (
        float(np.degrees(np.arccos(normal_cosine)).mean())
        if len(normal_cosine)
        else math.inf
    )
    target_points = _point_cloud(xx, yy, target_height, target_mask)
    reconstructed_points = _point_cloud(xx, yy, fused_height, fused_mask)
    best_baseline = min(see_nrmse, moire_nrmse)
    metrics = {
        "height_rmse_mm": fusion_nrmse * height_scale,
        "height_nrmse": fusion_nrmse,
        "height_correlation": correlation,
        "mask_iou": mask_iou,
        "normal_error_deg": normal_error_deg,
        "chamfer_mm": _chamfer_distance_mm(target_points, reconstructed_points),
        "see_through_only_nrmse": see_nrmse,
        "moire_only_nrmse": moire_nrmse,
        "coarse_height_nrmse": coarse_nrmse,
        "fusion_gain_vs_best_baseline": 1.0
        - fusion_nrmse / max(best_baseline, 1e-8),
    }
    metrics.update(
        _texture_metrics(target_height, fused_height, common, spacing_mm)
    )
    coarse_texture = _texture_metrics(
        target_height, coarse_fused_height, common, spacing_mm
    )
    carrier_only_texture = _texture_metrics(
        target_height, carrier_only_height, common, spacing_mm
    )
    physics_texture = _texture_metrics(
        target_height, membrane_height, target_mask, spacing_mm
    )
    inverse_texture = _texture_metrics(
        membrane_height, fused_height, common, spacing_mm
    )
    metrics.update(
        {
            "object_to_membrane_texture_nrmse": physics_texture[
                "texture_nrmse"
            ],
            "object_to_membrane_texture_correlation": physics_texture[
                "texture_correlation"
            ],
            "membrane_to_reconstruction_texture_nrmse": inverse_texture[
                "texture_nrmse"
            ],
            "membrane_to_reconstruction_texture_correlation": inverse_texture[
                "texture_correlation"
            ],
            "coarse_texture_nrmse": coarse_texture["texture_nrmse"],
            "coarse_texture_correlation": coarse_texture[
                "texture_correlation"
            ],
            "carrier_only_texture_nrmse": carrier_only_texture[
                "texture_nrmse"
            ],
            "carrier_only_texture_correlation": carrier_only_texture[
                "texture_correlation"
            ],
        }
    )
    recovered_appearance_texture = _texture_metrics(
        albedo, recovered_appearance_low, object_mask, spacing_mm
    )
    metrics.update(
        {
            "recovered_appearance_texture_nrmse": recovered_appearance_texture[
                "texture_nrmse"
            ],
            "recovered_appearance_texture_correlation": recovered_appearance_texture[
                "texture_correlation"
            ],
        }
    )
    metrics.update(
        _optical_texture_metrics(
            albedo,
            optical_albedo_low,
            object_mask,
            post_transmission_low,
            spacing_mm,
        )
    )
    return {
        "model_version": 8,
        "object_type": object_type,
        "axis_mm": np.linspace(
            -float(config["sensor_radius_mm"]),
            float(config["sensor_radius_mm"]),
            xx.shape[0],
        ),
        "camera_axis_mm": np.linspace(
            -float(config["sensor_radius_mm"]),
            float(config["sensor_radius_mm"]),
            camera_size,
        ),
        "camera_supersample": int(camera_supersample),
        "camera_pixels_per_grating_pitch": reconstruction["high_frequency"][
            "pixels_per_grating_pitch"
        ],
        "high_frequency_enabled": reconstruction["high_frequency"]["enabled"],
        "high_frequency_diagnostics": reconstruction["high_frequency"],
        "sensor_mask": sensor_mask,
        "camera_mask": camera_mask,
        "object_mask": object_mask,
        "target_mask": target_mask,
        "contact_mask": physics["contact_mask"],
        "estimated_object_mask": estimated_object_mask,
        "estimated_mask": fused_mask,
        "ground_truth_height_mm": target_height.astype(np.float32),
        "ground_truth_appearance": albedo.astype(np.float32),
        "membrane_height_mm": membrane_height,
        "contact_pressure_n_per_mm2": physics["contact_pressure_n_per_mm2"],
        "reconstructed_membrane_height_mm": reconstructed_membrane_height,
        "coarse_reconstructed_height_mm": coarse_fused_height,
        "high_frequency_detail_mm": high_frequency_detail,
        "carrier_only_detail_mm": carrier_only_detail,
        "carrier_only_height_mm": carrier_only_height,
        "ground_truth_apparent_shift_mm": np.stack(apparent_shift).astype(
            np.float32
        ),
        "reconstructed_apparent_shift_mm": reconstructed_apparent_shift,
        "reconstructed_height_mm": fused_height,
        "reconstructed_pressure_n_per_mm2": reconstructed_pressure,
        "see_through_only_height_mm": see_through_height,
        "moire_only_height_mm": moire_only_height,
        "ground_truth_point_cloud_mm": target_points,
        "ground_truth_point_appearance": albedo[target_mask].astype(np.float32),
        "reconstructed_point_cloud_mm": reconstructed_points,
        "reconstructed_point_appearance": recovered_appearance_low[
            fused_mask
        ].astype(np.float32),
        "ground_truth_normals": target_normals.astype(np.float32),
        "reconstructed_normals": fused_normals.astype(np.float32),
        "pre_image": pre_image,
        "post_image": post_image,
        "raw_observation_image": post_image,
        "moire_difference": post_image.astype(np.int16)
        - pre_image.astype(np.int16),
        "see_through_image": see_through_image,
        "ground_truth_albedo_image": np.clip(
            np.rint(albedo * 255.0), 0, 255
        ).astype(np.uint8),
        "recovered_appearance_image": np.clip(
            np.rint(reconstruction["recovered_appearance"] * 255.0), 0, 255
        ).astype(np.uint8),
        "recovered_appearance": recovered_appearance_low,
        "appearance_confidence": reconstruction["appearance_confidence"],
        "appearance_geometry_confidence": appearance_geometry_confidence_low,
        "raw_appearance_geometry_confidence": (
            raw_appearance_geometry_confidence_low
        ),
        "carrier_confidence": carrier_confidence_low,
        "reconstruction_confidence": reconstruction_confidence_low,
        "reconstruction_uncertainty": reconstruction_uncertainty_low,
        "expected_error_mm": expected_error_low,
        "abstention_mask": abstention_mask_low,
        "optical_object_image": np.clip(
            np.rint(optical_albedo * 255.0), 0, 255
        ).astype(np.uint8),
        "grating_transmission": post_transmission,
        "reference_moire_envelope": pre_envelope,
        "deformed_moire_envelope": post_envelope,
        "moire_envelope_difference": post_envelope - pre_envelope,
        "moire_envelope": post_envelope,
        "physics": {
            key: physics[key]
            for key in (
                "iterations",
                "bending_iterations",
                "bending_converged",
                "update_mm",
                "normal_force_n",
                "inflation_pressure_kpa",
                "effective_pressure_kpa",
                "sealed_air_volume_change_fraction",
                "membrane_bending_stiffness_n_mm",
                "cavity_depth_mm",
                "contact_fraction",
                "max_penetration_mm",
                "boundary_displacement_mm",
            )
        },
        "metrics": metrics,
    }
