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


def _solve_membrane_contact(
    obstacle_height_mm,
    sensor_mask,
    spacing_mm,
    membrane_tension_n_per_mm,
    inflation_pressure_kpa,
):
    """Solve local displacement from an inflated reference toward an obstacle."""
    if membrane_tension_n_per_mm <= 0.0:
        raise ValueError("membrane tension must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
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
    pressure_n_per_mm2 = float(inflation_pressure_kpa) * 1e-3
    # w is measured inward from the inflated reference, so preload lowers the
    # free membrane toward the obstacle in this local coordinate system.
    pressure_step_mm = (
        pressure_n_per_mm2
        * spacing_mm
        * spacing_mm
        / (4.0 * membrane_tension_n_per_mm)
    )

    for iteration in range(1, 1201):
        previous = displacement.copy()
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

    pressure = np.maximum(
        -membrane_tension_n_per_mm
        * _laplacian(displacement, spacing_mm)
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
        "update_mm": update_mm,
        "normal_force_n": float(np.sum(pressure) * spacing_mm * spacing_mm),
        "inflation_pressure_kpa": float(inflation_pressure_kpa),
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


def _recover_raw_flow_detail(
    config,
    pre_image,
    post_image,
    recovered_appearance,
    fused_mask,
    spacing_mm,
    slope_to_shift_mm,
    raw_flow_to_displacement_scale,
    raw_flow_highpass_mm,
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
        "pixels_per_grating_pitch": float(pixels_per_pitch),
        "appearance_flow_alignment": 0.0,
        "appearance_guide_weight": 0.0,
    }
    if not enabled or not np.any(fused_mask):
        return empty, diagnostics

    flow = cv2.calcOpticalFlowFarneback(
        pre_image,
        post_image,
        None,
        0.5,
        4,
        5,
        7,
        7,
        1.5,
        0,
    )
    flow_x = flow[..., 0] * spacing_mm
    flow_y = flow[..., 1] * spacing_mm
    sigma_pixels = max(1.0, raw_flow_highpass_mm / spacing_mm)
    high_x = flow_x - cv2.GaussianBlur(flow_x, (0, 0), sigma_pixels)
    high_y = flow_y - cv2.GaussianBlur(flow_y, (0, 0), sigma_pixels)
    flow_detail = _integrate_gradient(
        high_x * raw_flow_to_displacement_scale / slope_to_shift_mm,
        high_y * raw_flow_to_displacement_scale / slope_to_shift_mm,
        spacing_mm,
    )
    flow_detail -= float(np.mean(flow_detail[fused_mask]))

    support = fused_mask.astype(np.float32)
    local_form = cv2.GaussianBlur(
        recovered_appearance * support, (0, 0), sigma_pixels
    ) / np.maximum(
        cv2.GaussianBlur(support, (0, 0), sigma_pixels), 1e-4
    )
    appearance_detail = recovered_appearance - local_form
    interior = cv2.erode(
        fused_mask.astype(np.uint8), np.ones((5, 5), np.uint8)
    ).astype(bool)
    if np.count_nonzero(interior) > 8:
        appearance_values = appearance_detail[interior]
        flow_values = flow_detail[interior]
        if np.std(appearance_values) > 1e-6 and np.std(flow_values) > 1e-6:
            alignment = float(
                np.corrcoef(appearance_values, flow_values)[0, 1]
            )
            appearance_gain = float(
                np.dot(appearance_values, flow_values)
                / max(np.dot(appearance_values, appearance_values), 1e-8)
            )
            appearance_gain = float(np.clip(appearance_gain, -2.0, 2.0))
            guide_weight = float(np.clip((alignment - 0.20) / 0.40, 0.0, 0.8))
            flow_detail = (
                (1.0 - guide_weight) * flow_detail
                + guide_weight * appearance_gain * appearance_detail
            )
            diagnostics["appearance_flow_alignment"] = alignment
            diagnostics["appearance_guide_weight"] = guide_weight

    edge_width = max(1.0, 0.20 / spacing_mm)
    edge_weight = np.minimum(
        cv2.distanceTransform(
            fused_mask.astype(np.uint8), cv2.DIST_L2, 3
        )
        / edge_width,
        1.0,
    )
    return (flow_detail * edge_weight * fused_mask).astype(np.float32), diagnostics


def reconstruct_rigid_contact(
    config,
    pre_image,
    post_image,
    see_through_image,
    slope_to_shift_mm,
    membrane_tension_n_per_mm,
    inflation_pressure_kpa,
    grating_open_fraction,
    grating_line_transmittance,
    raw_flow_to_displacement_scale,
    raw_flow_highpass_mm,
):
    """Recover the local interface using observations and calibrated parameters only."""
    if slope_to_shift_mm <= 0.0:
        raise ValueError("slope_to_shift_mm must be positive")
    if membrane_tension_n_per_mm <= 0.0:
        raise ValueError("membrane tension must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
    if raw_flow_to_displacement_scale <= 0.0 or raw_flow_highpass_mm <= 0.0:
        raise ValueError("raw-flow calibration parameters must be positive")
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
    high_frequency_detail, high_frequency = _recover_raw_flow_detail(
        config,
        pre_image,
        post_image,
        recovered_appearance,
        fused_mask,
        spacing_mm,
        slope_to_shift_mm,
        raw_flow_to_displacement_scale,
        raw_flow_highpass_mm,
    )
    fused_height = np.clip(
        membrane_height + high_frequency_detail, 0.0, None
    ) * fused_mask
    see_through_height = 0.5 * estimated_object_mask
    recovered_pressure = np.maximum(
        -membrane_tension_n_per_mm
        * _laplacian(
            cv2.GaussianBlur(membrane_height, (0, 0), 1.5), spacing_mm
        )
        + float(inflation_pressure_kpa) * 1e-3,
        0.0,
    ) * fused_mask
    return {
        "estimated_object_mask": estimated_object_mask,
        "moire_only_mask": moire_mask,
        "fused_mask": fused_mask,
        "reconstructed_membrane_height_mm": membrane_height.astype(np.float32),
        "coarse_fused_height_mm": coarse_fused_height.astype(np.float32),
        "high_frequency_detail_mm": high_frequency_detail.astype(np.float32),
        "see_through_only_height_mm": see_through_height.astype(np.float32),
        "moire_only_height_mm": moire_height.astype(np.float32),
        "fused_height_mm": fused_height.astype(np.float32),
        "reconstructed_pressure_n_per_mm2": recovered_pressure.astype(np.float32),
        "recovered_appearance": recovered_appearance,
        "appearance_confidence": appearance_confidence,
        "high_frequency": high_frequency,
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
    texture_rmse = float(
        np.sqrt(np.mean((reconstructed_texture - target_texture) ** 2))
    )
    correlation = 0.0
    if (
        len(target_texture) > 1
        and np.std(target_texture) > 1e-8
        and np.std(reconstructed_texture) > 1e-8
    ):
        correlation = float(
            np.corrcoef(target_texture, reconstructed_texture)[0, 1]
        )
    return {
        "texture_target_rms_mm": target_rms,
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
    camera_psf_sigma=0.45,
    camera_supersample=2,
    grating_open_fraction=0.82,
    grating_line_transmittance=0.10,
    noise_std=0.004,
    slope_to_shift_mm=0.12,
    raw_flow_to_displacement_scale=1.5,
    raw_flow_highpass_mm=1.2,
    seed=7,
    height_map_mm=None,
    albedo_map=None,
):
    if texture_frequency <= 0.0 or visual_texture_frequency <= 0.0:
        raise ValueError("texture frequencies must be positive")
    if relief_scale <= 0.0 or indentation_mm <= 0.0:
        raise ValueError("relief scale and indentation must be positive")
    if inflation_pressure_kpa < 0.0:
        raise ValueError("inflation pressure must be non-negative")
    if camera_psf_sigma < 0.0 or noise_std < 0.0:
        raise ValueError("camera PSF and noise must be non-negative")
    if int(camera_supersample) != camera_supersample or camera_supersample < 1:
        raise ValueError("camera supersample must be a positive integer")
    if raw_flow_to_displacement_scale <= 0.0 or raw_flow_highpass_mm <= 0.0:
        raise ValueError("raw-flow calibration parameters must be positive")
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
        float(grating_open_fraction),
        float(grating_line_transmittance),
        float(raw_flow_to_displacement_scale),
        float(raw_flow_highpass_mm),
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
        and np.std(common_fused) > 1e-8
        and np.std(common_target) > 1e-8
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
        "model_version": 6,
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
        "optical_object_image": np.clip(
            np.rint(optical_albedo * 255.0), 0, 255
        ).astype(np.uint8),
        "grating_transmission": post_transmission,
        "moire_envelope": post_envelope,
        "physics": {
            key: physics[key]
            for key in (
                "iterations",
                "update_mm",
                "normal_force_n",
                "inflation_pressure_kpa",
                "contact_fraction",
                "max_penetration_mm",
                "boundary_displacement_mm",
            )
        },
        "metrics": metrics,
    }
