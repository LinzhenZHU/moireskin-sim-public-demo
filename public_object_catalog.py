#!/usr/bin/env python3
"""Named and deterministic procedural objects for the public R1 demo."""

import math

import numpy as np

from moire_sim_platform import _sensor_grid
from rigid_object_poc import OBJECT_TYPES


TRAINED_NAMED_OBJECTS = (
    "screwdriver",
    "satin",
    "coin",
    "thread_array",
    "bump_array",
    "coin_relief",
    "knurled_ring",
)
HELD_OUT_NAMED_OBJECTS = (
    "phillips_head",
    "gear_face",
    "herringbone_plate",
    "microgroove_plate",
)
EXPLORATORY_NAMED_OBJECTS = (
    "hex_socket_bolt",
    "spiral_groove_disk",
    "woven_grid_plate",
    "serrated_key",
)
PROCEDURAL_OBJECTS = (
    "procedural_fourier_relief",
    "procedural_scratch_field",
    "procedural_thread_patch",
    "procedural_voronoi_relief",
    "procedural_radial_lattice",
    "procedural_branching_grooves",
    "procedural_anisotropic_bands",
    "procedural_boolean_relief",
)
PUBLIC_OBJECTS = (*OBJECT_TYPES, *PROCEDURAL_OBJECTS)

OBJECT_LABELS = {
    "screwdriver": "螺丝刀",
    "satin": "缎面纹理刚性试片",
    "coin": "浮雕圆片",
    "thread_array": "多尺度螺纹阵列",
    "bump_array": "球形凸点阵列",
    "coin_relief": "硬币花纹浮雕",
    "knurled_ring": "菱形滚花圆环",
    "phillips_head": "十字槽螺丝头",
    "gear_face": "齿轮端面",
    "herringbone_plate": "人字纹硬质试片",
    "microgroove_plate": "多频微槽试片",
    "hex_socket_bolt": "内六角螺栓头",
    "spiral_groove_disk": "螺旋刻槽圆片",
    "woven_grid_plate": "编织网格硬质试片",
    "serrated_key": "锯齿钥匙",
    "procedural_fourier_relief": "随机 Fourier 浮雕",
    "procedural_scratch_field": "随机划痕表面",
    "procedural_thread_patch": "随机螺纹片",
    "procedural_voronoi_relief": "随机 Voronoi 浮雕",
    "procedural_radial_lattice": "随机径向晶格",
    "procedural_branching_grooves": "随机分叉沟槽",
    "procedural_anisotropic_bands": "随机各向异性条带",
    "procedural_boolean_relief": "随机 Boolean 浮雕",
}


def object_category(object_name):
    if object_name in TRAINED_NAMED_OBJECTS:
        return "trained named family"
    if object_name in HELD_OUT_NAMED_OBJECTS:
        return "held-out named family"
    if object_name in EXPLORATORY_NAMED_OBJECTS:
        return "exploratory named family"
    if object_name in PROCEDURAL_OBJECTS:
        return "procedural OOD"
    raise ValueError(f"unknown public object: {object_name}")


def public_object_label(object_name):
    category = object_category(object_name)
    short_category = {
        "trained named family": "trained",
        "held-out named family": "held-out",
        "exploratory named family": "exploratory",
        "procedural OOD": "procedural OOD",
    }[category]
    return f"{OBJECT_LABELS[object_name]} · {short_category}"


def _normalize(field, mask):
    values = np.asarray(field, dtype=np.float32)
    centered = values - float(np.mean(values[mask]))
    scale = float(np.percentile(np.abs(centered[mask]), 98))
    return centered / max(scale, 1e-6)


def _fourier_field(x, y, rng, frequency, components=7):
    field = np.zeros_like(x, dtype=np.float32)
    for _ in range(int(components)):
        angle = float(rng.uniform(0.0, np.pi))
        scale = float(rng.uniform(0.55, 1.55))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        coordinate = math.cos(angle) * x + math.sin(angle) * y
        field += float(rng.uniform(0.45, 1.0)) * np.cos(
            2.0 * np.pi * frequency * scale * coordinate + phase
        )
    return field


def _scratch_field(x, y, rng, frequency):
    field = np.zeros_like(x, dtype=np.float32)
    for _ in range(12):
        angle = float(rng.uniform(0.0, np.pi))
        offset = float(rng.uniform(-5.0, 5.0))
        width = float(rng.uniform(0.05, 0.18))
        coordinate = math.cos(angle) * x + math.sin(angle) * y - offset
        field -= np.exp(-0.5 * (coordinate / width) ** 2)
    field += 0.2 * _fourier_field(x, y, rng, frequency, components=3)
    return field


def _thread_field(x, y, rng, frequency):
    pitch = max(0.3, 1.0 / max(frequency, 0.1))
    axis = y + 0.18 * np.sin(0.55 * x)
    thread = np.cos(2.0 * np.pi * axis / pitch)
    taper = 0.65 + 0.35 * np.cos(0.25 * x)
    return taper * thread + 0.2 * np.cos(4.0 * np.pi * axis / pitch)


def _voronoi_field(x, y, rng):
    centers = rng.uniform((-6.5, -4.8), (6.5, 4.8), size=(18, 2))
    distance = np.stack(
        [
            np.hypot(x - float(center[0]), y - float(center[1]))
            for center in centers
        ],
        axis=0,
    )
    nearest = np.partition(distance, 1, axis=0)[:2]
    cell_edge = np.exp(-((nearest[1] - nearest[0]) / 0.24) ** 2)
    cell_fill = np.min(distance, axis=0)
    return 0.7 * cell_edge - 0.3 * cell_fill


def _radial_lattice(x, y, rng, frequency):
    center_x, center_y = rng.uniform(-1.2, 1.2, size=2)
    radius = np.hypot(x - center_x, y - center_y)
    theta = np.arctan2(y - center_y, x - center_x)
    return np.cos(2.0 * np.pi * frequency * radius) * (
        0.55 + 0.45 * np.cos(8.0 * theta)
    )


def _branching_grooves(x, y, rng, frequency):
    field = np.zeros_like(x, dtype=np.float32)
    trunk_angle = float(rng.uniform(-0.35, 0.35))
    trunk = math.cos(trunk_angle) * x + math.sin(trunk_angle) * y
    field -= np.exp(-0.5 * (trunk / 0.16) ** 2)
    for sign in (-1.0, 1.0):
        for offset in (-3.6, -1.8, 0.0, 1.8, 3.6):
            branch = (
                0.72 * x
                + sign * 0.69 * y
                - offset
                + 0.18 * np.sin(frequency * y)
            )
            field -= 0.7 * np.exp(-0.5 * (branch / 0.14) ** 2)
    return field


def _boolean_relief(x, y, rng):
    field = np.zeros_like(x, dtype=np.float32)
    for _ in range(15):
        center_x, center_y = rng.uniform((-6.0, -4.2), (6.0, 4.2))
        radius_x, radius_y = rng.uniform(0.35, 1.5, size=2)
        stamp = (
            ((x - center_x) / radius_x) ** 2
            + ((y - center_y) / radius_y) ** 2
            <= 1.0
        )
        field[stamp] += float(rng.choice((-1.0, 1.0)))
    return field


def procedural_object_maps(
    object_name,
    config,
    rotation_deg,
    texture_frequency,
    visual_texture_frequency,
    offset_x_mm,
    offset_y_mm,
    seed,
):
    """Return deterministic public-demo height, albedo, and mask arrays."""
    if object_name not in PROCEDURAL_OBJECTS:
        raise ValueError(f"unknown procedural object: {object_name}")
    rng = np.random.default_rng(int(seed) + 17_000_003)
    xx, yy, sensor_mask = _sensor_grid(config)
    angle = math.radians(float(rotation_deg))
    shifted_x = xx - float(offset_x_mm)
    shifted_y = yy - float(offset_y_mm)
    x = math.cos(angle) * shifted_x + math.sin(angle) * shifted_y
    y = -math.sin(angle) * shifted_x + math.cos(angle) * shifted_y
    exponent = float(rng.uniform(2.0, 4.5))
    mask = (
        np.abs(x / float(rng.uniform(7.2, 9.0))) ** exponent
        + np.abs(y / float(rng.uniform(5.2, 7.0))) ** exponent
        <= 1.0
    ) & sensor_mask

    if object_name == "procedural_fourier_relief":
        detail = _fourier_field(x, y, rng, texture_frequency)
    elif object_name == "procedural_scratch_field":
        detail = _scratch_field(x, y, rng, texture_frequency)
    elif object_name == "procedural_thread_patch":
        detail = _thread_field(x, y, rng, texture_frequency)
    elif object_name == "procedural_voronoi_relief":
        detail = _voronoi_field(x, y, rng)
    elif object_name == "procedural_radial_lattice":
        detail = _radial_lattice(x, y, rng, texture_frequency)
    elif object_name == "procedural_branching_grooves":
        detail = _branching_grooves(x, y, rng, texture_frequency)
    elif object_name == "procedural_anisotropic_bands":
        detail = (
            np.cos(2.0 * np.pi * texture_frequency * (0.92 * x + 0.18 * y))
            + 0.45
            * np.cos(
                2.0
                * np.pi
                * 0.37
                * texture_frequency
                * (-0.25 * x + 0.97 * y)
            )
        )
    else:
        detail = _boolean_relief(x, y, rng)

    detail = _normalize(detail, mask)
    macro = 0.49 + 0.035 * x / 8.0 - 0.025 * y / 6.0
    height = np.clip(macro + 0.115 * detail, 0.18, 0.82).astype(np.float32)
    height[~mask] = 0.0

    optical = _normalize(
        _fourier_field(
            x,
            y,
            rng,
            max(0.2, float(visual_texture_frequency)),
            components=4,
        ),
        mask,
    )
    albedo = np.full_like(height, 0.20)
    albedo[mask] = np.clip(0.62 + 0.12 * optical[mask], 0.36, 0.84)
    return height, albedo.astype(np.float32), mask
