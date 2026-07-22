#!/usr/bin/env python3
import threading
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from moire_sim_platform import (
    evaluate,
    fit_model,
    load_config,
    load_dataset,
    load_model,
    save_dataset,
    save_model,
    simulate_contact_state,
    simulate_dataset,
)
from public_object_catalog import (
    PUBLIC_OBJECTS,
    object_category,
    public_object_label,
)
from public_r1_inference import (
    PHYSICS_IMAGE_SIZE,
    compute_demo_metrics,
    load_model_metadata,
    run_r1_inference,
    simulate_pressure_series,
)
from rigid_object_poc import OBJECT_TYPES, simulate_rigid_object_poc
from rigid_texture_diagnostics import (
    run_appearance_control_benchmark,
    run_frequency_response_benchmark,
)


ROOT = Path(__file__).resolve().parent
BASE_CONFIG = ROOT / "configs" / "simulation_baseline.json"
RIGID_POC_CONFIG = ROOT / "configs" / "pos_force_0.4n_calibrated.json"
RIGID_CACHE_SCHEMA_VERSION = 3
PUBLIC_R1_CACHE_SCHEMA_VERSION = 1
PUBLIC_R1_SIMULATION_LOCK = threading.Lock()
MODE_LABELS = {"point": "点接触平台", "rigid": "刚体接触 3D POC v8"}
OBJECT_LABELS = {
    "screwdriver": "螺丝刀",
    "satin": "缎面纹理刚性试片",
    "coin": "浮雕圆片",
    "thread_array": "多尺度螺纹阵列",
    "bump_array": "球形凸点阵列",
    "coin_relief": "硬币花纹浮雕",
    "knurled_ring": "菱形滚花圆环",
    "phillips_head": "十字槽螺丝头（held-out）",
    "gear_face": "齿轮端面（held-out）",
    "herringbone_plate": "人字纹硬质试片（held-out）",
    "microgroove_plate": "多频微槽试片（held-out）",
    "hex_socket_bolt": "内六角螺栓头（exploratory）",
    "spiral_groove_disk": "螺旋刻槽圆片（exploratory）",
    "woven_grid_plate": "编织网格硬质试片（exploratory）",
    "serrated_key": "锯齿钥匙（exploratory）",
}
PATTERN_LABELS = {
    "parallel": "Parallel fringes",
    "cross": "Cross grating",
    "hexagonal": "Six-direction / hexagonal grating",
}
PATTERN_SHORT_LABELS = {
    "parallel": "平行",
    "cross": "十字",
    "hexagonal": "六边形",
}


def build_sensor_figure(
    state, sensor_radius_mm, contact_x_mm, contact_y_mm, contact_radius_mm
):
    axis = np.linspace(
        -sensor_radius_mm,
        sensor_radius_mm,
        state["pre_image"].shape[0],
    )
    difference = state["difference"]
    difference_limit = max(
        1.0,
        float(np.percentile(np.abs(difference[state["sensor_mask"]]), 99)),
    )
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("接触前", "接触后", "有符号差分"),
        horizontal_spacing=0.045,
    )
    panels = (
        (state["pre_image"], "gray", 0, 255, "灰度"),
        (state["post_image"], "gray", 0, 255, "灰度"),
        (
            difference,
            "RdBu_r",
            -difference_limit,
            difference_limit,
            "差分",
        ),
    )
    for column, (image, colorscale, zmin, zmax, value_label) in enumerate(
        panels, start=1
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=image,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=False,
                hovertemplate=(
                    "x %{x:.2f} mm<br>y %{y:.2f} mm<br>"
                    + value_label
                    + " %{z:.0f}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        figure.add_shape(
            type="circle",
            x0=-sensor_radius_mm,
            x1=sensor_radius_mm,
            y0=-sensor_radius_mm,
            y1=sensor_radius_mm,
            line={"color": "rgba(31, 40, 35, 0.45)", "width": 1.5},
            row=1,
            col=column,
        )

    for column in (2, 3):
        figure.add_shape(
            type="circle",
            x0=contact_x_mm - contact_radius_mm,
            x1=contact_x_mm + contact_radius_mm,
            y0=contact_y_mm - contact_radius_mm,
            y1=contact_y_mm + contact_radius_mm,
            line={"color": "#E4572E", "width": 2},
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=[contact_x_mm],
                y=[contact_y_mm],
                mode="markers",
                marker={"color": "#E4572E", "size": 9, "symbol": "x"},
                showlegend=False,
                hovertemplate=(
                    "接触中心<br>x %{x:.2f} mm<br>y %{y:.2f} mm<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )

    for column in range(1, 4):
        anchor = "x" if column == 1 else f"x{column}"
        figure.update_xaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="x (mm)",
            showgrid=False,
            zeroline=False,
            constrain="domain",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="y (mm)" if column == 1 else None,
            showgrid=False,
            zeroline=False,
            scaleanchor=anchor,
            scaleratio=1,
            constrain="domain",
            showticklabels=column == 1,
            row=1,
            col=column,
        )
    figure.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#1F2823", "size": 13},
        uirevision="moire-contact-preview",
    )
    return figure


def build_rigid_observation_figure(state, sensor_radius_mm):
    axis = state["axis_mm"]
    camera_axis = state["camera_axis_mm"]
    detail = state["high_frequency_detail_mm"]
    detail_limit = max(
        1e-3,
        float(np.percentile(np.abs(detail[state["sensor_mask"]]), 99)),
    )
    height_limit = max(
        0.1,
        float(
            max(
                state["ground_truth_height_mm"].max(),
                state["membrane_height_mm"].max(),
                state["coarse_reconstructed_height_mm"].max(),
                state["reconstructed_height_mm"].max(),
            )
        ),
    )
    figure = make_subplots(
        rows=2,
        cols=4,
        subplot_titles=(
            "1 · GT rigid object",
            "2 · Membrane surface",
            "3 · Moiré-only reconstruction",
            "4 · Multiband reconstruction",
            "GT visual texture",
            "Shared raw frame",
            "Recovered appearance",
            "Carrier-aware high-frequency detail",
        ),
        horizontal_spacing=0.035,
        vertical_spacing=0.12,
    )
    panels = (
        (
            1,
            1,
            np.where(
                state["target_mask"], state["ground_truth_height_mm"], np.nan
            ),
            "Viridis",
            0,
            height_limit,
            "height (mm)",
        ),
        (
            1,
            2,
            state["membrane_height_mm"],
            "Viridis",
            0,
            height_limit,
            "height (mm)",
        ),
        (
            1,
            3,
            np.where(
                state["estimated_mask"],
                state["coarse_reconstructed_height_mm"],
                np.nan,
            ),
            "Viridis",
            0,
            height_limit,
            "height (mm)",
        ),
        (
            1,
            4,
            np.where(
                state["estimated_mask"],
                state["reconstructed_height_mm"],
                np.nan,
            ),
            "Viridis",
            0,
            height_limit,
            "height (mm)",
        ),
        (2, 1, state["ground_truth_albedo_image"], "gray", 0, 255, "intensity"),
        (2, 2, state["see_through_image"], "gray", 0, 255, "intensity"),
        (2, 3, state["recovered_appearance_image"], "gray", 0, 255, "intensity"),
        (2, 4, detail, "RdBu_r", -detail_limit, detail_limit, "height (mm)"),
    )
    for row, column, image, colorscale, zmin, zmax, label in panels:
        panel_axis = (
            camera_axis if np.asarray(image).shape[0] == len(camera_axis) else axis
        )
        figure.add_trace(
            go.Heatmap(
                x=panel_axis,
                y=panel_axis,
                z=image,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    f"x %{{x:.2f}} mm<br>y %{{y:.2f}} mm<br>"
                    f"{label} %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=row,
            col=column,
        )
        figure.add_shape(
            type="circle",
            x0=-sensor_radius_mm,
            x1=sensor_radius_mm,
            y0=-sensor_radius_mm,
            y1=sensor_radius_mm,
            line={"color": "rgba(31, 40, 35, 0.35)", "width": 1},
            row=row,
            col=column,
        )
        axis_index = (row - 1) * 4 + column
        anchor = "x" if axis_index == 1 else f"x{axis_index}"
        figure.update_xaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="x (mm)" if row == 2 else None,
            showgrid=False,
            constrain="domain",
            showticklabels=row == 2,
            row=row,
            col=column,
        )
        figure.update_yaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="y (mm)" if column == 1 else None,
            showgrid=False,
            scaleanchor=anchor,
            scaleratio=1,
            showticklabels=column == 1,
            row=row,
            col=column,
        )
    figure.update_layout(
        height=760,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="rigid-object-observation",
    )
    return figure


def build_rigid_moire_figure(state, sensor_radius_mm):
    axis = state["camera_axis_mm"]
    mask = state["camera_mask"]
    envelope_difference = state["moire_envelope_difference"]
    envelope_limit = max(
        1e-3,
        float(np.percentile(np.abs(envelope_difference[mask]), 99)),
    )
    raw_difference = state["moire_difference"]
    raw_limit = max(
        1.0,
        float(np.percentile(np.abs(raw_difference[mask]), 99)),
    )
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Reference fringe",
            "Deformed fringe",
            "Fringe Δ",
            "Raw-frame Δ",
        ),
        horizontal_spacing=0.035,
    )
    panels = (
        (
            state["reference_moire_envelope"],
            "gray",
            0.5,
            1.5,
            "envelope",
        ),
        (
            state["deformed_moire_envelope"],
            "gray",
            0.5,
            1.5,
            "envelope",
        ),
        (
            envelope_difference,
            "RdBu_r",
            -envelope_limit,
            envelope_limit,
            "envelope difference",
        ),
        (
            raw_difference,
            "RdBu_r",
            -raw_limit,
            raw_limit,
            "intensity difference",
        ),
    )
    for column, (image, colorscale, zmin, zmax, label) in enumerate(
        panels, start=1
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=np.where(mask, image, np.nan),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    f"x %{{x:.2f}} mm<br>y %{{y:.2f}} mm<br>"
                    f"{label} %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        anchor = "x" if column == 1 else f"x{column}"
        figure.update_xaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="x (mm)",
            showgrid=False,
            constrain="domain",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            range=[-sensor_radius_mm, sensor_radius_mm],
            title_text="y (mm)" if column == 1 else None,
            showgrid=False,
            scaleanchor=anchor,
            scaleratio=1,
            showticklabels=column == 1,
            row=1,
            col=column,
        )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="rigid-object-moire-deformation",
    )
    return figure


def build_rigid_surface_figure(state):
    axis = state["axis_mm"]
    truth = np.where(
        state["target_mask"], state["ground_truth_height_mm"], np.nan
    )
    reconstruction = np.where(
        state["estimated_mask"], state["reconstructed_height_mm"], np.nan
    )
    truth_appearance = np.where(
        state["target_mask"], state["ground_truth_appearance"], np.nan
    )
    recovered_appearance = np.where(
        state["estimated_mask"], state["recovered_appearance"], np.nan
    )
    height_limit = max(
        0.1,
        float(
            max(
                np.nanmax(truth),
                np.nanmax(reconstruction),
            )
        ),
    )
    figure = make_subplots(
        rows=1,
        cols=3,
        specs=[[
            {"type": "surface"},
            {"type": "surface"},
            {"type": "surface"},
        ]],
        subplot_titles=(
            "GT 几何 + 真实外观",
            "重建几何（高度着色）",
            "重建几何 + Recovered appearance",
        ),
        horizontal_spacing=0.03,
    )
    panels = (
        (truth, truth_appearance, "gray", 0.0, 1.0),
        (reconstruction, reconstruction, "Viridis", 0.0, height_limit),
        (reconstruction, recovered_appearance, "gray", 0.0, 1.0),
    )
    for column, (height, surface_color, colorscale, cmin, cmax) in enumerate(
        panels, start=1
    ):
        figure.add_trace(
            go.Surface(
                x=axis,
                y=axis,
                z=height,
                surfacecolor=surface_color,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
                customdata=surface_color,
                hovertemplate=(
                    "x %{x:.2f} mm<br>y %{y:.2f} mm<br>"
                    "z %{z:.3f} mm<br>surface %{customdata:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
    scene = {
        "xaxis_title": "x (mm)",
        "yaxis_title": "y (mm)",
        "zaxis_title": "height (mm)",
        "zaxis_range": [0.0, height_limit],
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35},
        "camera": {"eye": {"x": 1.35, "y": -1.35, "z": 0.9}},
    }
    figure.update_layout(
        height=520,
        scene=scene,
        scene2=scene,
        scene3=scene,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="rigid-object-surface",
    )
    return figure


def build_rigid_ablation_figure(state):
    metrics = state["metrics"]
    labels = (
        "Object → membrane texture",
        "Moiré-only geometry",
        "Recovered appearance",
        "Carrier + appearance geometry",
    )
    values = 100.0 * np.asarray(
        (
            metrics["object_to_membrane_texture_correlation"],
            metrics["coarse_texture_correlation"],
            metrics["recovered_appearance_texture_correlation"],
            metrics["texture_correlation"],
        )
    )
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=values,
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
            marker_color=("#9AA09C", "#9AA09C", "#2E7D5B", "#2E7D5B"),
            hovertemplate="%{x}<br>%{y:.2f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=280,
        yaxis_title="Transfer / correlation (%)",
        yaxis_range=[min(-10.0, float(values.min()) * 1.2), 110.0],
        margin={"l": 20, "r": 20, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision="rigid-object-ablation",
    )
    return figure


def build_rigid_confidence_figure(state):
    axis = state["axis_mm"]
    shift = state["reconstructed_apparent_shift_mm"]
    shift_magnitude = np.hypot(shift[0], shift[1])
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "Recovered carrier displacement",
            "Carrier confidence",
            "Appearance-geometry confidence",
        ),
        horizontal_spacing=0.04,
    )
    panels = (
        (shift_magnitude, "Magma", 0.0, None, "位移 mm"),
        (state["carrier_confidence"], "Viridis", 0.0, 1.0, "置信度"),
        (
            state["appearance_geometry_confidence"],
            "Viridis",
            0.0,
            1.0,
            "置信度",
        ),
    )
    for column, (image, colorscale, zmin, zmax, label) in enumerate(
        panels, start=1
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=np.where(state["sensor_mask"], image, np.nan),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    f"x %{{x:.2f}} mm<br>y %{{y:.2f}} mm<br>"
                    f"{label} %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        anchor = "x" if column == 1 else f"x{column}"
        figure.update_xaxes(title_text="x (mm)", row=1, col=column)
        figure.update_yaxes(
            title_text="y (mm)" if column == 1 else None,
            scaleanchor=anchor,
            scaleratio=1,
            showticklabels=column == 1,
            row=1,
            col=column,
        )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="rigid-object-confidence",
    )
    return figure


def build_public_r1_comparison_figure(state, output):
    axis = state["axis_mm"]
    truth = np.where(
        state["target_mask"],
        state["ground_truth_height_mm"],
        np.nan,
    )
    physics = np.where(
        output["physics_mask"],
        output["base_height_mm"],
        np.nan,
    )
    prediction = np.where(
        output["predicted_mask"],
        output["height_mm"],
        np.nan,
    )
    error = np.where(
        state["target_mask"],
        np.abs(output["display_height_mm"] - state["ground_truth_height_mm"]),
        np.nan,
    )
    height_limit = max(
        0.1,
        float(
            max(
                np.nanmax(truth),
                np.nanmax(physics),
                np.nanmax(prediction),
            )
        ),
    )
    error_limit = max(0.01, float(np.nanpercentile(error, 99)))
    figure = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=(
            "Synthetic GT",
            "Pure physics",
            "Frozen R1",
            "R1 absolute error",
        ),
        horizontal_spacing=0.035,
    )
    panels = (
        (truth, "Viridis", 0.0, height_limit, "height (mm)"),
        (physics, "Viridis", 0.0, height_limit, "height (mm)"),
        (prediction, "Viridis", 0.0, height_limit, "height (mm)"),
        (error, "Magma", 0.0, error_limit, "|error| (mm)"),
    )
    for column, (image, colorscale, zmin, zmax, label) in enumerate(
        panels,
        start=1,
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=image,
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    f"x %{{x:.2f}} mm<br>y %{{y:.2f}} mm<br>"
                    f"{label} %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        anchor = "x" if column == 1 else f"x{column}"
        figure.update_xaxes(
            title_text="x (mm)",
            constrain="domain",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text="y (mm)" if column == 1 else None,
            scaleanchor=anchor,
            scaleratio=1,
            showticklabels=column == 1,
            row=1,
            col=column,
        )
    figure.update_layout(
        height=390,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="public-r1-comparison",
    )
    return figure


def build_public_r1_surface_figure(state, output):
    axis = state["axis_mm"]
    truth = np.where(
        state["target_mask"],
        state["ground_truth_height_mm"],
        np.nan,
    )
    physics = np.where(
        output["physics_mask"],
        output["base_height_mm"],
        np.nan,
    )
    prediction = np.where(
        output["predicted_mask"],
        output["height_mm"],
        np.nan,
    )
    truth_appearance = np.where(
        state["target_mask"],
        state["ground_truth_appearance"],
        np.nan,
    )
    recovered_appearance = np.where(
        output["predicted_mask"],
        state["recovered_appearance"],
        np.nan,
    )
    height_limit = max(
        0.1,
        float(
            max(
                np.nanmax(truth),
                np.nanmax(physics),
                np.nanmax(prediction),
            )
        ),
    )
    figure = make_subplots(
        rows=1,
        cols=4,
        specs=[[
            {"type": "surface"},
            {"type": "surface"},
            {"type": "surface"},
            {"type": "surface"},
        ]],
        subplot_titles=(
            "GT + appearance",
            "Pure physics",
            "R1 geometry",
            "R1 + recovered appearance",
        ),
        horizontal_spacing=0.025,
    )
    panels = (
        (truth, truth_appearance, "gray", 0.0, 1.0),
        (physics, physics, "Viridis", 0.0, height_limit),
        (prediction, prediction, "Viridis", 0.0, height_limit),
        (prediction, recovered_appearance, "gray", 0.0, 1.0),
    )
    for column, (height, surface_color, colorscale, cmin, cmax) in enumerate(
        panels,
        start=1,
    ):
        figure.add_trace(
            go.Surface(
                x=axis,
                y=axis,
                z=height,
                surfacecolor=surface_color,
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                showscale=False,
                customdata=surface_color,
                hovertemplate=(
                    "x %{x:.2f} mm<br>y %{y:.2f} mm<br>"
                    "z %{z:.3f} mm<br>surface %{customdata:.3f}"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
    scene = {
        "xaxis_title": "x (mm)",
        "yaxis_title": "y (mm)",
        "zaxis_title": "height (mm)",
        "zaxis_range": [0.0, height_limit],
        "aspectmode": "manual",
        "aspectratio": {"x": 1.0, "y": 1.0, "z": 0.35},
        "camera": {"eye": {"x": 1.35, "y": -1.35, "z": 0.9}},
    }
    figure.update_layout(
        height=500,
        scene=scene,
        scene2=scene,
        scene3=scene,
        scene4=scene,
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        uirevision="public-r1-surfaces",
    )
    return figure


def build_public_r1_evidence_figure(state, output):
    axis = state["axis_mm"]
    uncertainty_limit = max(
        0.01,
        float(
            np.percentile(
                output["uncertainty_mm"][state["sensor_mask"]],
                99,
            )
        ),
    )
    figure = make_subplots(
        rows=1,
        cols=5,
        subplot_titles=(
            "Contact probability",
            "Uncertainty",
            "Physics gate",
            "Appearance gate",
            "Safety gate",
        ),
        horizontal_spacing=0.03,
    )
    panels = (
        (output["mask_probability"], "Viridis", 0.0, 1.0, "probability"),
        (
            output["uncertainty_mm"],
            "Magma",
            0.0,
            uncertainty_limit,
            "σ mm",
        ),
        (output["physics_gate"], "Viridis", 0.0, 1.0, "gate"),
        (output["appearance_gate"], "Viridis", 0.0, 1.0, "gate"),
        (output["safety_gate"], "Viridis", 0.0, 2.0, "gate"),
    )
    for column, (image, colorscale, zmin, zmax, label) in enumerate(
        panels,
        start=1,
    ):
        figure.add_trace(
            go.Heatmap(
                x=axis,
                y=axis,
                z=np.where(state["sensor_mask"], image, np.nan),
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                zsmooth=False,
                showscale=False,
                hoverongaps=False,
                hovertemplate=(
                    f"x %{{x:.2f}} mm<br>y %{{y:.2f}} mm<br>"
                    f"{label} %{{z:.3f}}<extra></extra>"
                ),
            ),
            row=1,
            col=column,
        )
        anchor = "x" if column == 1 else f"x{column}"
        figure.update_xaxes(
            title_text="x (mm)",
            constrain="domain",
            row=1,
            col=column,
        )
        figure.update_yaxes(
            title_text="y (mm)" if column == 1 else None,
            scaleanchor=anchor,
            scaleratio=1,
            showticklabels=column == 1,
            row=1,
            col=column,
        )
    figure.update_layout(
        height=350,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        uirevision="public-r1-evidence",
    )
    return figure


def build_frequency_response_figure(benchmark):
    rows = benchmark["rows"]
    frequencies = [row["frequency_cycles_per_mm"] for row in rows]
    stages = (
        ("membrane", "Object → membrane", "#8C6D31"),
        ("oracle_integrated", "Exact-shift oracle", "#6B7280"),
        ("coarse_inverse", "Moiré-only inverse", "#3B82F6"),
        ("carrier_fused", "Carrier + appearance", "#0F7B5C"),
    )
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cumulative amplitude transfer", "Texture correlation"),
        horizontal_spacing=0.10,
    )
    for key, label, color in stages:
        figure.add_trace(
            go.Scatter(
                x=frequencies,
                y=[row[f"{key}_amplitude_gain"] for row in rows],
                mode="lines+markers",
                name=label,
                legendgroup=key,
                line={"color": color},
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=frequencies,
                y=[row[f"{key}_correlation"] for row in rows],
                mode="lines+markers",
                name=label,
                legendgroup=key,
                showlegend=False,
                line={"color": color},
            ),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Spatial frequency (cycles/mm)")
    figure.update_yaxes(title_text="Amplitude / object", rangemode="tozero", row=1, col=1)
    figure.update_yaxes(title_text="Correlation", range=[-0.1, 1.05], row=1, col=2)
    figure.update_layout(
        height=410,
        margin={"l": 20, "r": 20, "t": 48, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"orientation": "h", "y": -0.22},
        uirevision="rigid-frequency-response",
    )
    return figure


def build_appearance_control_figure(benchmark):
    names = ("flat_print", "neutral_relief", "coupled_relief")
    labels = ("Flat + print", "Relief + neutral", "Relief + matched appearance")
    cases = benchmark["cases"]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Recovered high-frequency RMS", "Fusion confidence"),
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[cases[name]["high_frequency_detail_rms_mm"] for name in names],
            marker_color=("#9AA09C", "#3B82F6", "#0F7B5C"),
            name="detail RMS",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=labels,
            y=[
                cases[name]["appearance_geometry_confidence"] for name in names
            ],
            marker_color=("#9AA09C", "#3B82F6", "#0F7B5C"),
            name="confidence",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(title_text="RMS (mm)", rangemode="tozero", row=1, col=1)
    figure.update_yaxes(title_text="Mean confidence", range=[0.0, 1.0], row=1, col=2)
    figure.update_layout(
        height=380,
        margin={"l": 20, "r": 20, "t": 48, "b": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        uirevision="rigid-appearance-controls",
    )
    return figure


@st.cache_data(max_entries=6, show_spinner=False)
def _simulate_rigid_object_poc_cached(
    config, parameters, cache_schema_version
):
    return simulate_rigid_object_poc(config, **parameters)


@st.cache_resource(max_entries=1, show_spinner=False)
def _simulate_public_pressure_series_cached(
    config,
    parameters,
    cache_schema_version,
):
    with PUBLIC_R1_SIMULATION_LOCK:
        states = simulate_pressure_series(
            base_config=config,
            **parameters,
        )
    model_fields = (
        "pre_image",
        "post_image",
        "recovered_appearance",
        "carrier_only_height_mm",
        "reconstructed_height_mm",
        "reconstruction_confidence",
        "appearance_geometry_confidence",
        "carrier_confidence",
        "expected_error_mm",
        "estimated_mask",
    )
    anchor_index = len(states) // 2
    return [
        state
        if index == anchor_index
        else {name: state[name] for name in model_fields}
        for index, state in enumerate(states)
    ]


@st.cache_data(max_entries=4, show_spinner=False)
def _run_rigid_diagnostics_cached(config, parameters, cache_schema_version):
    diagnostic_parameters = dict(parameters)
    diagnostic_parameters.pop("object_type", None)
    return {
        "frequency": run_frequency_response_benchmark(
            config,
            slope_to_shift_mm=float(parameters["slope_to_shift_mm"]),
            noise_std=0.0,
            simulation_parameters=diagnostic_parameters,
        ),
        "appearance": run_appearance_control_benchmark(
            config,
            slope_to_shift_mm=float(parameters["slope_to_shift_mm"]),
            noise_std=0.0,
            simulation_parameters=diagnostic_parameters,
        ),
    }


def render_public_r1_demo(config):
    metadata = load_model_metadata()
    with st.sidebar:
        st.header("MoiréSkin · Frozen R1")
        st.caption(
            "Five-pressure membrane contact with multimodal Moiré/see-through "
            "reconstruction. Fixed 192² physics grid, 2× camera sampling, "
            "and CPU ONNX inference."
        )
        with st.form("public-r1-controls"):
            object_name = st.selectbox(
                "Contact object",
                PUBLIC_OBJECTS,
                index=PUBLIC_OBJECTS.index("coin_relief"),
                format_func=public_object_label,
            )
            pattern = st.selectbox(
                "Grating pattern",
                tuple(PATTERN_LABELS),
                index=tuple(PATTERN_LABELS).index("cross"),
                format_func=PATTERN_LABELS.get,
            )
            pressure_frame_count = st.select_slider(
                "Pressure frames used for reconstruction",
                options=(1, 2, 3, 5),
                value=5,
            )
            indentation = st.slider(
                "Indentation depth (mm)",
                0.52,
                1.18,
                0.85,
                0.03,
            )
            rotation = st.slider(
                "In-plane rotation (deg)",
                -90.0,
                90.0,
                20.0,
                5.0,
            )
            offset_x = st.slider(
                "Object position x (mm)",
                -2.0,
                2.0,
                0.0,
                0.25,
            )
            offset_y = st.slider(
                "Object position y (mm)",
                -2.0,
                2.0,
                0.0,
                0.25,
            )
            with st.expander("Surface and imaging", expanded=True):
                texture_frequency = st.slider(
                    "Geometric texture frequency (cycles/mm)",
                    0.20,
                    2.60,
                    1.20,
                    0.05,
                )
                visual_texture_frequency = st.slider(
                    "Visual texture frequency (cycles/mm)",
                    0.20,
                    2.60,
                    0.70,
                    0.05,
                )
                relief_scale = st.slider(
                    "Surface relief scale",
                    0.72,
                    1.28,
                    1.00,
                    0.04,
                )
                camera_psf_sigma = st.slider(
                    "Camera PSF σ (px)",
                    0.20,
                    0.72,
                    0.45,
                    0.01,
                )
                grating_open_fraction = st.slider(
                    "Single-direction grating open fraction",
                    0.73,
                    0.90,
                    0.82,
                    0.01,
                )
                grating_line_transmittance = st.slider(
                    "Grating-line transmittance",
                    0.05,
                    0.16,
                    0.10,
                    0.01,
                )
                noise_std = st.slider(
                    "Readout noise σ",
                    0.0015,
                    0.0120,
                    0.0040,
                    0.0005,
                    format="%.4f",
                )
            with st.expander("Membrane and reproducibility"):
                base_tension = st.slider(
                    "0 kPa membrane tension (N/mm)",
                    0.064,
                    0.096,
                    0.080,
                    0.002,
                    format="%.3f",
                )
                seed = int(
                    st.number_input(
                        "Random seed",
                        min_value=0,
                        value=7,
                        step=1,
                    )
                )
            st.form_submit_button(
                "Run five-pressure simulation + R1",
                type="primary",
                use_container_width=True,
            )
        st.caption(
            "This public demo does not generate training data, train models, "
            "accept file paths, or contain sealed-test samples."
        )

    parameters = {
        "object_name": object_name,
        "pattern": pattern,
        "rotation_deg": rotation,
        "texture_frequency": texture_frequency,
        "visual_texture_frequency": visual_texture_frequency,
        "relief_scale": relief_scale,
        "indentation_mm": indentation,
        "offset_x_mm": offset_x,
        "offset_y_mm": offset_y,
        "base_tension_n_per_mm": base_tension,
        "camera_psf_sigma": camera_psf_sigma,
        "grating_open_fraction": grating_open_fraction,
        "grating_line_transmittance": grating_line_transmittance,
        "noise_std": noise_std,
        "seed": seed,
    }
    geometry_key = repr(
        (
            object_name,
            rotation,
            texture_frequency,
            relief_scale,
            indentation,
            offset_x,
            offset_y,
            seed,
        )
    )
    with st.spinner(
        "Generating 0/2/4/6/8 kPa observations and running Frozen R1…"
    ):
        states = _simulate_public_pressure_series_cached(
            config,
            parameters,
            PUBLIC_R1_CACHE_SCHEMA_VERSION,
        )
        output = run_r1_inference(
            states,
            pressure_frame_count,
            geometry_key,
        )

    anchor = states[2]
    metrics = output["metrics"]
    physics_metrics = output["physics_metrics"]
    poc_metrics = compute_demo_metrics(
        anchor,
        anchor["reconstructed_height_mm"],
        anchor["estimated_mask"],
    )
    iou_delta = metrics["mask_iou"] - physics_metrics["mask_iou"]
    height_delta_pp = 100.0 * (
        metrics["height_nrmse"] - physics_metrics["height_nrmse"]
    )
    texture_delta = (
        metrics["texture_correlation"]
        - physics_metrics["texture_correlation"]
    )
    normal_delta = (
        metrics["normal_error_deg"]
        - physics_metrics["normal_error_deg"]
    )
    category = object_category(object_name)
    category_labels = {
        "trained named family": "named training family",
        "held-out named family": "named held-out family",
        "exploratory named family": "named exploratory family",
        "procedural OOD": "procedural OOD",
    }

    st.title("Moiré + See-through · Frozen R1 Reconstruction")
    st.caption(
        f"{public_object_label(object_name)} · "
        f"{PATTERN_LABELS[pattern]} · "
        f"{pressure_frame_count} pressure frame(s) · "
        f"{category_labels[category]}"
    )
    if category == "procedural OOD":
        st.info(
            "This deterministic random surface is absent from the training set "
            "and stress-tests generalization. These values are live diagnostics "
            "for this synthetic sample, not sealed-test results."
        )

    summary = st.columns(4)
    summary[0].metric(
        "R1 Height NRMSE",
        f"{100.0 * metrics['height_nrmse']:.1f}%",
        f"{height_delta_pp:+.1f} pp vs physics",
        delta_color="inverse",
    )
    summary[1].metric(
        "R1 Texture Correlation r",
        f"{metrics['texture_correlation']:.3f}",
        f"{texture_delta:+.3f} vs physics",
    )
    summary[2].metric(
        "R1 Contact IoU",
        f"{metrics['mask_iou']:.3f}",
        (
            "≈0.000 vs physics"
            if abs(iou_delta) < 0.0005
            else f"{iou_delta:+.3f} vs physics"
        ),
        delta_color="off" if abs(iou_delta) < 0.0005 else "normal",
    )
    summary[3].metric(
        "R1 Normal Error",
        f"{metrics['normal_error_deg']:.1f}°",
        f"{normal_delta:+.1f}° vs physics",
        delta_color="inverse",
    )
    st.caption(
        f"Prediction uncertainty p95 {metrics['uncertainty_p95_mm']:.3f} mm · "
        "Height NRMSE and texture correlation use the Frozen evaluator's GT "
        "contact support. Online synthetic GT is shown only for interactive "
        "diagnostics."
    )

    st.subheader("Fair Comparison on the Same Synthetic Contact")
    st.plotly_chart(
        build_public_r1_comparison_figure(anchor, output),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.markdown(
        "| System | Height NRMSE ↓ | Texture r ↑ | IoU ↑ | Normal error ↓ |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| Pure physics | {100 * physics_metrics['height_nrmse']:.1f}% | "
        f"{physics_metrics['texture_correlation']:.3f} | "
        f"{physics_metrics['mask_iou']:.3f} | "
        f"{physics_metrics['normal_error_deg']:.1f}° |\n"
        f"| Analytic POC v8 @ 4 kPa | {100 * poc_metrics['height_nrmse']:.1f}% | "
        f"{poc_metrics['texture_correlation']:.3f} | "
        f"{poc_metrics['mask_iou']:.3f} | "
        f"{poc_metrics['normal_error_deg']:.1f}° |\n"
        f"| **Frozen R1** | **{100 * metrics['height_nrmse']:.1f}%** | "
        f"**{metrics['texture_correlation']:.3f}** | "
        f"**{metrics['mask_iou']:.3f}** | "
        f"**{metrics['normal_error_deg']:.1f}°** |"
    )

    st.subheader("Progressive Pressure Observations")
    selected_columns = st.columns(
        len(output["selected_pressure_indices"])
    )
    for column, pressure_index, pressure_kpa in zip(
        selected_columns,
        output["selected_pressure_indices"],
        output["selected_pressures_kpa"],
    ):
        column.image(
            np.flipud(states[pressure_index]["post_image"]),
            caption=f"{pressure_kpa:.0f} kPa",
            use_column_width=True,
            clamp=True,
        )
    st.caption(
        "The display uses a consistent y-axis orientation; the model still "
        "receives the original simulation arrays. The 1/2/3-frame modes only "
        "change pressure validity; they do not select or tune a different model."
    )

    st.subheader("Deformed Moiré at the 4 kPa Anchor")
    st.plotly_chart(
        build_rigid_moire_figure(
            anchor,
            float(config["sensor_radius_mm"]),
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )
    with st.expander("View the Full Moiré / See-through / Appearance Pipeline"):
        st.plotly_chart(
            build_rigid_observation_figure(
                anchor,
                float(config["sensor_radius_mm"]),
            ),
            use_container_width=True,
            config={"displaylogo": False},
        )

    st.subheader("Local 2.5D Surface and Appearance")
    st.plotly_chart(
        build_public_r1_surface_figure(anchor, output),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "Recovered appearance is used only as a color map on the R1 surface; "
        "visual patterns are not unconditionally converted into height. Geometry "
        "is determined jointly by multi-pressure raw images, the pure-physics "
        "initial estimate, and a confidence-gated residual."
    )

    st.subheader("Fusion and Uncertainty")
    st.plotly_chart(
        build_public_r1_evidence_figure(anchor, output),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.caption(
        "The physics, appearance, and safety gates are actual Frozen R1 outputs. "
        "They show how low-frequency physics, high-frequency appearance, and "
        "safe fallback divide responsibility across the surface."
    )

    source = metadata["source"]
    st.divider()
    st.caption(
        f"Frozen {source['candidate_id']} · "
        f"{source['model_variant']} · width {source['width']} · "
        f"ONNX {metadata['artifact_sha256'][:12]}… · "
        f"{PHYSICS_IMAGE_SIZE}² physics / 2× camera · "
        "sealed test was not exported into or read by this online app · "
        "synthetic local 2.5D proof of concept, not a real-sensor accuracy claim."
    )


def render_rigid_poc(config, public_demo=False):
    with st.sidebar:
        st.header("刚体接触 3D POC v8")
        st.caption(
            "低频 Moiré 相位与多尺度载波逆解联合恢复局部几何；"
            "held-out 与 exploratory 形状用于检查跨几何族表现。"
        )
        if public_demo:
            st.caption("公共只读 Demo：点击运行后才会计算，不写入服务器文件。")
        controls = (
            st.form("public-rigid-controls") if public_demo else st.container()
        )
        with controls:
            object_type = st.selectbox(
                "接触物体", OBJECT_TYPES, format_func=OBJECT_LABELS.get
            )
            indentation = st.slider(
                "压入深度 (mm)", 0.35, 1.20, 0.85, 0.05
            )
            rotation = st.slider(
                "平面旋转 (deg)", -90.0, 90.0, 20.0, 5.0
            )
            offset_x = st.slider(
                "物体位置 x (mm)", -2.0, 2.0, 0.0, 0.25
            )
            offset_y = st.slider(
                "物体位置 y (mm)", -2.0, 2.0, 0.0, 0.25
            )
            with st.expander("表面与成像", expanded=True):
                texture_frequency = st.slider(
                    "几何纹理频率 (cycles/mm)", 0.15, 2.20, 1.20, 0.05
                )
                visual_texture_frequency = st.slider(
                    "视觉纹理频率 (cycles/mm)", 0.15, 1.80, 0.70, 0.05
                )
                relief_scale = st.slider(
                    "表面起伏尺度", 0.5, 1.5, 1.0, 0.05
                )
                camera_psf_sigma = st.slider(
                    "相机 PSF σ (px)", 0.0, 1.5, 0.45, 0.05
                )
                camera_supersample = st.select_slider(
                    "相机采样倍率",
                    options=(1, 2) if public_demo else (1, 2, 3),
                    value=2,
                )
                grating_open_fraction = st.slider(
                    "单方向光栅开口率", 0.65, 0.95, 0.82, 0.01
                )
                grating_line_transmittance = st.slider(
                    "光栅线透射率", 0.0, 0.40, 0.10, 0.02
                )
                noise_std = st.slider(
                    "读出噪声 σ", 0.0, 0.02, 0.004, 0.001
                )
            with st.expander("物理与数值"):
                inflation_pressure = st.slider(
                    "充气压差 (kPa)", 0.0, 10.0, 4.0, 0.5
                )
                membrane_tension = st.slider(
                    "膜张力 (N/mm)", 0.03, 0.15, 0.08, 0.01
                )
                if public_demo:
                    membrane_bending_stiffness = 1e-4
                    cavity_depth = 8.0
                    sealed_air_coupling = True
                else:
                    membrane_bending_stiffness = st.select_slider(
                        "膜弯曲刚度 D (N·mm)",
                        options=(0.0, 2.5e-5, 5e-5, 1e-4, 2e-4, 5e-4),
                        value=1e-4,
                        format_func=lambda value: f"{value:.1e}",
                    )
                    sealed_air_coupling = st.checkbox(
                        "密闭腔体压力-体积耦合", value=True
                    )
                    cavity_depth = st.slider(
                        "等效腔体深度 (mm)",
                        3.0,
                        20.0,
                        8.0,
                        0.5,
                        disabled=not sealed_air_coupling,
                    )
                slope_to_shift = st.slider(
                    "斜率-光栅位移标定 (mm)", 0.06, 0.18, 0.12, 0.01
                )
                resolution = st.select_slider(
                    "物理网格分辨率",
                    options=(128, 160, 224)
                    if public_demo
                    else (96, 128, 160, 224, 256),
                    value=224,
                )
                seed = int(
                    st.number_input("随机种子", min_value=0, value=7, step=1)
                )
            if public_demo:
                st.form_submit_button(
                    "运行模拟", type="primary", use_container_width=True
                )

    simulation_config = dict(config, image_size=resolution)
    simulation_parameters = {
        "object_type": object_type,
        "rotation_deg": rotation,
        "texture_frequency": texture_frequency,
        "visual_texture_frequency": visual_texture_frequency,
        "relief_scale": relief_scale,
        "indentation_mm": indentation,
        "offset_x_mm": offset_x,
        "offset_y_mm": offset_y,
        "membrane_tension_n_per_mm": membrane_tension,
        "inflation_pressure_kpa": inflation_pressure,
        "membrane_bending_stiffness_n_mm": membrane_bending_stiffness,
        "cavity_depth_mm": cavity_depth,
        "sealed_air_coupling": sealed_air_coupling,
        "camera_psf_sigma": camera_psf_sigma,
        "camera_supersample": camera_supersample,
        "grating_open_fraction": grating_open_fraction,
        "grating_line_transmittance": grating_line_transmittance,
        "noise_std": noise_std,
        "slope_to_shift_mm": slope_to_shift,
        "seed": seed,
    }
    if public_demo:
        with st.spinner("正在计算膜接触、光学观测与 3D 重建…"):
            state = _simulate_rigid_object_poc_cached(
                simulation_config,
                simulation_parameters,
                RIGID_CACHE_SCHEMA_VERSION,
            )
    else:
        state = simulate_rigid_object_poc(
            simulation_config, **simulation_parameters
        )
    metrics = state["metrics"]
    physics = state["physics"]
    st.title("Moiré + See-through 刚体接触 3D POC v8")
    st.caption(
        "前向：含张力、弯曲刚度与密闭腔体 P-V 耦合的膜接触 → "
        "显式光栅遮挡与 Moiré 调制 → 单一相机原始帧。"
        "逆向：Moiré 低频形状 + 多尺度载波相位/LK + 置信度门控的 See-through 外观。"
    )
    summary = st.columns(3)
    summary[0].metric("深度 NRMSE", f"{100 * metrics['height_nrmse']:.1f}%")
    summary[1].metric(
        "Moiré-only 纹理 r",
        f"{metrics['coarse_texture_correlation']:.2f}",
    )
    summary[2].metric(
        "多频融合纹理 r",
        f"{metrics['texture_correlation']:.2f}",
    )
    if not state["high_frequency_enabled"]:
        st.warning(
            "当前相机采样低于每个光栅周期 4 px，高频载波逆解已自动关闭。"
        )
    st.caption(
        "第一排比较真实几何、膜面、Moiré-only 与多频融合；"
        "第二排检查真实外观、共享原始帧、恢复外观与高频几何增量。"
    )
    st.subheader("形变下的 Moiré 条纹")
    st.caption(
        "Reference 与 Deformed 显示去除物体外观后的低频 Moiré 包络；"
        "Envelope Δ 显示膜形变造成的条纹弯曲，Raw-frame Δ 则保留光栅遮挡与外观变化。"
    )
    st.plotly_chart(
        build_rigid_moire_figure(
            state, float(config["sensor_radius_mm"])
        ),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.plotly_chart(
        build_rigid_observation_figure(state, float(config["sensor_radius_mm"])),
        use_container_width=True,
        config={"displaylogo": False},
    )
    st.plotly_chart(
        build_rigid_surface_figure(state),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "3D 左图把真实外观贴到 GT 几何；中图只用高度着色检查几何；"
        "右图把 Recovered appearance 映射到同一重建表面。"
        "外观贴图改变颜色，不会把印刷图案强行变成高度。"
    )
    st.plotly_chart(
        build_rigid_ablation_figure(state),
        use_container_width=True,
        config={"displaylogo": False},
    )
    if not public_demo:
        st.subheader("逆解置信度")
        st.caption(
            "外观只在载波位移可信、且局部外观结构与几何结构一致时参与细节融合。"
        )
        st.plotly_chart(
            build_rigid_confidence_figure(state),
            use_container_width=True,
            config={"displaylogo": False},
        )
    st.caption(
        f"轮廓 IoU {metrics['mask_iou']:.3f} · "
        f"法向误差 {metrics['normal_error_deg']:.1f}° · "
        f"Chamfer {metrics['chamfer_mm']:.3f} mm · "
        f"接触覆盖 {100 * physics['contact_fraction']:.1f}% · "
        f"可用开口 {100 * metrics['clear_aperture_fraction']:.1f}% · "
        f"平均透射 {100 * metrics['mean_grating_transmission']:.1f}% · "
        f"采样 {state['camera_pixels_per_grating_pitch']:.2f} px/pitch · "
        "外观-几何对齐 "
        f"{state['high_frequency_diagnostics']['appearance_geometry_alignment']:.2f} · "
        "外观→几何权重 "
        f"{state['high_frequency_diagnostics']['appearance_geometry_confidence_mean']:.2f} · "
        "载波置信度 "
        f"{state['high_frequency_diagnostics']['carrier_confidence_mean']:.2f} · "
        f"纹理 NRMSE {metrics['texture_nrmse']:.2f} · "
        f"端到端纹理 r {metrics['texture_correlation']:.2f} · "
        "物体→膜 "
        f"r {metrics['object_to_membrane_texture_correlation']:.2f} · "
        "膜→重建 "
        f"r {metrics['membrane_to_reconstruction_texture_correlation']:.2f} · "
        f"接触求解 {physics['iterations']} + {physics['bending_iterations']} iterations · "
        f"弯曲收敛 {'是' if physics['bending_converged'] else '否'} · "
        f"有效压差 {physics['effective_pressure_kpa']:.2f} kPa · "
        f"腔体体积变化 {100 * physics['sealed_air_volume_change_fraction']:.2f}%。"
    )
    if not public_demo:
        st.divider()
        st.subheader("本地研究诊断：频响与外观反事实")
        st.caption(
            "Oracle 仅用真值位移计算理论上限，不参与实际重建。"
            "三组外观对照分别检查平面印刷误报、无纹理浮雕和外观-几何一致浮雕。"
        )
        diagnostic_signature = repr((simulation_config, simulation_parameters))
        if st.button("运行本地诊断", use_container_width=True):
            with st.spinner("正在运行频率扫描和外观反事实…"):
                st.session_state["rigid_research_diagnostics"] = (
                    _run_rigid_diagnostics_cached(
                        simulation_config,
                        simulation_parameters,
                        RIGID_CACHE_SCHEMA_VERSION,
                    )
                )
                st.session_state["rigid_research_diagnostics_signature"] = (
                    diagnostic_signature
                )
        diagnostics = None
        if (
            st.session_state.get("rigid_research_diagnostics_signature")
            == diagnostic_signature
        ):
            diagnostics = st.session_state.get("rigid_research_diagnostics")
        if diagnostics is not None:
            st.plotly_chart(
                build_frequency_response_figure(diagnostics["frequency"]),
                use_container_width=True,
                config={"displaylogo": False},
            )
            st.plotly_chart(
                build_appearance_control_figure(diagnostics["appearance"]),
                use_container_width=True,
                config={"displaylogo": False},
            )
            flat = diagnostics["appearance"]["cases"]["flat_print"]
            coupled = diagnostics["appearance"]["cases"]["coupled_relief"]
            st.caption(
                f"平面印刷伪浮雕 RMS {flat['high_frequency_detail_rms_mm']:.4f} mm · "
                f"平面外观-几何置信度 {flat['appearance_geometry_confidence']:.3f} · "
                f"一致浮雕纹理 r {coupled['texture_correlation']:.3f}。"
            )
    st.caption(
        "输出是单次触碰可观测的局部 2.5D 点云；"
        "缎面仍作为带纹理的刚性试片，不模拟织物柔性。"
    )


def _resolve_run_directory(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def render_app():
    st.set_page_config(
        page_title="MoiréSkin Simulation Lab",
        page_icon="◉",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        .block-container {max-width: 1480px; padding-top: 1.6rem; padding-bottom: 3rem;}
        h1 {font-size: 2.25rem !important; letter-spacing: -0.04em;}
        [data-testid="stSidebar"] {border-right: 1px solid rgba(31, 40, 35, 0.12);}
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(31, 40, 35, 0.12);
            border-radius: 10px;
            padding: 0.8rem 1rem;
        }
        [data-testid="stPlotlyChart"] {margin-top: 0.25rem;}
        @media (max-width: 1000px) {
            div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"]) {
                flex-wrap: wrap;
            }
            div[data-testid="stHorizontalBlock"]:has([data-testid="stMetric"])
            > div[data-testid="column"] {
                flex: 1 1 calc(50% - 0.5rem) !important;
                width: calc(50% - 0.5rem) !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    mode = st.sidebar.radio(
        "实验模式", tuple(MODE_LABELS), format_func=MODE_LABELS.get
    )
    if mode == "rigid":
        render_rigid_poc(load_config(RIGID_POC_CONFIG))
        return

    base_config = load_config(BASE_CONFIG)
    sensor_radius = float(base_config["sensor_radius_mm"])

    with st.sidebar:
        st.header("实验控制")
        st.caption("改变任一参数，主实验台会立即重算光学观测。")
        pattern_options = tuple(PATTERN_LABELS)
        pattern = st.selectbox(
            "底层光栅 pattern",
            pattern_options,
            index=pattern_options.index(base_config["pattern"]),
            format_func=PATTERN_LABELS.get,
        )
        pressure = st.slider(
            "腔体压力 (kPa)",
            float(base_config["pressure_kpa"][0]),
            float(base_config["pressure_kpa"][1]),
            2.5,
            0.1,
        )
        force = st.slider(
            "法向力 (N)",
            float(base_config["normal_force_n"][0]),
            float(base_config["normal_force_n"][1]),
            1.2,
            0.1,
        )
        contact_radius = st.slider(
            "接触半径 (mm)",
            float(base_config["contact_radius_mm"][0]),
            float(base_config["contact_radius_mm"][1]),
            2.4,
            0.1,
        )
        contact_limit = round(0.65 * sensor_radius, 1)
        contact_x = st.slider(
            "接触位置 x (mm)", -contact_limit, contact_limit, 2.0, 0.1
        )
        contact_y = st.slider(
            "接触位置 y (mm)", -contact_limit, contact_limit, -3.0, 0.1
        )
        st.divider()
        st.caption("观测设置")
        resolution = st.select_slider(
            "预览 / 数据分辨率", options=(64, 96, 128), value=96
        )
        noise_std = st.slider("读出噪声 σ", 0.0, 0.05, 0.012, 0.001)
        seed = int(st.number_input("随机种子", min_value=0, value=7, step=1))

    preview_config = dict(base_config)
    preview_config.update(
        {
            "pattern": pattern,
            "image_size": resolution,
            "noise_std": [noise_std, noise_std],
        }
    )
    state = simulate_contact_state(
        preview_config,
        contact_x_mm=contact_x,
        contact_y_mm=contact_y,
        normal_force_n=force,
        pressure_kpa=pressure,
        contact_radius_mm=contact_radius,
        seed=seed,
    )

    st.title("MoiréSkin Simulation Lab")
    st.caption("单接触状态 → 光学观测 → 模拟数据 → 训练与测试")
    metrics = st.columns(4)
    metrics[0].metric("光栅 pattern", PATTERN_SHORT_LABELS[pattern])
    metrics[1].metric("接触中心 (mm)", f"({contact_x:+.1f}, {contact_y:+.1f})")
    metrics[2].metric("法向力", f"{force:.1f} N")
    metrics[3].metric("压力", f"{pressure:.1f} kPa")

    st.plotly_chart(
        build_sensor_figure(
            state,
            sensor_radius,
            contact_x,
            contact_y,
            contact_radius,
        ),
        use_container_width=True,
        config={"displaylogo": False, "scrollZoom": True},
    )
    st.caption(
        "橙色圆为接触区域，叉号为接触中心。当前后端是可校准的准静态相位模型，用于打通平台；不是 FEM 精度声明。"
    )

    st.divider()
    st.subheader("模拟数据 → 训练 → 测试")
    st.caption(
        "数据生成使用基线配置中的压力、力、接触半径与光学随机范围；采用当前 pattern 和分辨率。"
    )
    inputs = st.columns((1, 1, 2))
    train_samples = int(
        inputs[0].number_input(
            "训练样本数", min_value=10, value=1000, step=100
        )
    )
    test_samples = int(
        inputs[1].number_input("测试样本数", min_value=10, value=200, step=50)
    )
    output_value = inputs[2].text_input(
        "输出目录", "simulation_runs/interactive"
    )
    run_directory = _resolve_run_directory(output_value)
    train_path = run_directory / "train.npz"
    test_path = run_directory / "test.npz"
    model_path = run_directory / "model.npz"

    actions = st.columns(3)
    generate_clicked = actions[0].button(
        "1  生成数据", type="primary", use_container_width=True
    )
    train_clicked = actions[1].button("2  训练基线", use_container_width=True)
    test_clicked = actions[2].button("3  测试模型", use_container_width=True)

    dataset_config = dict(base_config)
    dataset_config.update({"pattern": pattern, "image_size": resolution})
    try:
        if generate_clicked:
            with st.spinner("正在生成模拟数据…"):
                save_dataset(
                    train_path,
                    simulate_dataset(dataset_config, train_samples, seed),
                    dataset_config,
                    domain="sim",
                )
                save_dataset(
                    test_path,
                    simulate_dataset(dataset_config, test_samples, seed + 1),
                    dataset_config,
                    domain="sim",
                )
            st.session_state.pop("last_metrics", None)
            st.success(f"数据已写入 {run_directory.resolve()}")

        if train_clicked:
            with st.spinner("正在训练基线模型…"):
                save_model(model_path, fit_model([load_dataset(train_path)]))
            st.session_state.pop("last_metrics", None)
            st.success(f"模型已写入 {model_path.resolve()}")

        if test_clicked:
            with st.spinner("正在测试…"):
                st.session_state["last_metrics"] = evaluate(
                    load_model(model_path), load_dataset(test_path)
                )
                st.session_state["metrics_path"] = str(run_directory.resolve())
    except (FileNotFoundError, OSError, ValueError, np.linalg.LinAlgError) as exc:
        st.error(str(exc))

    if "last_metrics" in st.session_state:
        results = st.session_state["last_metrics"]
        st.caption(
            f"测试集 {results['samples']} 个样本 · {st.session_state['metrics_path']}"
        )
        result_columns = st.columns(3)
        result_columns[0].metric(
            "x MAE", f"{results['mae']['contact_x_mm']:.3f} mm"
        )
        result_columns[1].metric(
            "y MAE", f"{results['mae']['contact_y_mm']:.3f} mm"
        )
        result_columns[2].metric(
            "力 MAE", f"{results['mae']['normal_force_n']:.3f} N"
        )

    with st.expander("真实数据接口（已预留）"):
        st.write(
            "真实数据只需提供成对的接触前/后图像、接触位置、法向力和压力。"
            "`pack-real` 会转换成与模拟数据完全相同的 `.npz` schema，之后可直接混合训练或独立测试。"
        )
        st.code(
            "python3 moire_sim_platform.py pack-real --manifest real_data/manifest.csv "
            "--config configs/simulation_baseline.json --output real_data/packed.npz"
        )


if __name__ == "__main__":
    render_app()
