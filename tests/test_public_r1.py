import unittest

import numpy as np

from moire_sim_platform import load_config
from public_object_catalog import PROCEDURAL_OBJECTS, procedural_object_maps
from public_r1_inference import (
    CALIBRATION_PATH,
    FRAME_COUNTS,
    MODEL_PATH,
    OUTPUT_NAMES,
    load_model_metadata,
    load_onnx_session,
    pressure_subset_indices,
    run_r1_inference,
    simulate_pressure_series,
)
from rigid_object_poc import simulate_rigid_object_poc
from rigid_texture_calibration import load_calibration_bundle


class PublicR1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/pos_force_0.4n_calibrated.json")

    def test_release_metadata_and_onnx_contract(self):
        metadata = load_model_metadata()
        self.assertTrue(MODEL_PATH.is_file())
        self.assertEqual(metadata["frame_counts"], list(FRAME_COUNTS))
        self.assertFalse(metadata["test_policy"]["online_demo_uses_sealed_test"])
        session = load_onnx_session()
        self.assertEqual(
            [output.name for output in session.get_outputs()],
            list(OUTPUT_NAMES),
        )

    def test_pressure_subsets_are_geometry_stable(self):
        for frame_count in FRAME_COUNTS:
            first = pressure_subset_indices(frame_count, "geometry-17")
            second = pressure_subset_indices(frame_count, "geometry-17")
            self.assertEqual(first, second)
            self.assertEqual(len(first), frame_count)
            self.assertIn(2, first)

    def test_procedural_catalog_is_finite_and_nonempty(self):
        config = dict(self.config, image_size=64)
        for object_name in PROCEDURAL_OBJECTS:
            with self.subTest(object_name=object_name):
                height, albedo, mask = procedural_object_maps(
                    object_name,
                    config,
                    23.0,
                    1.2,
                    0.7,
                    0.5,
                    -0.25,
                    7,
                )
                self.assertEqual(height.shape, (64, 64))
                self.assertTrue(np.isfinite(height).all())
                self.assertTrue(np.isfinite(albedo).all())
                self.assertTrue(mask.any())
                self.assertTrue((~mask).any())
                self.assertTrue(np.all(height[~mask] == 0.0))

    def test_all_public_grating_patterns_simulate(self):
        calibration = load_calibration_bundle(CALIBRATION_PATH)
        grating = calibration["grating"]
        camera = calibration["camera"]
        config = dict(
            self.config,
            image_size=48,
            grating_pitch_mm=float(grating["pitch_mm"]),
            grating_angle_a_deg=float(grating["angle_a_deg"]),
            grating_angle_b_deg=float(grating["angle_b_deg"]),
            brightness_gain=list(camera["brightness_gain"]),
            brightness_offset=list(camera["brightness_offset"]),
            radial_distortion_k1=float(camera["radial_distortion_k1"]),
        )
        for pattern in ("parallel", "cross", "hexagonal"):
            with self.subTest(pattern=pattern):
                state = simulate_rigid_object_poc(
                    config,
                    object_type="coin",
                    pattern=pattern,
                    camera_supersample=1,
                    noise_std=0.0,
                    seed=3,
                )
                self.assertEqual(state["pre_image"].shape, (48, 48))
                self.assertTrue(np.isfinite(state["reconstructed_height_mm"]).all())

    def test_frozen_r1_end_to_end_improves_public_physics_baseline(self):
        states = simulate_pressure_series(
            self.config,
            "coin_relief",
            "parallel",
            20.0,
            1.2,
            0.7,
            1.0,
            0.85,
            0.0,
            0.0,
            0.08,
            0.45,
            0.82,
            0.10,
            0.004,
            7,
        )
        outputs = {
            frame_count: run_r1_inference(
                states,
                frame_count,
                "coin-relief-smoke",
            )
            for frame_count in FRAME_COUNTS
        }
        for frame_count, output in outputs.items():
            with self.subTest(frame_count=frame_count):
                self.assertEqual(output["height_mm"].shape, (192, 192))
                self.assertTrue(np.isfinite(output["height_mm"]).all())
                self.assertEqual(
                    len(output["selected_pressure_indices"]),
                    frame_count,
                )
        output = outputs[5]
        self.assertLess(
            output["metrics"]["height_nrmse"],
            output["physics_metrics"]["height_nrmse"],
        )
        self.assertGreater(output["metrics"]["mask_iou"], 0.90)


if __name__ == "__main__":
    unittest.main()
