# MoiréSkin R1 Simulation Public Demo

[Open the hosted demo](https://moireskinsimdemo.streamlit.app/)

Read-only Streamlit proof of concept for five-pressure rigid-object contact
simulation and local 2.5D reconstruction from Moiré deformation plus
see-through appearance.

The public app runs the frozen R1
`adaptive_pressure_fusion_unet_hr` model through an 8.6 MB CPU ONNX artifact.
It compares R1 against the pure-physics and analytical POC v8 baselines on the
same generated interaction. The catalog includes 15 named objects and eight
deterministic procedural OOD surfaces. Parallel, cross, and hexagonal grating
patterns are supported.

The hosted app exposes simulation controls only. It does not generate datasets,
train models, accept filesystem paths, or include sealed-test samples.

```bash
python -m pip install -r requirements.txt
streamlit run moire_public_app.py
```

Online metrics use synthetic ground truth for per-example diagnostics. This is
a synthetic research prototype, not a validated real-sensor accuracy claim.
