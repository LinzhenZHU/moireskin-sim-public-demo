#!/usr/bin/env python3
import streamlit as st

from moire_sim_app import RIGID_POC_CONFIG, render_public_r1_demo
from moire_sim_platform import load_config


def render_public_app():
    st.set_page_config(
        page_title="MoiréSkin Public Demo",
        page_icon="◉",
        layout="wide",
    )
    render_public_r1_demo(load_config(RIGID_POC_CONFIG))


if __name__ == "__main__":
    render_public_app()
