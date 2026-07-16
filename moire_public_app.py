#!/usr/bin/env python3
import streamlit as st

from moire_sim_app import RIGID_POC_CONFIG, render_rigid_poc
from moire_sim_platform import load_config


def render_public_app():
    st.set_page_config(
        page_title="MoiréSkin Public Demo",
        page_icon="◉",
        layout="wide",
    )
    render_rigid_poc(load_config(RIGID_POC_CONFIG), public_demo=True)


if __name__ == "__main__":
    render_public_app()
