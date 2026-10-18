"""Shepp–Logan phantom parallel-beam projection demonstration with animation."""
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

matplotlib.rcParams["font.family"] = ["DejaVu Sans", "sans-serif"]

IMAGE_SIZE = 256
ARROW_RADIUS = IMAGE_SIZE * 0.43
ARROW_LENGTH = IMAGE_SIZE * 0.09
ANIMATION_DELAY = 0.03

st.set_page_config(page_title="CT投影デモ", layout="wide")
st.title("CT投影デモ：Shepp–Loganファントムとサイノグラム")

n_projections = st.sidebar.slider("投影数", min_value=1, max_value=360, value=180)


@st.cache_data(show_spinner=False)
def calculate_sinogram(n_angles: int):
    phantom = resize(
        shepp_logan_phantom(),
        (IMAGE_SIZE, IMAGE_SIZE),
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True,
    )
    angles = np.linspace(0.0, 360.0, n_angles, endpoint=False)
    sinogram = radon(phantom, theta=angles, circle=True, preserve_range=True)
    return phantom, sinogram, angles


def draw_phantom(phantom: np.ndarray, angle_deg: float | None = None):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(phantom, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

    if angle_deg is not None:
        # Align the arrow with the Radon projection angle definition.
        # angle_deg = 0 should indicate a top-to-bottom projection.
        theta = np.deg2rad(angle_deg + 90.0)
        cx = (phantom.shape[1] - 1) / 2.0
        cy = (phantom.shape[0] - 1) / 2.0

        end_x = cx + ARROW_RADIUS * np.cos(theta)
        end_y = cy - ARROW_RADIUS * np.sin(theta)
        start_x = cx + (ARROW_RADIUS + ARROW_LENGTH) * np.cos(theta)
        start_y = cy - (ARROW_RADIUS + ARROW_LENGTH) * np.sin(theta)

        ax.annotate(
            "",
            xy=(end_x, end_y),
            xytext=(start_x, start_y),
            arrowprops=dict(arrowstyle="->", color="red", lw=2.5),
        )

    fig.tight_layout(pad=0)
    return fig


def draw_partial_sinogram(sinogram: np.ndarray, filled_cols: int):
    fig, ax = plt.subplots(figsize=(5, 5))

    partial = np.full_like(sinogram, np.nan, dtype=float)
    if filled_cols > 0:
        partial[:, :filled_cols] = sinogram[:, :filled_cols]

    cmap = plt.cm.gray.copy()
    cmap.set_bad(color="white")

    ax.imshow(
        partial,
        cmap=cmap,
        aspect="auto",
        origin="lower",
        extent=(0, 360, -sinogram.shape[0] / 2, sinogram.shape[0] / 2),
        vmin=float(np.min(sinogram)),
        vmax=float(np.max(sinogram)),
    )
    ax.set_xlabel("Projection angle [deg]")
    ax.set_ylabel("Detector position [pixel]")
    fig.tight_layout()
    return fig


phantom, sinogram, angles = calculate_sinogram(n_projections)

left, right = st.columns(2)
phantom_placeholder = left.empty()
sinogram_placeholder = right.empty()

with phantom_placeholder.container():
    st.subheader("Shepp–Loganファントム")
    fig = draw_phantom(phantom)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

with sinogram_placeholder.container():
    st.subheader("サイノグラム")
    fig = draw_partial_sinogram(sinogram, 0)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

if st.button("投影処理開始", use_container_width=True):
    for idx, angle in enumerate(angles, start=1):
        with phantom_placeholder.container():
            st.subheader("Shepp–Loganファントム")
            fig = draw_phantom(phantom, angle)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        with sinogram_placeholder.container():
            st.subheader("サイノグラム")
            fig = draw_partial_sinogram(sinogram, idx)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

        time.sleep(ANIMATION_DELAY)
