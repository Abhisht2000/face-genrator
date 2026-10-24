"""Streamlit dashboard for a trained mini DCGAN face generator."""
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision.utils import make_grid

from models import Generator

ROOT = Path(__file__).resolve().parent
GENERATOR_PATH = ROOT / "checkpoints" / "generator.pt"
HISTORY_PATH = ROOT / "training_history.csv"
SAMPLES_DIR = ROOT / "samples"


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@st.cache_resource
def load_generator(path_string, modified_time):
    path = Path(path_string)
    device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    latent_dim = int(checkpoint.get("latent_dim", 128))
    base_channels = int(checkpoint.get("base_channels", 64))
    model = Generator(latent_dim=latent_dim, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, latent_dim, int(checkpoint.get("epoch", 0)), device


def tensor_to_pil(tensor):
    tensor = tensor.detach().cpu().clamp(-1, 1)
    tensor = (tensor + 1) / 2
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(array)


def make_download(images):
    buffer = BytesIO()
    images[0].save(buffer, format="PNG")
    return buffer.getvalue()


st.set_page_config(page_title="Mini GAN Face Studio", page_icon="🧠", layout="wide")
st.title("🧠 Mini GAN Face Studio")
st.caption("Generate synthetic 64×64 face images using your locally trained DCGAN.")

with st.sidebar:
    st.header("Generator controls")
    count = st.select_slider("Images to generate", options=[1, 4, 9, 16, 25], value=9)
    seed = st.number_input("Random seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
    st.caption("A fixed seed makes the same latent samples reproducible for the same model.")

if not GENERATOR_PATH.exists():
    st.warning("No trained generator found yet.")
    st.markdown(
        "Train the model first. Put face images in `data/faces/`, then run "
        "`python train.py --data data/faces --epochs 30 --batch-size 32`."
    )
    st.stop()

try:
    model, latent_dim, trained_epochs, device = load_generator(
        str(GENERATOR_PATH), GENERATOR_PATH.stat().st_mtime
    )
except Exception as exc:
    st.error(f"Could not load the generator checkpoint: {exc}")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("Training epochs", trained_epochs)
col2.metric("Latent dimension", latent_dim)
col3.metric("Inference device", str(device).upper())

if st.button("✨ Generate faces", type="primary"):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise = torch.randn(count, latent_dim, 1, 1, generator=generator).to(device)
    with torch.inference_mode():
        generated = model(noise)
    images = [tensor_to_pil(img) for img in generated]
    st.session_state["generated_images"] = images

images = st.session_state.get("generated_images", [])
if images:
    st.subheader("Generated results")
    columns = st.columns(5)
    for i, image in enumerate(images):
        with columns[i % 5]:
            st.image(image, caption=f"Sample {i + 1}", use_container_width=True)
            st.download_button(
                "Download PNG",
                data=make_download([image]),
                file_name=f"synthetic_face_{i + 1:02d}.png",
                mime="image/png",
                key=f"download_{i}_{seed}",
            )
    zip_buffer = BytesIO()
    import zipfile
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for i, image in enumerate(images, 1):
            image_buffer = BytesIO()
            image.save(image_buffer, format="PNG")
            archive.writestr(f"synthetic_face_{i:02d}.png", image_buffer.getvalue())
    st.download_button(
        "Download all images (.zip)", data=zip_buffer.getvalue(),
        file_name="synthetic_faces.zip", mime="application/zip"
    )

st.divider()
st.subheader("Training analytics")
if HISTORY_PATH.exists():
    try:
        import pandas as pd
        history = pd.read_csv(HISTORY_PATH)
        if not history.empty:
            st.line_chart(
                history.set_index("epoch")[["generator_loss", "discriminator_loss"]]
            )
            st.download_button(
                "Download training history (CSV)",
                data=HISTORY_PATH.read_bytes(),
                file_name="training_history.csv",
                mime="text/csv",
            )
    except Exception as exc:
        st.info(f"Training history is not available yet: {exc}")
else:
    st.info("Training history will appear here after the first epoch.")

sample_files = sorted(SAMPLES_DIR.glob("epoch_*.png")) if SAMPLES_DIR.exists() else []
if sample_files:
    st.subheader("Training progress")
    selected = st.select_slider(
        "Preview saved epoch",
        options=sample_files,
        value=sample_files[-1],
        format_func=lambda path: path.stem.replace("_", " ").title(),
    )
    st.image(str(selected), caption=selected.stem.replace("_", " ").title(), width=500)

st.caption(
    "Research/educational demo. Generated faces may look distorted, and images should "
    "be clearly labelled as AI-generated."
)
