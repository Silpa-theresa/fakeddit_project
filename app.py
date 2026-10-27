"""
Streamlit app — Multimodal Fake News Detection (Text + Image fusion)
Needs in the same folder:  fakenews_core.py,  models/best_model.pt,
                           models/config.json,  models/feature_stats.npz
Run:  streamlit run app.py
"""
import os, json
from io import BytesIO
import numpy as np
import torch
import streamlit as st
from PIL import Image

import fakenews_core as core

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "models")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(page_title="Fake News Detector", page_icon="🔎", layout="centered")


@st.cache_resource(show_spinner="Loading DistilBERT, ResNet50 and the fusion model…")
def load_everything():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    s = np.load(os.path.join(MODEL_DIR, "feature_stats.npz"))
    model = core.build_model(cfg["model_name"], **cfg["model_kwargs"])
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    model.to(DEVICE).eval()
    fx = core.FeatureExtractor(device=DEVICE, max_len=cfg["max_len"], text_pool=cfg["text_pool"],
                               img_size=cfg["img_size"])
    return cfg, s, model, fx


def load_image_from_url(url):
    import requests
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def analyse(headline, image, cfg, s, model, fx):
    t = core.standardise(fx.text([headline]), s["text_mean"], s["text_std"]).astype(np.float32)
    i = core.standardise(fx.image([image]), s["image_mean"], s["image_std"]).astype(np.float32)
    t, i = torch.tensor(t, device=DEVICE), torch.tensor(i, device=DEVICE)
    if len(model.uses) == 2:
        return core.explain(model, t, i)
    p, _ = core.predict_proba(model, t, i)          # single-modality model: no explanation
    return {"pred": p.argmax(1).cpu().numpy(), "prob": p.cpu().numpy()}


# ----------------------------------------------------------------- UI
st.title("🔎 Multimodal Fake News Detector")
st.caption("Judges a post's credibility from its **headline and image together** "
           "(DistilBERT + ResNet50 features, fused by a neural network trained on Fakeddit).")

try:
    cfg, stats, model, fx = load_everything()
except FileNotFoundError as e:
    st.error(f"Model files not found ({e.filename}). Run the notebook's save cell first, "
             "and keep the `models/` folder next to app.py.")
    st.stop()

with st.sidebar:
    st.subheader("Model")
    st.write(f"**{cfg['model_name'].replace('_', ' ').title()}**")
    st.metric("Test macro-F1 (5-seed mean)", f"{cfg['mean_test_macro_f1_5_seeds']:.3f}")
    st.metric("Test accuracy (saved seed)", f"{cfg['test_accuracy']:.3f}")
    st.caption("Research prototype trained on Reddit posts. Use it as a signal, not as a fact-check.")

headline = st.text_area("Headline / post text", placeholder="e.g. Scientists discover a new species of glowing frog")
src = st.radio("Image", ["Upload", "Image URL"], horizontal=True)
image = None
if src == "Upload":
    f = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if f:
        image = Image.open(f).convert("RGB")
else:
    url = st.text_input("Image URL")
    if url:
        try:
            image = load_image_from_url(url)
        except Exception as e:
            st.warning(f"Couldn't load that image: {e}")
if image is not None:
    st.image(image, width=320)

if st.button("Analyse", type="primary", disabled=not (headline.strip() and image is not None)):
    with st.spinner("Analysing…"):
        out = analyse(headline.strip(), image, cfg, stats, model, fx)
    pred, prob = int(out["pred"][0]), out["prob"][0]
    conf = float(prob[pred])
    if pred == 0:
        st.error(f"### Likely FAKE  ·  {conf:.0%} confidence")
    else:
        st.success(f"### Likely REAL  ·  {conf:.0%} confidence")
    st.progress(float(prob[1]), text=f"P(real) = {prob[1]:.2f}   ·   P(fake) = {prob[0]:.2f}")

    if "text_share" in out:
        st.subheader("What drove the decision?")
        ts, ims = float(out["text_share"][0]), float(out["image_share"][0])
        c1, c2 = st.columns(2)
        c1.metric("Text contribution", f"{ts:.0%}")
        c2.metric("Image contribution", f"{ims:.0%}")
        st.bar_chart({"share": {"Text": ts, "Image": ims}})
        pt, pi = out["prob_text_only_view"][0], out["prob_image_only_view"][0]
        st.caption(f"Text alone suggests **{core.LABEL_NAMES[int(pt.argmax())]}** ({pt.max():.0%}); "
                   f"image alone suggests **{core.LABEL_NAMES[int(pi.argmax())]}** ({pi.max():.0%}). "
                   "Contribution = how much the verdict's probability changes when that input is removed.")
        if "gate_text" in out:
            st.caption(f"Attention gate weights — text {out['gate_text'][0]:.2f}, image {out['gate_image'][0]:.2f}")
        if pt.argmax() != pi.argmax():
            st.info("The text and the image point in different directions — a common pattern in "
                    "out-of-context or misleading posts.")
