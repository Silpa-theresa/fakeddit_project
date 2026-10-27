"""
fakenews_core.py
Shared code for the notebook AND the Streamlit app, so both use exactly the
same feature extractors and model definitions.

Multimodal Fake News Detection using Text-Image Fusion (Fakeddit, 2-way labels)
Label convention: 0 = fake, 1 = real
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TEXT_DIM, IMAGE_DIM, NUM_CLASSES = 768, 2048, 2
LABEL_NAMES = {0: "Fake", 1: "Real"}
TEXT_MODEL_NAME = "distilbert-base-uncased"


# ---------------------------------------------------------------------------
# 1. Frozen feature extractors (DistilBERT 768-d, ResNet50 2048-d)
# ---------------------------------------------------------------------------
class FeatureExtractor:
    """Frozen DistilBERT + ResNet50. Heavy imports happen lazily."""

    def __init__(self, device="cpu", max_len=64, text_pool="mean", img_size=224):
        from transformers import AutoTokenizer, AutoModel
        import torchvision
        from torchvision import transforms

        self.device, self.max_len, self.text_pool = device, max_len, text_pool
        self.tok = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        self.bert = AutoModel.from_pretrained(TEXT_MODEL_NAME).eval().to(device)

        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        cnn = torchvision.models.resnet50(weights=weights)
        cnn.fc = nn.Identity()  # keep the 2048-d pooled features
        self.cnn = cnn.eval().to(device)
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def text(self, texts, batch_size=64):
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(list(texts[s:s + batch_size]), padding=True, truncation=True,
                           max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.bert(**enc).last_hidden_state                     # (B, L, 768)
            if self.text_pool == "cls":
                v = h[:, 0]
            else:                                                       # masked mean pooling
                m = enc["attention_mask"].unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
            out.append(v.float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def image(self, pil_images, batch_size=64):
        out = []
        for s in range(0, len(pil_images), batch_size):
            x = torch.stack([self.tf(im.convert("RGB")) for im in pil_images[s:s + batch_size]])
            out.append(self.cnn(x.to(self.device)).float().cpu().numpy())
        return np.concatenate(out).astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Classifiers on frozen features. Every model: forward(t, i) -> (logits, info)
#    "Masking" a modality = feeding zeros. Features are standardised with
#    train-set mean/std, so zeros == the average train example (mean imputation).
# ---------------------------------------------------------------------------
def _head(in_dim, hidden, dropout):
    return nn.Sequential(nn.LayerNorm(in_dim), nn.Dropout(dropout), nn.Linear(in_dim, hidden),
                         nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, NUM_CLASSES))


class TextOnly(nn.Module):
    uses = ("text",)

    def __init__(self, hidden=256, dropout=0.3, **_):
        super().__init__()
        self.head = _head(TEXT_DIM, hidden, dropout)

    def forward(self, t, i):
        return self.head(t), {}


class ImageOnly(nn.Module):
    uses = ("image",)

    def __init__(self, hidden=256, dropout=0.3, **_):
        super().__init__()
        self.head = _head(IMAGE_DIM, hidden, dropout)

    def forward(self, t, i):
        return self.head(i), {}


class EarlyFusion(nn.Module):
    """Feature-level fusion: concatenate [text ; image] -> MLP."""
    uses = ("text", "image")

    def __init__(self, hidden=512, dropout=0.3, **_):
        super().__init__()
        self.head = _head(TEXT_DIM + IMAGE_DIM, hidden, dropout)

    def forward(self, t, i):
        return self.head(torch.cat([t, i], dim=1)), {}


class LateFusion(nn.Module):
    """Decision-level fusion: separate text and image classifiers whose logits are
    combined with a learned weight alpha. Each branch also gets its own auxiliary
    loss (see aux_loss) so that both stay independently meaningful."""
    uses = ("text", "image")

    def __init__(self, hidden=256, dropout=0.3, **_):
        super().__init__()
        self.text_head = _head(TEXT_DIM, hidden, dropout)
        self.image_head = _head(IMAGE_DIM, hidden, dropout)
        self.w = nn.Parameter(torch.zeros(1))  # alpha = sigmoid(w), starts at 0.5

    def forward(self, t, i):
        lt, li = self.text_head(t), self.image_head(i)
        a = torch.sigmoid(self.w)
        return a * lt + (1 - a) * li, {"logits_text": lt, "logits_image": li, "alpha": a}

    @staticmethod
    def aux_loss(info, y, crit):
        return 0.5 * (crit(info["logits_text"], y) + crit(info["logits_image"], y))


class AttentionFusion(nn.Module):
    """Attention-based fusion.
    1) project text and image to a shared d-dim space -> 2 modality tokens
    2) multi-head self-attention between the tokens (cross-modal interaction)
    3) a learned gate scores each token; softmax -> per-sample modality weights
    4) weighted sum -> classifier.  The gate weights are returned for explanation."""
    uses = ("text", "image")

    def __init__(self, d=256, heads=4, dropout=0.3, **_):
        super().__init__()
        self.pt = nn.Sequential(nn.LayerNorm(TEXT_DIM), nn.Dropout(dropout), nn.Linear(TEXT_DIM, d))
        self.pi = nn.Sequential(nn.LayerNorm(IMAGE_DIM), nn.Dropout(dropout), nn.Linear(IMAGE_DIM, d))
        self.mod_emb = nn.Parameter(torch.randn(2, d) * 0.02)   # tells tokens apart
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.gate = nn.Linear(d, 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, NUM_CLASSES))

    def forward(self, t, i):
        tok = torch.stack([self.pt(t), self.pi(i)], dim=1) + self.mod_emb      # (B, 2, d)
        a, attn_w = self.attn(tok, tok, tok, need_weights=True)                 # attn_w (B, 2, 2)
        z = self.ln1(tok + a)
        z = self.ln2(z + self.ffn(z))
        g = torch.softmax(self.gate(z).squeeze(-1), dim=1)                      # (B, 2)
        fused = (g.unsqueeze(-1) * z).sum(1)
        return self.cls(fused), {"gate": g, "attn": attn_w}


MODELS = {
    "text_only": TextOnly,
    "image_only": ImageOnly,
    "early_fusion": EarlyFusion,
    "late_fusion": LateFusion,
    "attention_fusion": AttentionFusion,
}


def build_model(name, **kw):
    return MODELS[name](**kw)


# ---------------------------------------------------------------------------
# 3. Inference + modality-contribution explanation (ablation)
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, t, i, mask=None):
    """t, i: standardised float tensors. mask: None | 'text' | 'image' (zeroed)."""
    model.eval()
    if mask == "text":
        t = torch.zeros_like(t)
    if mask == "image":
        i = torch.zeros_like(i)
    logits, info = model(t, i)
    return F.softmax(logits, dim=1), info


@torch.no_grad()
def explain(model, t, i):
    """Per-sample modality contribution by ablation.
    contribution(text)  = |p_full(pred) - p_text_masked(pred)|
    contribution(image) = |p_full(pred) - p_image_masked(pred)|
    Shares are these normalised to sum to 1."""
    p_full, info = predict_proba(model, t, i)
    pred = p_full.argmax(1)
    idx = torch.arange(len(pred), device=pred.device)
    p_nt, _ = predict_proba(model, t, i, mask="text")
    p_ni, _ = predict_proba(model, t, i, mask="image")
    c_text = (p_full[idx, pred] - p_nt[idx, pred]).abs()
    c_img = (p_full[idx, pred] - p_ni[idx, pred]).abs()
    tot = (c_text + c_img).clamp(min=1e-8)
    out = {
        "pred": pred.cpu().numpy(),
        "prob": p_full.cpu().numpy(),
        "text_share": (c_text / tot).cpu().numpy(),
        "image_share": (c_img / tot).cpu().numpy(),
        "prob_text_only_view": p_ni.cpu().numpy(),   # image masked -> the text's view
        "prob_image_only_view": p_nt.cpu().numpy(),  # text masked  -> the image's view
    }
    if "gate" in info:
        out["gate_text"], out["gate_image"] = info["gate"][:, 0].cpu().numpy(), info["gate"][:, 1].cpu().numpy()
    if "alpha" in info:
        out["alpha_text"] = float(info["alpha"].item())
    return out


def standardise(x, mean, std):
    return (x - mean) / std
