# Multimodal Fake News Detection using Text–Image Fusion

Detects fake Reddit posts (Fakeddit, 2-way: 0 = fake, 1 = real) by combining what the **headline says** (DistilBERT, 768-d) with what the **image shows** (ResNet50, 2048-d).

## Files
| File | What it is |
|---|---|
| `Multimodal_Fake_News_Detection.ipynb` | Complete Colab notebook: data → features → 5 models × 5 seeds → results → ablation → save → launch app |
| `fakenews_core.py` | Shared feature extractors, models, explanation code (the notebook also writes this file to Drive) |
| `app.py` | Streamlit web app (the notebook also writes this to Drive) |
| `requirements.txt` | For running the app locally |

## How to run
1. Upload the notebook to Colab, set **Runtime ▸ T4 GPU**.
2. Put `multimodal_train.tsv` and `multimodal_test_public.tsv` from Fakeddit in `MyDrive/fakeddit_project/data/` (or your existing subset CSVs + images — see §1 of the notebook).
3. *Runtime ▸ Run all*. First run ≈ 30–60 min (mostly image downloads); later runs use the Drive cache.
4. The last code cell prints a public `trycloudflare.com` link to the app.

## Models compared (same test split, mean ± std over 5 seeds)
- TF-IDF + LogReg / DT / GB / RF (reference-repo baseline, text only)
- Text-only MLP · Image-only MLP
- **Early fusion** (concatenate features) · **Late fusion** (learned weighting of two classifiers) · **Attention fusion** (cross-modal attention + gate)

Explanation: global ablation (F1 drop when a modality is removed) and per-post text/image contribution shares.

---

## Prompt to continue or extend this project with an AI assistant

Paste this, attach the notebook, and edit the bracketed parts:

```
You are an ML engineer helping me finish my final-year project
"Multimodal Fake News Detection using Text-Image Fusion".

CONTEXT
- Dataset: Fakeddit, 2-way labels (0 = fake, 1 = real). Subset: 6,000 balanced
  train posts (split 85/15 -> 5,100 train / 900 val, stratified, seed 42) and
  1,500 held-out test posts. Only posts with images on i.redd.it/imgur; images
  resized to 224x224; headlines truncated to 64 tokens.
- Features are FROZEN: DistilBERT (768-d, mean pooled) and ResNet50 (2048-d),
  cached as .npy on Google Drive and standardised with train mean/std.
- Already done: text-only, image-only, early/late/attention fusion MLPs trained
  with 5 seeds; accuracy + macro-F1 (mean ± std); ablation; Streamlit app.
  The attached notebook is the current code.
- Environment: Google Colab free tier, T4 GPU, ~12 GB RAM.

TASK
[Describe exactly what you want, e.g. "add CLIP features as a 4th fusion
variant and compare it with the others" / "fix this error: <paste traceback>"
/ "write the Results and Discussion chapter from these numbers: <paste table>"]

RULES
1. Keep the same data split and test set so all numbers stay comparable.
2. Select models with the validation set only; report test results once.
3. Report accuracy and macro-F1 as mean ± std over seeds [0,1,2,3,4].
4. Give complete, runnable Colab cells (no "..." placeholders), in order,
   with a one-line comment per cell saying what it does.
5. Reuse fakenews_core.py; if you change a model, update it there so the
   Streamlit app still works.
6. Before the code, explain your plan in 5 bullet points or fewer. After it,
   tell me what output to expect and how to verify it worked.
7. If something is ambiguous, state your assumption and continue.
```
