# Knowledge base: Generative AI Assignment 1

Re-read this file before every phase. Source of truth order: `docs/assignment.pdf` > `CLAUDE.md` > this file. If they disagree, the PDF wins and the conflict is reported.

---

## 1. The big picture

We build four generative systems and put them into one web app:

1. **Universal Restoration (Task 1):** one denoising autoencoder that fixes any of 3 corruptions (or leaves a clean image alone) without being told which corruption it is.
2. **Hard-Routed Restoration (Task 2):** a classifier guesses the corruption type, then sends the image to one of 3 specialist autoencoders. A clean image goes through an identity bypass (it's returned unchanged).
3. **Soft Mixture-of-Experts Restoration (Task 3):** the Task 2 classifier becomes a "gate" that gives a weight to all 4 branches (identity + 3 specialists). The output is the weighted sum. Gate and experts are fine-tuned together.
4. **Face-to-Sketch Generator (Task 4):** a conditional GAN (U-Net generator + PatchGAN discriminator) that turns a face photo into a sketch in one of 3 artist styles.

Tasks 1-3 use Oxford-IIIT Pet with synthetic corruptions. Task 4 uses FS2K. Every inference model is exported to ONNX. A FastAPI backend runs ONNX Runtime (CPU), a React + Tailwind frontend shows the four workspaces, and Docker Compose starts everything with one command.

---

## 2. Data rules

### 2.1 Oxford-IIIT Pet (Tasks 1, 2, 3)
- Development data = official `trainval` (3,680 images). Split 80/20 (2,944 / 736) with **seed 42**. Save the split once to `artifacts/splits/` and reuse it in Tasks 1, 2 and 3. Never re-split.
- Official `test` (3,669 images) stays untouched until final evaluation. No Optuna objective, early stopping or checkpoint choice may use it.
- Convert to RGB, resize to **128x128**. Class labels (37 breeds) are not used. The clean image is the target.
- Only load ids listed in `trainval.txt` / `test.txt`. Ignore `._*` files, `.mat` files and the 41 unlisted jpgs.

### 2.2 Training corruptions (runtime, random every time an image is loaded)
Pick one of 4 conditions with equal probability (1/4 each); the condition index is the classifier label:

| Label | Condition | Training configuration |
|---|---|---|
| 0 | clean | unchanged image |
| 1 | salt-and-pepper | p ~ U(0.02, 0.15); each selected pixel becomes black or white with equal probability |
| 2 | Gaussian blur | kernel size from {3, 5, 7}; sigma ~ U(0.5, 2.5) |
| 3 | occlusion | 1-3 black rectangles, together covering 10%-35% of the image, random positions |

Do **not** save corrupted copies of the dataset. (A cache of the *clean* 128x128 images in `artifacts/` is fine and makes loading faster.)

### 2.3 Validation and test corruptions (deterministic)
- Generate a **validation manifest** and a **test manifest** once and store them in `artifacts/manifests/`. Each entry stores: image id, corruption type, severity, mask coordinates, blur kernel/sigma, and the random seed for that image. Always reuse them.
- **Test** = every clean test image gets **3 fixed severity levels for every corruption** (plus the clean version):

| Corruption | Low | Medium | High |
|---|---|---|---|
| Salt-and-pepper p | 0.03 | 0.08 | 0.15 |
| Blur (kernel, sigma) | (3, 0.7) | (5, 1.5) | (7, 2.5) |
| Occlusion (area, rectangles) | ~10%, 1 rect | ~20%, 2 rects | ~35%, 3 rects |

  That's 1 clean + 9 corrupted = 10 inputs per test image, i.e. 36,690 test inputs.
- Results are always reported **per corruption AND per severity** (low/medium/high), and separately for clean.

### 2.4 FS2K (Task 4)
- Official train (1,058) / test (1,046) from `anno_train.json` / `anno_test.json`.
- Hold out **15% of official train** as validation, **stratified by `style`**, **seed 42**. Test is never used in training or tuning.
- Resize photo and sketch to 128x128. Keep pairing exact. Any spatial augmentation (flip, crop, rotation, resize) is applied **identically** to photo and sketch.
- Style field: `style` in {0,1,2} = UI "Style 1/2/3". Details in `docs/dataset_findings.md`.

---

## 3. Task 1: Universal denoising autoencoder

- **Architecture:** conv encoder (spatial size shrinks, channels grow) -> genuinely compressed latent bottleneck -> conv decoder back to 3x128x128 RGB. No unrestricted skip connections (that would be a U-Net that can copy the input). If limited skips are used, run an ablation and justify them in the report.
- **Forward:** x_hat = D(E(x_tilde)), where x = clean, x_tilde = corrupted, x_hat = output.
- **Loss:** `L_UDAE = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))`, starting at alpha = 0.8, final alpha chosen by Optuna.
- **Optuna (at least):** learning rate, batch size, bottleneck dimension, number of encoder channels, dropout, alpha. The validation objective must combine reconstruction quality and structural similarity. Report: full search space, completed/pruned/failed trial counts, best trial, final config.
- **Evaluation (test manifest):** PSNR, SSIM, L1 for clean / salt / blur / occlusion, each at low/medium/high.
- **Visuals:** clean target | corrupted input | output | absolute error map. At least **12 representative examples** and **4 meaningful failure cases**, discussed in text.
- **ONNX** + app workspace **"Universal Restoration"**: upload a corrupted image, or pick a clean sample and apply a corruption type + severity. Show input, output, chosen corruption settings, inference time.

## 4. Task 2: Classifier + hard-routed specialists

- **Classifier:** CNN, 4 classes [clean, salt, blur, occlusion], cross-entropy, **balanced batches**. Output p = C(x_tilde) = [p_clean, p_salt, p_blur, p_occlusion], prediction r = argmax_k p_k.
  - Optuna: learning rate, batch size, channel configuration, dropout, weight decay.
  - Report: overall accuracy, macro precision/recall/F1, per-class metrics, **normalized 4x4 confusion matrix**.
- **Specialists:** 3 autoencoders (salt, blur, occlusion). Each trained **only on its own corruption**, clean image as target, **independent weights**. Same architecture is allowed. A shared Optuna search for the common architecture is allowed. Tune: learning rate, bottleneck size, channel config, batch size, L1-vs-SSIM weight.
- **Hard routing:**
  - r = clean -> x_hat = x_tilde (identity bypass, no expert)
  - r = salt -> A_salt(x_tilde); r = blur -> A_blur(x_tilde); r = occlusion -> A_occ(x_tilde)
- **Two evaluation modes:** oracle routing (label from the test manifest) and predicted routing (classifier). Find and discuss cases where a classifier error caused a restoration failure.
- **ONNX:** classifier + all 3 specialists. App workspace **"Hard-Routed Restoration"**: show the 4 probabilities, predicted corruption, selected expert, output, inference time.

## 5. Task 3: Soft mixture-of-experts

- **Gate:** `w = softmax(G(x_tilde) / T)`, w = [w0 identity, w1 salt, w2 blur, w3 occlusion]. Temperature T: small T = sharp (near one-hot), large T = spread out.
- **Output:** `x_hat = w0 * x_tilde + w1 * A_salt(x_tilde) + w2 * A_blur(x_tilde) + w3 * A_occ(x_tilde)`. This is differentiable, so gate and experts train together through the reconstruction error.
- **Initialization (mandatory):** gate from the Task 2 classifier, experts from the Task 2 specialists. No random start.
- **Stage 1 warm-up:** experts frozen, train only the gate. **Stage 2:** unfreeze everything, joint fine-tune with a **smaller** learning rate.
- **Loss:** `L_MoE = l1_w * L1 + ssim_w * (1 - SSIM) + cls_w * CE(gate logits, true label) + bal_w * L_balance`, starting at 0.8 / 0.2 / 0.1 / 0.01.
- **Balance loss:** `L_balance = sum_{k=0..3} (mean_w_k - 1/4)^2`, where mean_w_k = average weight of branch k over a **balanced** batch. An alternative (e.g. entropy, Switch-Transformer load-balancing loss) is allowed only with research-backed justification.
- **Optuna:** fine-tune learning rate, temperature T, classification weight, balance weight, reconstruction weighting. Pruning is allowed for bad trials or **routing collapse**.
- **Analysis:** average expert weight per true corruption type **and** severity; routing heatmap / weight-distribution plot; examples where one expert dominates and where weights are spread out; check for inactive experts and for one expert dominating unrelated inputs.
- **ONNX:** export the **entire** pipeline (gate + 3 experts + identity + mixing) as **one** graph. App workspace **"Soft Mixture-of-Experts Restoration"**: show all 4 weights, the output, inference time, and a visual cue for which experts contributed most.

## 6. Task 4: FS2K face-to-sketch cGAN

- **Generator:** U-Net, `y_hat = G(x, s)`, x = photo, s = style. The style is a **learned categorical embedding** (3 styles) that goes **into the generator AND the discriminator** (not just a UI label).
- **Discriminator:** PatchGAN, sees photo + style + sketch: D(x, y, s) for real, D(x, G(x, s), s) for fake. It judges local patches as real/fake.
- **Losses:** D learns to output real for real pairs and fake for generated ones (BCE-with-logits is fine). `L_G = L_adv + lambda_L1 * L1(y, G(x, s))`, starting at lambda_L1 = 100 (pix2pix value), tuned by Optuna.
- **Optuna (at least):** generator LR, discriminator LR, batch size, base channels, dropout, style-embedding dimension, L1 weight. Trials may use fewer epochs; then **retrain the best config on the full schedule**.
- **Logging (separately):** D real loss, D fake loss, G adversarial loss, G L1 loss, validation metrics. Log sample images at fixed intervals using the **same validation photos** every time.
- Save checkpoints often (GANs can diverge). Handle diverging Optuna trials gracefully (prune on NaN/explosion).
- **ONNX:** generator only. App workspace **"Face-to-Sketch Generator"**: upload a photo or capture from webcam, choose Style 1/2/3, show photo and sketch side by side, download button.

---

## 7. Cross-cutting requirements

### Optuna (all four tasks)
- Persistent SQLite storage in `artifacts/optuna/` so studies can resume.
- Short trials (few epochs, maybe a subset) + pruner (e.g. MedianPruner).
- AMP where stable (check it doesn't destabilize the GAN).
- Search ranges that fit 4 GB (batch <= 64 for 128x128 autoencoders, smaller for the GAN).
- Catch `torch.cuda.OutOfMemoryError` -> `torch.cuda.empty_cache()` -> `raise optuna.TrialPruned`.
- Report completed, pruned and failed trial counts.

### Experiment tracking (MLflow)
- Local MLflow (`mlruns/`, gitignored) from the first training script. One experiment per task. Log params, per-epoch losses, metrics, checkpoints, sample images. Each Optuna trial = an MLflow run nested under a parent run per study.
- The demo video must show the experiment-tracking records (MLflow UI).

### ONNX
- Models: Task 1 AE, classifier, 3 specialists, full soft-MoE, Task 4 generator (7 files).
- `model.eval()` before export. Fixed 128x128 input, dynamic batch axis OK.
- `scripts/verify_onnx.py`: ONNX Runtime vs PyTorch on real samples, report max/mean absolute difference, save results for the report.
- Files go in `models/onnx/`, committed directly (no LFS). Report file sizes. If any is > 50 MB: GitHub Release + download script instead.

### Application
- One app, four workspaces with the **exact names**:
  - **Universal Restoration**
  - **Hard-Routed Restoration**
  - **Soft Mixture-of-Experts Restoration**
  - **Face-to-Sketch Generator**
- Design first in **Google Stitch** (done by the user; screenshots go in the report). Don't invent a different design.
- Frontend: React + Tailwind. Backend: FastAPI + ONNX Runtime (CPU).
- **Minimum backend endpoints** (exact paths decided in Phase 6):
  - health check (`GET /health`: which models loaded; detect Git LFS pointer files instead of real ONNX files)
  - universal restoration
  - hard routing
  - soft mixture
  - face-to-sketch
  - plus helpers the UI needs (list samples, apply corruption)
- Backend validates uploads (type, size, decodable image), preprocesses, runs ONNX, returns output + routing info + timing.
- Frontend features: upload, pick a clean sample, apply chosen corruption + severity, upload an already-corrupted image, show classifier probabilities and mixture weights, webcam for Task 4, download button, inference time.
- 15-20 small sample images bundled in `backend/samples/` so the app works on a fresh clone without `data/`.
- In Compose the frontend reaches the backend via service name or a proxy (not `localhost`). Configure CORS.

### Docker (see CLAUDE.md section 9; follow it strictly)
- Frontend and backend in containers; one documented `docker compose up` command.
- `docker info` before anything; `docker system df` before/after builds; stop if C: free < 15 GB; max 2 build attempts per problem; build timeout ~10 min; `.dockerignore` in every context; slim pinned base images; multi-stage; no PyTorch in the backend; models mounted read-only (`./models/onnx:/app/models:ro`); healthchecks; log rotation; no pruning without asking.
- The evaluator must be able to: clone, get models, start containers, open the browser, with no VS Code and no manual Python scripts.

### Repository contents (required)
Source code, configs, `requirements.txt`, data-prep scripts, training scripts, evaluation scripts, Optuna studies, ONNX export code, app code, Dockerfiles, Compose file, README with complete execution instructions. No datasets, no large model files.

### Report (IEEE LaTeX, written by the user)
- Sections: introduction, related research, dataset preparation, architecture, losses, training, Optuna design, experimental setup, results, analysis, application architecture, limitations, conclusion. Each task gets its own clearly identifiable methodology and results.
- Must include: architecture diagrams, full corruption configuration, train/val curves, Optuna results, confusion matrices, result tables, routing-weight visualizations, generated-image grids, error maps, app screenshots, failure cases, Stitch evidence, GitHub link, YouTube link.
- **Every figure/table must be interpreted in the text.** Explain alternatives investigated, why choices were made, difficulties, and how research/experiments resolved them.
- **AI-use appendix:** tools used, what for, how outputs were tested or corrected (we keep `docs/ai_use_log.md` running).

### Video (user)
5-7 minutes on YouTube (link only in report). Must show: app startup, image upload, runtime corruption, universal restoration, hard routing, soft expert weights, face-to-sketch, result download, experiment-tracking records.

### Viva (live)
May be asked to justify architecture, explain a research decision, interpret a result, **modify part of the code**, or run the app on **unseen images**, including restarting the app from the repo. So code stays simple and every decision is explained.

---

## 8. Key formulas (plain text)

- Autoencoder: `x_hat = D(E(x_tilde))`
- Restoration loss: `L = alpha * L1(x, x_hat) + (1 - alpha) * (1 - SSIM(x, x_hat))`
- Classifier: `p = softmax(C(x_tilde))`, `r = argmax_k p_k`, loss = cross-entropy
- Gate: `w = softmax(G(x_tilde) / T)`
- Soft-MoE output: `x_hat = w0*x_tilde + w1*A_salt(x_tilde) + w2*A_blur(x_tilde) + w3*A_occ(x_tilde)`
- MoE loss: `L = l1_w*L1 + ssim_w*(1 - SSIM) + cls_w*CE + bal_w*L_balance` (start 0.8, 0.2, 0.1, 0.01)
- Balance: `L_balance = sum_k (mean_w_k - 1/4)^2` over a balanced batch
- cGAN discriminator: `L_D = BCE(D(x, y, s), 1) + BCE(D(x, G(x, s), s), 0)` (logged as "D real" and "D fake")
- cGAN generator: `L_G = BCE(D(x, G(x, s), s), 1) + lambda_L1 * L1(y, G(x, s))` (start lambda_L1 = 100)
- PSNR (images in [0,1]): `PSNR = 10 * log10(1 / MSE)`

---

## 9. Context and constraints from planning (not in the PDF)

- Windows 11, PowerShell, VS Code. Project at `C:\dev\gen-ai-ass1`, deliberately outside OneDrive. Never move it back.
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop, 4 GB VRAM. Driver 616.92 (CUDA 13.4) since 2026-10-03.
- Training + Optuna run in a local `.venv`, not Docker. Docker is only for the inference app (CPU onnxruntime). Verify `torch.cuda.is_available()`; never silently fall back to CPU. Ask before large downloads/installs.
- Colab/Kaggle may be used for heavy runs (e.g. Task 4 full retrain).
- Docker Desktop 28.3.2, Compose v2.38.2, ~7.6 GB RAM. A past build loop filled the disk, so the Docker safety rules are strict.
- Raw data under `data/` is READ-ONLY. Derived files go to `artifacts/`. Archive backups are in `C:\dataset-backup` (do not touch).
- No hosting (no Vercel). Local Docker Compose only (compulsory).
- The PDF deadline (March 16, 2024) is stale; ignore it for planning. **Reminder for the user: confirm the real deadline with the instructor.**
- GitHub repo: `https://github.com/humaa-taj/Image-Processing-Engine` (branch `main`).
- ONNX models committed directly in `models/onnx/` (no LFS) unless > 50 MB. Checkpoints stay gitignored.
- Fresh-clone requirement: `git clone`, one documented `docker compose` command, open the browser, working within minutes. `/health` reports loaded models and detects LFS pointer files. A fresh-clone test is done at the end.
- The user does: Google Stitch design, YouTube video, IEEE LaTeX report, AI-use appendix. Claude prepares figures, tables, result files and `docs/ai_use_log.md` entries, and reminds the user when these are due.
- Never fabricate results; every report number comes from a real logged run.
- Communication: plain, concise, exact commands, at most one question at a time.

---

## 10. Dataset findings (summary; full details in `docs/dataset_findings.md`)

- **Pet:** trainval 3,680, test 3,669, none missing or unreadable, no overlap. 3 listed images are RGBA with fully opaque alpha (`.convert("RGB")` is safe). Sizes range from 114x108 to 3264x2606. The 80/20 seed-42 split gives 2,944 / 736.
- **FS2K:** train 1,058 (style 357/350/351), test 1,046 (style 619/381/46). Label field `style` in {0,1,2}. Pairing: `photo/photoN/imageXXXX` -> `sketch/sketchN/sketchXXXX`, with mixed `.jpg`/`.JPG`/`.png`. All pairs exist and have matching sizes. Sketches are grayscale. Style is confounded with photo source (style 2 = photo2/photo3 only, in train). The stratified 15% val split (~159 images) is feasible.

---

## 11. Glossary (viva)

- **Autoencoder:** a network that compresses an input into a small code (encoder) and rebuilds an image from that code (decoder). A *denoising* autoencoder gets a corrupted input and is trained to output the clean version.
- **Bottleneck / latent:** the smallest internal representation between encoder and decoder. Because it's small, the network must keep only the important content instead of copying pixels.
- **Skip connection:** a shortcut passing encoder features straight to the decoder. Unrestricted skips let the network bypass the bottleneck (copy the input), which is why the assignment limits them.
- **L1 loss:** mean absolute pixel difference. Robust and gives sharper results than L2/MSE.
- **SSIM:** Structural Similarity Index. Compares local brightness, contrast and structure in small windows; 1 means identical. `1 - SSIM` is used as a loss.
- **PSNR:** Peak Signal-to-Noise Ratio, in dB, derived from MSE. Higher is better.
- **Salt-and-pepper noise:** random pixels forced to pure black or pure white. **Gaussian blur:** each pixel replaced by a Gaussian-weighted average of its neighbours. **Occlusion:** parts of the image hidden by black boxes, so the model must "inpaint" them.
- **Manifest:** a saved file listing exactly which corruption (with all parameters and seed) is applied to each val/test image, so evaluation is repeatable.
- **Hard routing:** pick exactly one expert (argmax of the classifier). Fast, but a wrong classification sends the image to the wrong expert.
- **Soft routing / mixture-of-experts (MoE):** a gate gives every expert a weight, and the output is the weighted sum. It's differentiable, so the gate can be trained by the reconstruction loss, and uncertain or mixed inputs can blend experts.
- **Gate:** the small network that produces the routing weights (here: the Task 2 classifier).
- **Temperature (T):** divides the logits before softmax. Low T makes weights sharper (closer to hard routing); high T makes them more even.
- **Routing collapse:** the gate sends almost everything to one expert (or never uses one), so the other experts stop learning. A balance loss counteracts it.
- **Balance loss:** a penalty that grows when the average weight per branch drifts away from equal use (1/4 each) over a balanced batch.
- **Identity branch:** a "do nothing" expert that returns the input. The correct choice for clean images.
- **GAN:** a generator makes images and a discriminator tries to tell real from fake; they train against each other.
- **cGAN (conditional GAN):** both networks also receive a condition (here: the photo and the style), so the output must match that condition.
- **pix2pix:** the standard paired image-to-image cGAN (U-Net generator, PatchGAN discriminator, adversarial + lambda * L1 loss, lambda = 100).
- **U-Net:** encoder-decoder with skip connections at every level. Good for image-to-image tasks where the output aligns with the input (the photo and sketch share layout). Allowed in Task 4, not as a plain copy path in Task 1.
- **PatchGAN:** a discriminator that outputs a grid of real/fake scores, one per local patch (e.g. 70x70 receptive field), instead of one score per image. It focuses on local texture and sharpness.
- **Style embedding:** a learned vector (one per style) that is fed into the networks so they know which style to produce/judge.
- **Mode collapse / divergence:** GAN failure modes. The generator produces nearly the same output for everything, or the losses blow up. Frequent checkpoints guard against both.
- **Optuna:** a hyperparameter-search library. A *study* runs many *trials*, each with sampled hyperparameters (TPE sampler by default).
- **Optuna pruning:** stopping a trial early when its intermediate validation score is worse than others at the same step (MedianPruner), which saves GPU time. We also prune on CUDA out-of-memory.
- **AMP (mixed precision):** running parts of training in float16 to save memory and time.
- **MLflow:** experiment tracker that logs parameters, metrics, images and checkpoints per run, with a web UI.
- **ONNX:** an open, framework-independent file format for trained networks. **ONNX Runtime** runs it fast on CPU without PyTorch, which is why the Docker image stays small.
- **FastAPI:** Python web framework for the backend API. **React + Tailwind:** frontend library + utility-class CSS framework. **Docker Compose:** starts several containers (frontend, backend) with one command.

---

## 12. Risks and traps

1. **GPU driver too old.** RESOLVED 2026-10-03: driver updated to 616.92 (CUDA 13.4); using torch 2.14.1+cu126.
2. **Optuna objective must not depend on a tuned loss weight.** If the objective were the training loss itself, Optuna could "win" by changing alpha (or lambda_L1, or MoE weights) rather than improving restoration. Use a fixed objective, e.g. a fixed mix of val L1 and val SSIM, or PSNR/SSIM. DECIDED: see section 13, item 6.
3. **Clean vs mild blur look alike.** Blur with kernel 3, sigma ~0.5 barely changes an image (and Pet JPEGs are already slightly soft). Expect clean/blur confusion in the classifier. Same for very low salt probability vs clean. This is a real failure source to discuss, not a bug.
4. **Occlusion area control.** Rectangles must jointly cover 10-35% (test: ~10/20/35% with 1/2/3 rects). Overlapping rectangles would under-count the area, so we generate non-overlapping rects and store exact coordinates in the manifest.
5. **Salt-and-pepper definition.** We treat a "pixel" as all 3 channels at once (the whole pixel turns black or white), not per channel. State this in the report.
6. **Balanced batches.** Random 1/4 sampling is only balanced on average. For the classifier (and the MoE balance loss) we enforce exact balance: each batch has B/4 of each condition.
7. **Validation manifest design is not specified in the PDF.** DECIDED: one condition per val image, 184 per class (section 13, item 4).
8. **Data loading speed on Windows.** Decoding large JPEGs every epoch with `num_workers=0-2` would bottleneck the GPU. Plan: cache clean 128x128 uint8 arrays in `artifacts/cache/` (~180 MB for trainval) and corrupt on the fly. That isn't a "corrupted copy", so it's allowed.
9. **Aspect ratio.** Direct 128x128 resize distorts non-square images (Pet and FS2K photo2/photo3). Literal reading of the PDF = direct resize. Must be identical between training and the app. DECIDED: direct resize (section 13, item 5).
10. **FS2K style confound + tiny style-2 test set (46).** The GAN may tie style to photo source. Report per-style results with this caveat.
11. **Webcam in the browser** only works in a secure context. `http://localhost` qualifies; a LAN IP does not.
12. **ONNX export of the soft-MoE** must include the identity branch, softmax with T, and the weighted sum in one graph. Verify the graph returns both the image and the weights.
13. **Train/serve skew:** the backend must use exactly the same preprocessing (RGB, resize method, [0,1] scaling) and the same corruption code as training. Share/port the corruption code carefully and test it against the PyTorch version.
14. **AMP + SSIM / GAN:** SSIM in float16 can be unstable. Compute losses in float32; disable AMP for the GAN if losses spike.
15. **Docker disk usage** (past incident). Strict rules in CLAUDE.md section 9. Recommend capping BuildKit cache in Docker Desktop settings.
16. **Global Python has torch 2.11.0+cpu installed.** Always use `.venv\Scripts\python`, otherwise training silently runs on CPU.
17. **`gh` CLI is not installed.** Pushing uses Git Credential Manager over HTTPS. Fine, but GitHub Releases (only needed if a model > 50 MB) would need the web UI or `gh`.
18. **Unanchored .gitignore patterns.** `data/` also matched `src/data/` (found in Phase 1 before committing; fixed to `/data/`). Before each commit, check that `git add -A -n` lists every new source file.

---

## 13. Approved design decisions (2026-10-03)

These were approved by the user. Each one must be stated and justified in the report.

1. **Phases:** 7 phases as in `docs/implementation_plan.md` (CLAUDE.md section 0 updated to match).
2. **PyTorch:** NVIDIA driver updated to 616.92 (CUDA 13.4), so we use torch 2.14.1+cu126 in `.venv`.
3. **ONNX files** are committed directly to `models/onnx/` (no LFS). If any file is over 50 MB, tell the user first.
4. **Validation manifest:** each of the 736 validation images gets exactly **one** condition, **184 per class** (clean / salt / blur / occlusion). Severity is drawn from the *training* ranges using a per-image seed stored in the manifest. Why: it matches the training distribution, keeps classes exactly balanced (important for classifier metrics and the MoE balance check), and keeps Optuna validation fast (736 inputs instead of 7,360). The test manifest still uses the full fixed 3-severity scheme.
5. **Resizing:** direct resize to 128x128 (no crop, no padding), as the PDF literally says. Limitation to document: non-square images are squashed (Pet sizes vary widely; FS2K photo2 is 223x318 and photo3 is 475x340). The app applies the identical resize, so training and serving match. For FS2K, photo and sketch are squashed identically, so pairing stays exact.
6. **Optuna objective (Tasks 1-3):** a fixed score `0.5 * val_L1 + 0.5 * (1 - val_SSIM)` that does **not** depend on the tuned loss weight (alpha / L1-SSIM weight / MoE weights). Why: if the objective were the training loss itself, Optuna could lower the score just by changing alpha (e.g. towards the term that is numerically smaller) without the images getting any better. A fixed yardstick makes trials comparable. Task 4 uses its own fixed validation score, not the GAN loss.
7. **Clean vs mild-blur confusion** in the classifier is expected (kernel 3 / sigma ~0.5 barely changes a slightly soft JPEG). We measure it and discuss it in the report as a property of the data, not hide it.
8. **Small artifacts are committed:** `artifacts/splits/`, `artifacts/manifests/`, `artifacts/optuna/*.db` (the PDF requires "Optuna studies" in the repo) and the final result tables. Check sizes first and keep each file under ~20 MB; tell the user if any is larger. Everything else in `artifacts/` (e.g. `cache/`) stays ignored.
9. **AI-use log:** a short entry in `docs/ai_use_log.md` at the end of every phase.

## 14. Phase 1 implementation decisions (2026-10-03)

1. **MLflow storage:** MLflow 3.16 refuses the old plain-folder store, so runs live in `mlruns/mlflow.db` (SQLite) and files in `mlruns/artifacts/`. Both are inside the gitignored `mlruns/`. Open the UI with `.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db` and browse to http://127.0.0.1:5000.
2. **Corruption code is NumPy-only** (`src/corruptions.py`), with every corruption described by a JSON params dict. The manifests store these dicts, and the backend will import the same file, so training, evaluation and the app corrupt images identically.
3. **Occlusion rectangles never overlap**, so the covered area is exactly the sum of rectangle areas (tests check coverage is within 1 percentage point of the target). Area is split randomly between rectangles; aspect ratio is between 1:2 and 2:1.
4. **Blur** is a separable Gaussian with reflect borders. A test checks it against SciPy (max difference 1 grey level, i.e. rounding).
5. **Resize** uses PIL bilinear (with PIL's built-in antialiasing when shrinking) for both Pet and FS2K. The app must use the same call.
6. **Clean image cache:** `artifacts/cache/pet_{trainval,test}.npy`, 181 + 180 MB, gitignored and rebuilt by `scripts/prepare_data.py` in ~45 s. Clean images only, no corrupted copies.
7. **Randomness:** training corruptions use a NumPy generator seeded from PyTorch's RNG, so each DataLoader worker draws different corruptions and a seeded run is repeatable.
8. **Balanced batches:** `BalancedBatchSampler` gives exactly batch_size/4 of each condition, with a new shuffle each epoch (`set_epoch`). Batch sizes must be multiples of 4.
9. **Manifest seeds:** the val manifest uses master seed 42, the test manifest uses 43 (separate streams). Each entry stores its own seed and full params. Re-running `prepare_data.py` checks that the outputs are byte-identical.
10. **FS2K sketches load as 1-channel grayscale** (they are R=G=B). Training augmentation = load at 143x143, then the same random 128x128 crop and the same horizontal flip for photo and sketch (pix2pix "jitter"). A test confirms the pair stays aligned.
11. **SSIM** is our own ~20-line implementation (11x11 Gaussian window, sigma 1.5, Wang et al. 2004), always computed in float32. A test checks it against scikit-image (difference < 0.01, from border handling).
12. **DataLoader:** `num_workers=2` is stable on Windows (spawn) and ~35% faster than 0. Scripts are guarded with `if __name__ == "__main__":`.
13. **Benchmark** (stand-in models): AE epoch 2.8-2.9 s (batch 32, AMP, 0.38 GB); GAN epoch 7.9 s (batch 8, fp32, 2.71 GB peak). GAN memory is the tight constraint on 4 GB.

## 15. Phase 2 (Task 1) design decisions

1. **SSIM verified on real images:** our SSIM vs scikit-image (`gaussian_weights=True, sigma=1.5, use_sample_covariance=False`) on 160 real validation inputs (40 per condition): max abs difference 1.9e-5, mean 2.0e-6 (float32 rounding). Both use an 11x11 window and average only where the full window fits. `scripts/check_ssim.py` -> `artifacts/results/ssim_check.json`; enforced by a test.
2. **Convolutional bottleneck, no skip connections.** Encoder: 4 stages (strided 3x3 conv + 3x3 conv, BN, ReLU), 128->8 pixels, channels b->2b->4b->8b. A 1x1 conv squeezes to an 8x8xc latent (bottleneck_dim = 64c = 256..2048, i.e. 192x..24x fewer numbers than the 49,152 input values). Decoder: 1x1 conv back up, then 4 x (nearest upsample + two 3x3 convs), sigmoid output. Why spatial, not a dense vector: a dense layer from 8x8x512 needs tens of millions of weights (ONNX > 100 MB; Task 3 puts three experts in one file) and throws away spatial layout. Why no skips: the PDF requires a genuine bottleneck, and with no skips there is nothing to ablate. A test proves `forward(x) == decode(encode(x))`.
3. **Upsampling:** nearest-neighbour upsample + 3x3 conv instead of transposed convolution, to avoid checkerboard artifacts (Odena, Dumoulin & Olah, 2016, "Deconvolution and Checkerboard Artifacts", Distill).
4. **Dropout** is applied to the latent code (training only).
5. **Training:** AdamW (weight decay 1e-5), cosine learning-rate decay over the run, AMP fp16 for the network, loss computed in float32. Training condition per image: uniform over the 4 conditions (as specified). Validation = the fixed val manifest (736 images, 184 per class).
6. **Optuna (Task 1):** TPE sampler (seed 42), MedianPruner (5 startup trials, 3 warm-up epochs), 15 epochs per trial, every trial starts from seed 42. Objective = best epoch's `0.5*val_L1 + 0.5*(1 - val_SSIM)`. Space: lr log[1e-4, 3e-3]; batch {16, 32, 64}; bottleneck {256, 512, 1024, 2048}; base channels {24, 32, 48, 64}; dropout [0, 0.3]; alpha [0.5, 0.95]. Largest model = 7.1 M params (~28 MB ONNX), peak 1.7 GB at batch 64.
7. **Final training:** best config, up to 100 epochs, early stopping (patience 20) on the validation score; best-validation checkpoint kept. Test is used once, by `evaluate_task1.py`, which refuses to re-run without `--force`.
8. **Evaluation baseline:** every test metric is reported for the output AND for the untouched corrupted input, so the gain is visible (for clean inputs the input PSNR is infinite, so the autoencoder can only lose quality there; this is the cost of a single universal model, and it motivates the identity branch in Tasks 2-3).
9. **Figure selection is rule-based, not cherry-picked:** 12 examples = the median-SSIM test input of each corruption x severity group (9) + clean inputs at the 25/50/75th SSIM percentile (3). 4 failures = the lowest-PSNR test input of each condition. Error maps use one fixed colour scale (0-0.5 mean abs error).
10. **ONNX:** torch.export-based exporter (`dynamo=True`, the default in torch 2.14); `verbose=False` is required on Windows (the exporter prints emoji that crash a cp1252 console). Verified on 64 real validation inputs at batch 64 and batch 1 (tolerance 1e-4).
11. **Task 1 results and what they mean (test set, run once):**
    - Optuna: 30 trials (16 completed, 14 pruned, 0 failed). Best = the largest model in the space (base 64, bottleneck 2048) with almost no dropout (0.013) and alpha 0.56 (more SSIM weight than the 0.8 start). fANOVA importance: dropout 0.75, lr 0.17, alpha 0.07, rest < 0.01 (from 16 completed trials only, so rough). Best at the edge of the space -> a bigger model might do better; capped for ONNX size (limitation).
    - The autoencoder has a **quality ceiling of ~27.3 dB / SSIM 0.85**: even a clean input comes out at that level, because everything must pass through the 2048-number latent (fine texture is lost). So:
      - **salt**: big win (+7 to +14 dB) because noisy inputs are far below the ceiling;
      - **occlusion**: +7.7 to +9.6 dB PSNR, but holes are filled with plausible blurry colour, not real detail (information is truly missing);
      - **low/medium blur** and **clean**: the output is *worse* than the input (blur low: 34.1 -> 27.4 dB), since those inputs are already above the ceiling. Only high blur improves slightly (+1.0 dB).
    - This is the main argument for Task 2/3: an identity bypass for clean images and specialists that don't have to share one latent space.
    - Failure cases (lowest PSNR per condition): dark/black backgrounds get brightened at the borders (clean and salt cases, a regression-to-the-mean effect); busy high-frequency textures under strong blur can't be recovered; a large occluder hiding a distinctive object (a white frisbee) gets filled with background-like texture.

## 16. Phase 3 (Task 2) design decisions

1. **Classifier architecture:** 4 stages of (two 3x3 conv + BN + ReLU, 2x2 max-pool), widths b, 2b, 4b, 8b, then global average pool AND global max pool concatenated, dropout, linear -> 4 logits. Average pooling captures image-wide properties (blur = missing fine detail everywhere); max pooling keeps the strongest local evidence (a few salt pixels, one rectangle edge) that averaging would dilute.
2. **Balanced batches:** `BalancedBatchSampler` gives exactly batch/4 images per class, new shuffle each epoch. Batch sizes {32, 64, 128} (multiples of 4).
3. **Classifier objective = validation cross-entropy**, not accuracy: the classifier initializes the Task 3 gate, whose softmax outputs are used as mixing weights, so calibrated probabilities matter. Accuracy and macro-F1 are logged alongside.
4. **Classifier search:** lr log[1e-4, 3e-3]; batch {32, 64, 128}; base channels {16, 24, 32, 48}; dropout [0, 0.5]; weight decay log[1e-6, 1e-2]; 12 epochs per trial; 25 trials.
5. **Specialists:** same `ConvAutoencoder` as Task 1, three independent models, each trained AND validated only on its own corruption (184 val images each). One shared Optuna search (allowed by the PDF): each trial trains salt -> blur -> occlusion with identical hyperparameters, objective = mean of their best val scores. Reported to the pruner as steps 1-10 / 11-20 / 21-30 so trials are compared at the same specialist and epoch.
6. **Specialist search:** lr log[1e-4, 3e-3]; batch {16, 32, 64}; bottleneck {512, 1024, 2048}; base channels {24, 32, 40}; alpha [0.5, 0.95]; dropout fixed 0 (not in the PDF's list for specialists; Task 1 found ~0 best). Base channels capped at 40 so Task 3's single ONNX file (3 experts + gate) stays < 50 MB: worst case 3 x 11.2 MB + 10.6 MB = ~44 MB.
7. **Hard routing (`src/routing.py`):** route 0 -> identity (input returned bit-exact, no expert runs), routes 1-3 -> the matching expert. Predicted route = argmax of the classifier logits; oracle route = manifest label.
8. **Evaluation:** oracle vs predicted routing per corruption x severity, side by side with Task 1 on the same test inputs. Misrouting cost measured in SSIM and L1 (a clean image routed to an expert has infinite oracle PSNR, so PSNR alone can't express that cost).
9. **ONNX:** classifier exported with two outputs, `logits` (for the Task 3 gate) and `probs` (shown in the app); each specialist as its own file.
10. **Task 2 results and what they mean (test set, run once):**
    - Classifier Optuna: 25 trials (7 completed, 18 pruned, 0 failed); best #23: base 48, batch 32, lr 5.5e-4, dropout 0.32, wd 2.4e-5. Final: best epoch 28 of 38 (early stop), val acc 99.73%.
    - Specialist Optuna: 20 trials (10 completed, 10 pruned, 0 failed); best #17: base 40 (the cap), bottleneck 2048, lr 2.1e-3, batch 16, alpha 0.53. Final val SSIM: salt 0.844, blur 0.848, occlusion 0.739.
    - **Classifier on test: accuracy 99.83%, macro P/R/F1 0.998/0.997/0.997.** Only 63 of 36,690 inputs misrouted: 41 clean -> blur, 3 clean -> occlusion, 19 low occlusion -> clean. Blur is detected 100% even at the lowest level (k=3, sigma 0.7), so the expected clean/blur confusion shows up the other way round: naturally soft/out-of-focus clean photos get called "blur".
    - **Misrouting costs:** clean -> blur expert loses ~0.11 SSIM and the blur expert visibly shifts colours on sharp, saturated images (it only ever saw blurred inputs, so sharp images are out of its training distribution). Clean -> occlusion happens on pets photographed on pure black backgrounds (large black areas look like occluders). Low occlusion -> identity *helps* (+0.085 SSIM): the occlusion expert damages the uncovered 90% of the image more than one small box costs.
    - **Oracle vs predicted routing are almost identical** (overall SSIM 0.8269 vs 0.8268), because the classifier is so accurate.
    - **vs Task 1:** the big win is clean images (identity: SSIM 1.0 vs 0.849), which lifts overall SSIM from 0.810 to 0.827. Per corruption, the specialists are about equal in SSIM (salt 0.848 vs 0.843, blur 0.828 vs 0.827, occlusion 0.747 vs 0.748) but 0.3-0.8 dB *lower* in PSNR. Likely cause: the specialists are smaller (base 40 vs 64, capped for the Task 3 ONNX size), and their alpha (0.53) weights SSIM more than Task 1's (0.56). So specialisation alone did not beat a bigger universal model at the same bottleneck ceiling (~26-27 dB). Low/medium blur is still made worse than the input, since the bottleneck ceiling also applies to the blur expert.

## 17. Phase 4 (Task 3) design decisions

1. **Model (`src/models/soft_moe.py`):** `logits = G(x)`, `w = softmax(logits / T)`, `x_hat = w0*x + w1*A_salt(x) + w2*A_blur(x) + w3*A_occ(x)`. All 4 branches run on every image. T is a stored buffer (exported with the model, not trained). Returns (x_hat, w, logits).
2. **Initialization:** gate = Task 2 classifier weights, experts = Task 2 specialist weights (`build_moe_from_task2`). Epoch 0 of every run evaluates this untouched model as a baseline.
3. **Two stages:** warm-up = experts frozen (`requires_grad=False`), only the gate trains (lr 5e-4); joint = everything trains with the tuned, smaller lr (1e-5..3e-4, vs 2e-3 for the specialists).
4. **Experts' BatchNorm statistics stay frozen** during joint training (weights still train). Reason: in the mixture every expert sees every input (clean, salt, blur, occlusion), so updating its running mean/variance would drift its normalization away from its own corruption. The gate's BatchNorm trains normally (it always saw all 4 conditions).
5. **Loss:** `l1_w*L1 + ssim_w*(1-SSIM) + cls_w*CE + bal_w*balance`, with CE computed on `logits / T` (the actual routing distribution), and balance = `sum_k (mean_w_k - 1/4)^2` on exactly balanced batches (the PDF's suggestion). We kept the PDF's balance loss: with exactly balanced batches its target (1/4 per branch) is exactly right, so no alternative was needed.
6. **Optuna:** lr log[1e-5, 3e-4], T [0.5, 3], cls_w log[0.01, 1], bal_w log[0.001, 0.1], l1_share [0.5, 0.95] (ssim_w = 1 - l1_share). Fixed: batch 32, warm-up 2 epochs, 6 joint epochs per trial. Objective: the same fixed val score as Tasks 1-2.
7. **Routing collapse pruning:** a trial is pruned if, on the balanced val set, any branch's mean weight < 0.05 (inactive expert) or > 0.5 (one branch takes over).
8. **Evaluation:** comparison with Task 1 and Task 2 (oracle and predicted) on the same test inputs; mean branch weights per corruption x severity (heatmap); distribution of the correct-branch weight; dominant (lowest-entropy) vs distributed (highest-entropy) examples; expert health check (inactive / dominating unrelated inputs); re-check of the 63 inputs Task 2 misrouted. Note: on the test set clean inputs are only 10% of the data, so identity's overall mean weight is naturally ~0.1; per-condition weights are the meaningful view.

## 18. Model-size policy change (2026-10-04, user decision)

- **No more model-size caps for the sake of file size.** The 50 MB figure is GitHub's warning threshold (it hard-blocks files > 100 MB), not a hardware limit. Model quality is the higher priority.
- ONNX files <= 50 MB: committed to git as before. ONNX files > 50 MB: kept local only (`scripts/export_onnx.py` adds them to `models/onnx/.gitignore`) and shared via a download link later.
- Effect so far: Task 1 (base <= 64) and the Task 2 specialists (base <= 40, chosen so Task 3's single file stayed < 50 MB) were capped, and Optuna picked the cap both times. This is a stated limitation and the target of a later upgrade pass (bigger models, re-evaluated). Task 3 inherits the Task 2 experts.
- Task 4: generator base channels {32, 48, 64} (64 = standard pix2pix, ~168 MB ONNX), no cap.

## 19. Task 3 results and what they mean (test set, run once)

- **Optuna `task3`:** 20 trials = 16 completed, 4 pruned, 0 failed. Best #17: T = 2.63, joint lr 2.7e-4, l1/ssim = 0.59/0.41, cls_w 0.010 and bal_w 0.0011 (both at the bottom of their ranges: the best mixture is the least constrained one).
- **Final training:** val score 0.0860 (untouched Task 2 init) -> 0.0702 (best epoch 30 of 3 warm-up + 30 joint), val SSIM 0.859 -> 0.885. The gate's top-1 "accuracy" fell from 99.7% to 77.5%: it stopped classifying and started blending.
- **Test, all corrupted inputs:** SSIM 0.806 (Task 1) / 0.808 (Task 2 predicted) / **0.845 (Task 3)**; PSNR 25.55 / 24.93 / **26.18 dB**. Clean: SSIM 0.9995 (PSNR 59 dB, near-identity).
- **Biggest gains where the bottleneck ceiling hurt before:** low blur SSIM 0.843 -> 0.950 (PSNR 26.3 -> 33.1 dB, now *close to the input's 34.1*); medium blur 0.841 -> 0.874; occlusion overall 0.747 -> 0.801. Salt gains a little (0.848 -> 0.856).
- **How (routing heatmap):** the gate mixes the original image back in, more for milder damage: blur low/med/high = 79/55/46% identity; occlusion low/med/high = 52/38/27% identity; salt ~96% salt expert (no part of a noisy image is worth keeping). The identity branch acts like a learned, input-dependent shortcut that escapes the autoencoders' ~27 dB ceiling for mild corruptions. Nobody programmed this; it emerged from the reconstruction loss once cls_w was small.
- **Expert health:** no inactive expert (each gets >= 0.39 mean weight on its own inputs). The automatic check flags *identity* as "dominating unrelated inputs" (mean 0.34 on corrupted inputs, threshold 0.2), but this is the beneficial blending described above, not collapse: restoration got better, not worse. The salt/blur/occlusion experts get ~0 weight on unrelated corruptions (no cross-talk).
- **Task 2's 63 misrouted inputs:** Task 3 is better on 95% of them; mean SSIM 0.891 (Task 2 hard) -> 0.974 (Task 3), above even Task 2's oracle routing (0.947). Clean photos that hard routing sent to the blur expert now get 94% identity (SSIM 0.999).
- **Trade-offs to discuss:** lower top-1 gate accuracy (77.5%) is expected and harmless here; PSNR on high occlusion is slightly lower than Task 1 (20.25 vs 20.40 dB) because 27% of the black box is blended back in, while SSIM is higher (0.707 vs 0.686).
- **ONNX:** whole pipeline (gate + identity + 3 experts + softmax(logits/T) + weighted sum) as one graph, 44.36 MB, outputs `output`, `weights`, `logits`; max diff vs PyTorch 2.3e-5.

## 20. Task 4 results and what they mean (FS2K official test set, run once)

- **Optuna `task4`:** 20 trials x 20 epochs = 19 completed, 1 pruned, 0 failed (0 OOM; even base 64 at batch 16 fit). Best #11 (val score 0.3022): **generator base 64** (the standard pix2pix size, i.e. the largest option), batch 8, lr_G 3.4e-4, lr_D 3.4e-4, dropout 0.33, style embedding 16, lambda_L1 152.6.
- **Final training:** 150 epochs (~25 min). Best validation score at **epoch 55** (0.2884, SSIM 0.513); by epoch 150 it drifted to 0.309. Typical GAN behaviour: later epochs push towards crisper, more sketch-like strokes, which pixel metrics (L1/SSIM) score slightly worse. The checkpoint is chosen on validation only (epoch 55).
- **Test (1,046 pairs):** L1 0.0986, SSIM 0.498, PSNR 16.4 dB. Per style: Style 1 SSIM 0.542 (n=619), Style 2 0.411 (n=381, heavy dark shading is hardest), Style 3 0.626 (n=46; small sample).
- **Style conditioning works:** for the same photo, switching the style changes the sketch by a mean |difference| of 0.11 (11% of the grey range): Style 1 = light thin lines, Style 2 = dark heavy shading, Style 3 = soft grey tones (see `docs/figures/task4_examples.png`).
- **Confound reminder:** style is tied to photo source in FS2K (style 3 = stock photos in train). Per source: photo1/style 1 SSIM 0.525 vs photo3/style 1 0.583; the single photo1/style-3 test image scores 0.339. Per-style numbers must be read with this in mind.
- **Why GAN metrics look low:** sketches are mostly white paper with thin lines; L1/SSIM punish strokes that are plausible but a few pixels off. That's why pix2pix-style work also relies on visual inspection (the fixed validation samples logged every 10 epochs in MLflow).
- **ONNX:** generator only, inputs `photo` [N,3,128,128] + `style` [N] int64, output `sketch` [N,1,128,128]; **167.9 MB** (kept local per the size policy, listed in `models/onnx/.gitignore`; needs a download link); max diff vs PyTorch 7.2e-7.
