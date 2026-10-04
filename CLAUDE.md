# CLAUDE.md: Generative AI Assignment 1

Read `docs/assignment.pdf` fully before writing any code. It is the source of truth. If this file and the PDF disagree, the PDF wins; tell me about the conflict.

## 0. Working style

- Work in phases. Finish one phase, summarize, and STOP for my review. Do not start the next phase on your own.
- Phases (details in `docs/implementation_plan.md`): (1) foundation: env + data + corruption + manifests + shared utilities, (2) Task 1 + ONNX export/verify tooling, (3) Task 2, (4) Task 3, (5) Task 4, (6) FastAPI backend + React/Tailwind frontend, (7) Docker Compose + fresh-clone test + README + report material. Each task phase exports and verifies its own ONNX models.
- Commit to git after each phase with a clear message. Add a short entry to `docs/ai_use_log.md` for every phase.
- Keep `PROGRESS.md` updated (done / in progress / next / known issues) so a new session can resume.
- NEVER fabricate numbers. Every metric in the report must come from a real logged run. If something has not been run, say so.
- Ask before any destructive action (deleting files, pruning Docker, overwriting checkpoints).
- Stay inside this project folder. Do not touch files elsewhere on the machine.
- I must understand all code (live viva). Prefer simple, readable code with short comments explaining design decisions.
- Never commit secrets (API keys, tokens). Use environment variables and `.env` (gitignored).

## 1. Environment

- OS: Windows 11, PowerShell. Project at `C:\dev\gen-ai-ass1` (outside OneDrive; keep it that way). Avoid spaces in paths.
- GPU: NVIDIA laptop GPU with **4 GB VRAM**. Docker Desktop has about 7.6 GB RAM.
- Training and Optuna run in a local Python venv (NOT in Docker). Docker is for the inference app only.
- Create a venv (`.venv`, gitignored). Install the CUDA build of PyTorch (check pytorch.org for the command that matches my driver). Verify `torch.cuda.is_available()` is True before any training. Never silently fall back to CPU.
- Windows DataLoader: spawn start method. Guard scripts with `if __name__ == "__main__":`. Start with `num_workers=0` or 2 and only raise it if it is stable.
- All shell scripts must use LF line endings (`.gitattributes` enforces this).
- Pin dependencies in `requirements.txt` (training) and `backend/requirements.txt` (inference, CPU `onnxruntime` only).

## 2. Data (READ-ONLY)

Raw data lives in `data/`. **Never write to, modify, move or delete anything under `data/`.** Write all derived output (splits, manifests, caches) to `artifacts/`.

- Oxford-IIIT Pet: `data/oxford-iiit-pet/images/` and `data/oxford-iiit-pet/annotations/` (`trainval.txt`, `test.txt`, `list.txt`, `trimaps/`, `xmls/`). Read only the files listed in `trainval.txt` / `test.txt`. Ignore stray `._*` files.
- FS2K: `data/fs2k/FS2K/` with `photo/`, `sketch/`, `anno_train.json`, `anno_test.json`, `README.pdf`. The repo code is in `data/FS2K-code/FS2K-main/` (see its `tools/` and README).
- I have NOT verified the FS2K JSON structure or the key names for the style label. Inspect the JSON files, `README.pdf` and the `tools/` scripts first, report what you find, and confirm the photo-to-sketch pairing and the style label before writing any loader.
- Backups of the original archives are in `C:\dataset-backup` (outside the project).

## 3. Tasks 1-3: Oxford-IIIT Pet rules

- Dev data = official `trainval`. Split 80/20 with **random seed 42**. Save the split indices to `artifacts/splits/` and reuse the same file for Tasks 1, 2 and 3. Never re-split on the fly.
- Official `test` stays untouched until final evaluation. No Optuna objective, early stopping or checkpoint choice may use test data.
- Convert to RGB, resize to 128x128.
- Training corruption is applied at RUNTIME in the data pipeline. Each time an image is loaded, pick one of 4 conditions with equal probability (clean, salt-and-pepper, gaussian blur, occlusion) and sample new severity. Do not save corrupted copies of the dataset. The corruption label is returned for the classifier.
  - Salt-and-pepper: p ~ U(0.02, 0.15); affected pixels become black or white with equal probability.
  - Gaussian blur: kernel in {3, 5, 7}; sigma ~ U(0.5, 2.5).
  - Occlusion: 1 to 3 black rectangles, jointly covering 10-35% of the image area, random locations.
- Validation and test corruptions are DETERMINISTIC. Generate a validation manifest and a test manifest once (type, severity, mask coordinates, blur settings, random seed per image), store them in `artifacts/manifests/`, and always reuse them.
- Test severities, 3 fixed levels per corruption:
  - Salt-and-pepper: 0.03, 0.08, 0.15
  - Blur (kernel, sigma): (3, 0.7), (5, 1.5), (7, 2.5)
  - Occlusion: ~10%, 20%, 35% with 1, 2, 3 rectangles
- Report results per corruption AND per severity (low / medium / high).

### Task 1: Universal denoising autoencoder
- Conv encoder -> genuine compressed latent bottleneck -> conv decoder. Encoder shrinks spatial size and grows channels. No unrestricted skip connections. If limited skips are used, investigate and justify them (ablation).
- Loss: `alpha * L1 + (1 - alpha) * (1 - SSIM)`. Start alpha = 0.8; final value chosen by Optuna.
- Optuna must tune at least: learning rate, batch size, bottleneck dim, encoder channels, dropout, alpha. Report the search space, completed trials, best trial, final config.
- Evaluation: PSNR, SSIM, L1 per corruption and severity. Visuals: clean target, corrupted input, output, absolute error map. At least 12 examples and 4 failure cases.
- App workspace name: **Universal Restoration**.

### Task 2: Classifier + hard-routed specialists
- CNN classifier, 4 classes (clean, salt, blur, occlusion), cross-entropy, BALANCED batches. Optuna: learning rate, batch size, channel config, dropout, weight decay.
- Report accuracy, macro precision/recall/F1, per-class metrics, normalized 4x4 confusion matrix.
- Three specialist autoencoders (salt, blur, occlusion), each trained ONLY on its own corruption with independent weights. A shared Optuna search for a common architecture is allowed; tune learning rate, bottleneck size, channels, batch size, L1/SSIM weight.
- Clean input uses an IDENTITY bypass (no expert).
- Evaluate in two modes: oracle routing (manifest label) and predicted routing (classifier). Identify and discuss failures caused by classifier errors.
- App workspace name: **Hard-Routed Restoration** (show 4 probabilities, predicted class, selected expert, output, inference time).

### Task 3: Soft mixture-of-experts
- Gate outputs `softmax(g(x)/T)` over 4 branches: [identity, salt, blur, occlusion]. Output = weighted sum of branches.
- Initialize gate from the Task 2 classifier and experts from the Task 2 specialists. No random start.
- Stage 1: warm-up with experts frozen, train gate only. Stage 2: unfreeze all, joint fine-tune with a smaller learning rate.
- Loss: `l1_w*L1 + ssim_w*(1-SSIM) + cls_w*CE + bal_w*balance`. Start 0.8, 0.2, 0.1, 0.01. Balance loss e.g. sum over branches of (mean_weight_i - 1/4)^2 on a balanced batch; any alternative must be justified with research.
- Optuna: fine-tune learning rate, temperature T, classification weight, balance weight, reconstruction weighting. Pruning allowed for routing collapse.
- Analysis: average expert weights per true corruption type AND severity, routing heatmap, examples of one dominant expert vs distributed weights, check for inactive or over-dominant experts.
- Export the ENTIRE soft-MoE pipeline (gate + experts + identity + mixing) as one ONNX graph.
- App workspace name: **Soft Mixture-of-Experts Restoration**.

## 4. Task 4: FS2K face-to-sketch cGAN

- Use the official train/test definitions (`anno_train.json`, `anno_test.json`). Hold out 15% of the official TRAIN set as validation, stratified by sketch style, seed 42. Never touch test during training or tuning.
- Resize photos and sketches to 128x128. Keep photo-sketch pairing exact.
- Generator: U-Net, input = photo + style condition (learned categorical embedding for 3 styles). The embedding must go INTO the generator and the discriminator, not just be a UI label.
- Discriminator: PatchGAN conditioned on photo, style and real/generated sketch.
- Generator loss: `adversarial + lambda_L1 * L1`. Start lambda_L1 = 100; tune with Optuna. BCE-with-logits is fine.
- Optuna: generator LR, discriminator LR, batch size, base channels, dropout, style-embedding dimension, L1 weight. Trials may use fewer epochs; retrain the best config for the full schedule.
- Spatial augmentation (flip, crop, rotation, resize) must be applied IDENTICALLY to photo and sketch.
- Log SEPARATELY: D real loss, D fake loss, G adversarial loss, G L1 loss, validation metrics. Log samples at fixed intervals using the SAME validation photos.
- Save checkpoints often (GANs diverge). Handle diverging Optuna trials gracefully.
- Only the generator is exported to ONNX.
- App workspace name: **Face-to-Sketch Generator** (upload or webcam, Style 1/2/3, side-by-side view, download button).

## 5. Optuna on a 4 GB GPU

- Use Optuna in ALL four tasks. Use a persistent SQLite storage in `artifacts/optuna/` so studies can resume.
- Use short trials (few epochs, optionally a training subset) plus a pruner (e.g. MedianPruner).
- Use mixed precision (AMP) where it is stable. In GAN training check that it does not destabilize the run.
- Restrict search ranges to configs that fit in 4 GB (e.g. batch size up to 64 for 128x128 autoencoders, smaller for the GAN).
- Wrap each trial in a handler for `torch.cuda.OutOfMemoryError`: free memory (`torch.cuda.empty_cache()`) and raise `optuna.TrialPruned`, so one oversized trial does not kill the study.
- Record in the report the number of completed, pruned and failed trials.
- Before any long run, estimate its time and tell me.

## 6. Experiment tracking

- Use MLflow (local, `mlruns/`, gitignored) from the FIRST training script, not retrofitted later. Log params, per-epoch losses, metrics, checkpoints and sample images. One experiment per task.
- Tag each Optuna trial as an MLflow run (nested under a parent run per study).

## 7. ONNX

- Export every inference model: Task 1 autoencoder, classifier, 3 specialists, full soft-MoE, Task 4 generator.
- Set `model.eval()` first (dropout/batchnorm). Fixed 128x128 input, dynamic batch axis is fine.
- Write `scripts/verify_onnx.py` that compares ONNX Runtime vs PyTorch outputs numerically (max abs diff, mean abs diff) on real samples. Save the results for the report.
- ONNX files go in `models/onnx/`. Report file sizes after export. **Model size is NOT limited by the 50 MB GitHub threshold** (decided 2026-10-04): model quality comes first. Files up to 50 MB are committed directly to git (no LFS); files over 50 MB stay local only (auto-listed in `models/onnx/.gitignore` by the export script) and get a download link later.
- Priorities: never limit a high-priority goal (model quality, required features) to satisfy a lower-priority convenience (file size, repo tidiness). If a trade-off appears, flag it to me with options before deciding.

## 8. Application

- Design first in Google Stitch (I do this manually and keep screenshots for the report). Do not invent a different design; ask me for the Stitch output.
- Frontend: React + Tailwind CSS. Backend: FastAPI. One app, four workspaces with the exact names above.
- Backend minimum endpoints: health check, universal restoration, hard routing, soft mixture, face-to-sketch.
- Backend must validate uploads (type, size, decodable image), preprocess, run ONNX inference, return the result plus routing info and timing.
- Frontend features: upload an image, pick a clean sample, apply a chosen corruption type and severity, upload an already-corrupted image, show classifier probabilities and mixture weights, webcam capture for Task 4, download button, inference time display.
- Inside Docker Compose the frontend reaches the backend through the service name or a proxy, not `localhost`. Configure CORS.

## 9. DOCKER SAFETY (important: I once had a build loop that filled my disk)

Docker Desktop must be running. Before ANY docker command, run `docker info`; if the Server section errors, STOP and tell me.

**Hard rules:**
1. Run `docker system df` BEFORE and AFTER every build and report the numbers. If free disk space on C: is under 15 GB, STOP and tell me.
2. Never put docker build / compose commands in a loop, retry script or watch process. Maximum 2 consecutive build attempts for the same problem; then STOP, show me the error, and wait.
3. Always run builds with a timeout and stream the output. If a build runs more than about 10 minutes, stop it and tell me.
4. Dockerfiles must NOT download datasets, train models, or run Optuna. They only install dependencies and copy application code.
5. Every build context needs a `.dockerignore` excluding at least: `data/`, `artifacts/`, `models/`, `mlruns/`, `.venv/`, `node_modules/`, `.git/`, `*.pt`, `*.pth`, `*.ckpt`, `*.zip`, `*.tar.gz`. Check the build context size before building.
6. Use slim base images and multi-stage builds (e.g. `python:3.11-slim`, Node build stage then a small nginx or static stage). Use CPU `onnxruntime` only. No PyTorch in the backend image.
7. Model files are mounted as a volume (`./models/onnx:/app/models:ro`), NOT copied into the image.
8. In `docker-compose.yml`: use `restart: "no"` or `unless-stopped` only after I approve, never a restart loop; add `healthcheck`s; add log rotation (`logging: driver: json-file, options: max-size: "10m", max-file: "3"`); no containers that write large data to volumes.
9. Never run `docker system prune -a`, `docker volume prune` or `docker builder prune` without asking me first. When I approve cleanup, show what will be removed before running it.
10. Never mount the whole project root or `data/` into a container as writable.
11. Pin base image versions; do not use `latest`.
12. Tell me how to cap BuildKit cache if it grows (Docker Desktop > Settings > Builders / Resources), and recommend it.

## 10. Repository and deliverables

Repo must contain: source code, configs, `requirements.txt`, data-prep scripts, training scripts, evaluation scripts, Optuna studies, ONNX export code, app code, Dockerfiles, Docker Compose file, and a README with complete execution instructions (clone, get models, one command to start). Do not commit datasets or large model files.

Suggested structure:
```
src/            shared code (data, corruptions, models, losses, metrics)
scripts/        prepare_data, train_*, optuna_*, evaluate_*, export_onnx, verify_onnx
configs/
artifacts/      splits, manifests, optuna dbs, results (gitignored where large)
models/         ONNX + checkpoints (gitignored / LFS)
backend/        FastAPI app + Dockerfile
frontend/       React + Tailwind + Dockerfile
docs/           assignment PDFs, Stitch screenshots, report figures
report/         IEEE LaTeX source
docker-compose.yml
README.md
PROGRESS.md
```

Also needed later (I do these myself, remind me): Google Stitch design, YouTube demo (5-7 min), IEEE LaTeX report with every figure interpreted in text, AI-use appendix. Keep a running log of AI-assisted work in `docs/ai_use_log.md` (what was generated, how I tested or corrected it).

## 11. First task for this session

Do NOT write model or training code yet. Do this:
1. Read `docs/assignment.pdf` and this file.
2. Summarize the requirements back to me briefly.
3. Inspect FS2K (`anno_train.json`, `anno_test.json`, `README.pdf`, `tools/`) and report the JSON structure, style label field, pairing between photo and sketch, and train/test counts.
4. Check `torch.cuda.is_available()` once PyTorch is installed.
5. Propose the repo structure and list anything ambiguous or risky.
Then STOP and wait for my approval.