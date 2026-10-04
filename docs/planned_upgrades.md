# Planned upgrades

Improvements identified from the current results. Each one is evaluated on the validation set first; the test set is used once per final model.

## Tasks 1-3 (restoration autoencoders)

Current limit: every output passes through the required compressed bottleneck (8x8 latent, 2,048 values), which removes fine texture (measured ceiling ~27 dB / SSIM 0.85). Optuna picked the largest size allowed in Task 1 and for the Task 2 specialists. Task 3 is built from the Task 2 experts, so improving them improves Tasks 2 and 3 together.

| Upgrade | Expected effect | Notes |
|---|---|---|
| Larger latent grid (16x16 instead of 8x8, still compressed, e.g. 12x) | Clearly sharper outputs | No skip connections; fully within the PDF rules |
| Wider search space (`--version v2`: base up to 96, bottleneck up to 4096) | +0.5-1.5 dB | Scripts ready (`optuna_task1.py`, `optuna_task2_specialists.py`, `optuna_task3.py` with `--version v2`) |
| Limited skip connections (1-2 shortcuts at 32x32 / 64x64) | Largest sharpness gain | Needs an ablation (with vs without skips) to justify, as the PDF requires |

Order: Task 1 -> Task 2 specialists -> Task 3 rebuilt on the new specialists -> evaluate -> export ONNX.

## Task 4 (face-to-sketch cGAN)

Current limits: checkpoint chosen by L1/SSIM (rewards soft strokes), lambda_L1 = 153 (strong pull towards the pixel average), ~300 training pairs per style at 128x128, and style/photo-source confound in FS2K (style 2 only appears with `photo1` faces in training).

| Upgrade | Expected effect |
|---|---|
| Choose the checkpoint by inspecting the fixed validation samples (epoch 55 vs later epochs) | Crisper strokes |
| Lower lambda_L1 (~30-50) and longer training (250+ epochs) | More "drawn", less smudgy sketches |
| Balance styles across photo sources in training batches + colour/brightness augmentation | Better generalisation to new photos |
| Spectral normalisation in the discriminator or a least-squares GAN loss | More stable training, sharper results |
