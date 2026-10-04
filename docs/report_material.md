# Report material index

Everything below comes from logged runs (MLflow experiments `task1-universal-ae`, `task2-hard-routing`, `task3-soft-moe`, `task4-face-to-sketch`). Design decisions and their reasons are in `docs/explanation.md` (sections 13-20); the API in `docs/api_contract.md`. The PDF requires every figure and table to be interpreted in the text: the "Say" column gives the main point of each one.

## Data and corruptions (all tasks)

| Item | File | Say |
|---|---|---|
| Corruption examples (test severities + runtime training samples) | `docs/figures/corruption_grid.png` | The exact corruption definitions; low/medium/high test levels |
| Dataset facts, splits | `docs/dataset_findings.md` | Pet 2,944 / 736 / 3,669; FS2K 899 / 159 / 1,046; style confound in FS2K |
| Test manifest | `artifacts/manifests/test_manifest.json` | 36,690 deterministic test inputs (clean + 3 corruptions x 3 levels) |

## Task 1: Universal Restoration

| Item | File | Say |
|---|---|---|
| Optuna search space, trials, best trial | `artifacts/results/task1_optuna_summary.json`, `task1_optuna_trials.csv` | 30 trials: 16 completed, 14 pruned, 0 failed; best #24 |
| Optuna plots | `docs/figures/task1_optuna_history.png`, `task1_optuna_importance.png` | Dropout most important (0.75); best at the top of the size range |
| Training curves | `docs/figures/task1_curves.png` | Converged by ~epoch 90, no overfitting; occlusion hardest |
| Results table (per corruption x severity, input vs output) | `artifacts/results/task1_test_table.tex` | Corrupted inputs 19.71 -> 25.55 dB, SSIM 0.639 -> 0.806; quality ceiling ~27 dB |
| 12 examples + error maps | `docs/figures/task1_examples.png` | Median-SSIM example of each group (not cherry-picked) |
| 4 failure cases | `docs/figures/task1_failures.png` | Dark backgrounds brightened, heavy blur on texture, large occluder hides an object |

## Task 2: Hard-Routed Restoration

| Item | File | Say |
|---|---|---|
| Classifier Optuna | `artifacts/results/task2_classifier_optuna_summary.json`, `docs/figures/task2_classifier_optuna_*.png` | 25 trials: 7 completed, 18 pruned, 0 failed |
| Specialist Optuna (shared search) | `artifacts/results/task2_specialists_optuna_summary.json`, `docs/figures/task2_specialists_optuna_*.png` | 20 trials: 10 completed, 10 pruned, 0 failed |
| Curves | `docs/figures/task2_classifier_curves.png`, `task2_specialists_curves.png` | Classifier val acc 99.73%; specialists converge |
| Classifier metrics | `artifacts/results/task2_classifier_metrics.json` | Test acc 99.83%, macro P/R/F1 0.998/0.997/0.997 |
| Normalized confusion matrix | `docs/figures/task2_confusion_matrix.png` | Only 63 / 36,690 misrouted; clean -> blur is the main error |
| Oracle vs predicted vs Task 1 | `artifacts/results/task2_comparison_table.tex` (+ `task2_oracle_table.tex`, `task2_predicted_table.tex`) | Oracle ~= predicted; gain over Task 1 comes from the identity bypass on clean images |
| Misrouting analysis | `artifacts/results/task2_misrouting.json`, `docs/figures/task2_misrouted.png` | Soft clean photos -> blur expert (colour shift); black backgrounds -> occlusion; low occlusion -> identity helps |
| Examples | `docs/figures/task2_examples.png` | Probabilities, route, output per group |

## Task 3: Soft Mixture-of-Experts Restoration

| Item | File | Say |
|---|---|---|
| Optuna | `artifacts/results/task3_optuna_summary.json`, `docs/figures/task3_optuna_*.png` | 20 trials: 16 completed, 4 pruned, 0 failed; best = high T, tiny cls/balance weights |
| Training curves (warm-up -> joint) | `docs/figures/task3_curves.png` | Val score 0.0860 (Task 2 init) -> 0.0702; gate accuracy falls as it learns to blend |
| Comparison with Tasks 1 and 2 | `artifacts/results/task3_comparison_table.tex` | Corrupted SSIM 0.806 / 0.808 / 0.845; low blur 26.3 -> 33.1 dB |
| Routing heatmap + weight distribution | `docs/figures/task3_routing_heatmap.png`, `artifacts/results/task3_routing_weights.csv` | Identity weight grows as damage gets milder; salt -> salt expert |
| Dominant vs distributed examples | `docs/figures/task3_dominant_vs_distributed.png` | One expert vs blended weights |
| Expert health | `artifacts/results/task3_expert_health.json` | No inactive expert; identity's use on corrupted inputs is beneficial blending |
| Re-check of Task 2's misrouted inputs | `artifacts/results/task3_task2_misrouted_recheck.json` | Better on 95%; SSIM 0.891 -> 0.974 |
| Examples / failures | `docs/figures/task3_examples.png`, `task3_failures.png` | |

## Task 4: Face-to-Sketch Generator

| Item | File | Say |
|---|---|---|
| Optuna | `artifacts/results/task4_optuna_summary.json`, `docs/figures/task4_optuna_*.png` | 20 trials: 19 completed, 1 pruned, 0 failed; best generator = base 64 |
| Losses (D real, D fake, G adv, G L1) + val SSIM per style | `docs/figures/task4_curves.png`, `artifacts/results/task4_history.csv` | Losses logged separately; best val at epoch 55 |
| Fixed validation samples over time | MLflow `task4-face-to-sketch` -> run `final-train` -> `samples/epoch_*.png` | Same 6 validation photos every 10 epochs |
| Test table (per style and photo source) | `artifacts/results/task4_test_table.tex` | SSIM 0.498, L1 0.099; style 2 hardest; confound by photo source |
| Same photo in all 3 styles | `docs/figures/task4_examples.png` | Style embedding clearly changes the sketch (style effect 0.11) |
| Failures | `docs/figures/task4_failures.png` | |

## ONNX and application

| Item | File | Say |
|---|---|---|
| ONNX sizes | `artifacts/results/onnx_sizes.csv` | 7 models; the Task 3 file contains the whole mixture pipeline as one graph |
| ONNX vs PyTorch | `artifacts/results/onnx_verification.csv` | All pass; max abs diff <= 2.3e-5 on real images |
| SSIM implementation check | `artifacts/results/ssim_check.json` | Matches scikit-image (max diff 1.9e-5) |
| Timing benchmark | `artifacts/results/benchmark_epoch.json` | Basis of the training time estimates |
| API | `docs/api_contract.md` | Endpoints, fields, error codes |
| Architecture of the app | `docker-compose.yml`, `README.md` | Browser -> nginx (React) -> /api -> FastAPI -> ONNX Runtime (CPU); models mounted read-only |

## Still to produce (student)

- App screenshots of the four workspaces (http://localhost:8080 after `docker compose up`).
- Google Stitch screens as design evidence (`Stitch Screens/`).
- Architecture diagrams (autoencoder, classifier, soft-MoE, U-Net + PatchGAN).
- MLflow screenshots (experiment list, an Optuna parent run with nested trials, the Task 4 sample images).
