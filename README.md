# Combining Diffusion-Based and Traditional Augmentation for Data-Scarce Visual Classification

Code for the experiments in the paper *Combining Diffusion-Based and Traditional
Augmentation for Data-Scarce Visual Classification* (submitted to Multimedia Tools
and Applications, 2026).

The framework augments small image-classification datasets by combining
**diffusion-based synthesis** (FLUX.2 [klein] with a per-class LoRA adapter,
reference-guided generation) with **traditional geometric/photometric
augmentation**, and trains an **EfficientNetV2B0 classifier from scratch** to
isolate the effect of augmentation. The central finding is that generative and
traditional augmentation act on different levels of the data distribution and are
**complementary**.

## Repository contents
- `efficientnetv2b0_kfold_runner.py` — main k-fold training/evaluation runner.
- `run_all_experiments.py`, `run_all_efficientnetv2b0_variants.py`,
  `run_all_experiments_supervisor.py`, `run_rerun_queue.py` — experiment drivers.
- `build_controlled_mix_dataset.py` — builds the augmented (mixed) datasets at the
  controlled synthetic-to-real ratios (2:1, 5:1, 10:1).
- `scripts/` — reporting and analysis: `generate_reports.py`,
  `gradcam_compare_experiments.py` (Grad-CAM), `preview_mix_dataset.py`.
- `*.ipynb` — notebooks per augmentation strategy (original / traditional /
  generative / generative+traditional).
- `experiment_suites/` — experiment configurations.
- `requirements_tf.txt`, `requirements_tf_wsl_gpu.txt` — Python dependencies
  (TensorFlow; the WSL/GPU variant for CUDA).

## Reproducing
1. Install dependencies: `pip install -r requirements_tf.txt` (or
   `requirements_tf_wsl_gpu.txt` for GPU on WSL).
2. Prepare datasets (see **Data** below) and build the augmented sets with
   `build_controlled_mix_dataset.py`.
3. Run the experiments with `run_all_experiments.py` (or the per-variant /
   supervisor drivers).

## Data
- **Synthetic images and derived artifacts** (accuracy tables, FID/IS, reports):
  archived on Zenodo — DOI: *to be added*.
- **Real architectural images**: sourced from third-party websites and therefore
  **not redistributed here**; their sources are described in the Zenodo record.
- **Generalist (ImageNet) subset**: the ten ImageNet validation classes used are
  listed in the paper; images are governed by the ImageNet terms of use and are not
  redistributed.

Diffusion generation was performed with FLUX.2 [klein] via ComfyUI (not included).

## Citation
> G. H. M. Dias, Y. de A. M. Barbosa, T. G. do Rêgo. Combining Diffusion-Based and
> Traditional Augmentation for Data-Scarce Visual Classification. Multimedia Tools
> and Applications, 2026 (under review).

## License
MIT — see [LICENSE](LICENSE).
