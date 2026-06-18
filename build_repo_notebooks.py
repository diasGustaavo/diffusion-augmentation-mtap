from __future__ import annotations

import json
from pathlib import Path

WORKSPACE_ROOT = Path("/home/guga/masters")
RUNNER_FILENAME = "efficientnetv2b0_kfold_runner.py"

VARIANT_SPECS = [
    {"variant_name": "Variante_A", "methodology_label": "Variante A", "results_label": "Variante D", "dense_units": [512, 256], "dropout_rate": 0.2},
    {"variant_name": "Variante_B", "methodology_label": "Variante B", "results_label": "Variante C", "dense_units": [256, 128], "dropout_rate": 0.2},
    {"variant_name": "Variante_C", "methodology_label": "Variante C", "results_label": "Variante E", "dense_units": [512, 256, 128], "dropout_rate": 0.2},
    {"variant_name": "Variante_D", "methodology_label": "Variante D", "results_label": "Variante F", "dense_units": [128], "dropout_rate": 0.2},
    {"variant_name": "Variante_E", "methodology_label": "Variante E", "results_label": "", "dense_units": [96], "dropout_rate": 0.2},
    {"variant_name": "Variante_G", "methodology_label": "Variante F", "results_label": "Variante G", "dense_units": [64], "dropout_rate": 0.0},
    {"variant_name": "Variante_H", "methodology_label": "Variante F", "results_label": "Variante H", "dense_units": [64], "dropout_rate": 0.2},
]


def markdown_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code_cell(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook_intro(title: str, dataset_root: str, augmentation_mode: str) -> str:
    runner_path = WORKSPACE_ROOT / RUNNER_FILENAME
    augmentation_label = (
        "sem aumento online adicional"
        if augmentation_mode == "none"
        else "com aumento tradicional online via ImageDataGenerator"
    )
    return (
        f"# {title}\n\n"
        "Notebook para treinamento em `TensorFlow` com `EfficientNetV2B0`, validação cruzada "
        "`StratifiedKFold (k=5)`, curvas de acurácia/loss, matriz de confusão e explicabilidade por `Grad-CAM`.\n\n"
        f"Esta versão usa o dataset em `{dataset_root}` e executa as variantes metodológicas A-H ({augmentation_label}).\n\n"
        "Hiperparâmetros globais configurados no notebook:\n\n"
        "- Resolução: `224x224`\n"
        "- Otimizador: `SGD`\n"
        "- Momentum: `0.9`\n"
        "- Weight decay: `5e-4`\n"
        "- Learning rate inicial: `1e-5`\n"
        "- Scheduler: `ReduceLROnPlateau`\n"
        "- Máximo de épocas: `300`\n"
        "- Early stopping patience: `30`\n"
        "- Batch size: `32`\n"
        "- Dropout especial da Variante H: `0.2`\n\n"
        "Os checkpoints de progresso ficam salvos no `output_dir` de cada variante. "
        "Se você interromper o treino e rodar de novo com a mesma pasta de saída, ele retoma automaticamente.\n\n"
        "Para execuções longas, prefira terminal/script em vez de manter o Jupyter aberto por horas, porque o notebook acumula saída e tende a ser menos estável em memória.\n\n"
        "Para abrir com a venv criada neste workspace, rode no WSL:\n\n"
        "```bash\n"
        "source /home/guga/masters/.venv_tf/bin/activate\n"
        "jupyter lab\n"
        "```\n\n"
        "Para um teste rápido de fumaça via terminal, use:\n\n"
        "```bash\n"
        f"/home/guga/masters/.venv_tf/bin/python {runner_path} --dataset-root {dataset_root} --smoke-test --variant-name Variante_D --dense-units 128 --augmentation-mode {augmentation_mode}\n"
        "```\n"
    )


def config_cell_source(dataset_root: str, default_output_dir: str, augmentation_mode: str) -> str:
    variant_specs_json = json.dumps(VARIANT_SPECS, indent=4, ensure_ascii=True)
    return f"""from pathlib import Path
import json
import os

from efficientnetv2b0_kfold_runner import PipelineConfig, apply_smoke_test_defaults, run_pipeline

DATASET_ROOT = Path(r"{dataset_root}")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", r"{default_output_dir}"))
SMOKE_TEST = os.environ.get("SMOKE_TEST", "0") == "1"
USE_IMAGENET_WEIGHTS = os.environ.get("USE_IMAGENET_WEIGHTS", "1") == "1"
UNFREEZE_BASE = os.environ.get("UNFREEZE_BASE", "0") == "1"
RESUME_TRAINING = os.environ.get("RESUME_TRAINING", "1") == "1"

COMMON_CONFIG = {{
    "image_size": 224,
    "batch_size": 32,
    "epochs": 300,
    "learning_rate": 1e-5,
    "n_splits": 5,
    "early_stopping_patience": 30,
    "early_stopping_min_delta": 1e-4,
    "optimizer_name": "sgd",
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "reduce_lr_factor": 0.5,
    "reduce_lr_patience": 10,
    "reduce_lr_min_lr": 1e-7,
    "reduce_lr_min_delta": 1e-4,
    "augmentation_mode": "{augmentation_mode}",
    "augmentation_rotation_deg": 5.0,
    "augmentation_width_shift": 0.1,
    "augmentation_height_shift": 0.1,
    "augmentation_shear": 0.1,
    "augmentation_zoom": 0.1,
    "augmentation_horizontal_flip": True,
    "augmentation_fill_mode": "nearest",
    "dataset_num_parallel_calls": 1,
    "dataset_prefetch_buffer": 1,
    "stop_on_min_lr_patience": 15,
    "resume_save_frequency_epochs": 25,
}}

VARIANT_SPECS = json.loads(r'''{variant_specs_json}''')
VARIANT_SPECS
"""


RUN_CELL_SOURCE = """all_artifacts = []

for variant in VARIANT_SPECS:
    variant_output_dir = OUTPUT_DIR / variant["variant_name"].lower()
    aliases = [variant["methodology_label"]]
    if variant.get("results_label"):
        aliases.append(variant["results_label"])

    config = PipelineConfig(
        dataset_root=str(DATASET_ROOT),
        output_dir=str(variant_output_dir),
        use_imagenet_weights=USE_IMAGENET_WEIGHTS,
        freeze_base=not UNFREEZE_BASE,
        dense_units=tuple(variant["dense_units"]),
        dropout_rate=float(variant["dropout_rate"]),
        variant_name=variant["variant_name"],
        variant_aliases=tuple(aliases),
        resume_training=RESUME_TRAINING,
        **COMMON_CONFIG,
    )

    if SMOKE_TEST:
        config = apply_smoke_test_defaults(config)

    print(f"\\n===== Executando {variant['variant_name']} =====")
    artifacts = run_pipeline(config)
    all_artifacts.append(
        {
            "variant_name": variant["variant_name"],
            "methodology_label": variant["methodology_label"],
            "results_label": variant["results_label"],
            "dense_units": variant["dense_units"],
            "dropout_rate": variant["dropout_rate"],
            "output_dir": str(variant_output_dir),
            "test_accuracy": artifacts["test_metrics"].get("accuracy"),
            "validation_accuracy": artifacts["validation_metrics"].get("accuracy"),
        }
    )

    if artifacts.get("interrupted"):
        print("\\nExecucao interrompida. Rode novamente com o mesmo OUTPUT_DIR para continuar.")
        break

all_artifacts
"""


def build_notebook(title: str, dataset_root: str, default_output_dir: str, augmentation_mode: str) -> dict:
    cells = [
        markdown_cell(notebook_intro(title, dataset_root, augmentation_mode), "intro-cell"),
        code_cell(config_cell_source(dataset_root, default_output_dir, augmentation_mode), "config-cell"),
        code_cell(RUN_CELL_SOURCE, "run-cell"),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    notebook_specs = [
        (
            "Treinamento EfficientNetV2B0 - Augmented",
            str(WORKSPACE_ROOT / "datasets" / "augmented"),
            str(WORKSPACE_ROOT / "outputs" / "augmented_efficientnetv2b0"),
            "none",
            WORKSPACE_ROOT / "augmented_efficientnetv2b0_kfold.ipynb",
        ),
        (
            "Treinamento EfficientNetV2B0 - Original",
            str(WORKSPACE_ROOT / "datasets" / "original"),
            str(WORKSPACE_ROOT / "outputs" / "original_efficientnetv2b0"),
            "none",
            WORKSPACE_ROOT / "original_efficientnetv2b0_kfold.ipynb",
        ),
        (
            "Treinamento EfficientNetV2B0 - Aumento Tradicional",
            str(WORKSPACE_ROOT / "datasets" / "original"),
            str(WORKSPACE_ROOT / "outputs" / "original_efficientnetv2b0_traditional"),
            "traditional",
            WORKSPACE_ROOT / "traditional_augmented_efficientnetv2b0_kfold.ipynb",
        ),
        (
            "Treinamento EfficientNetV2B0 - Generativo + Aumento Tradicional",
            str(WORKSPACE_ROOT / "datasets" / "augmented"),
            str(WORKSPACE_ROOT / "outputs" / "augmented_efficientnetv2b0_generative_plus_traditional"),
            "traditional",
            WORKSPACE_ROOT / "generative_plus_traditional_efficientnetv2b0_kfold.ipynb",
        ),
    ]

    for title, dataset_root, default_output_dir, augmentation_mode, output_path in notebook_specs:
        notebook = build_notebook(
            title=title,
            dataset_root=dataset_root,
            default_output_dir=default_output_dir,
            augmentation_mode=augmentation_mode,
        )
        output_path.write_text(json.dumps(notebook, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"Notebook gerado em {output_path}")


if __name__ == "__main__":
    main()
