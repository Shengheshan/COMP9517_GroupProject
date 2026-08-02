# COMP9517 Group Project — iNaturalist Species Classification

## 1. Project overview

This project compares traditional computer-vision methods and deep-learning methods for fine-grained species classification on a selected 500-class subset of the iNaturalist 2021 dataset.

The repository contains three implementation streams developed in parallel:

1. **Traditional machine learning:** HOG / colour features with an SVM classifier.
2. **ResNet-18 trained from scratch:** no ImageNet pretraining.
3. **Transfer learning with ResNet-18:** frozen-backbone training and full-network fine-tuning.

Because the streams were implemented independently, some directory and file names are not perfectly symmetrical. We intentionally preserve the current names to avoid breaking existing imports, command-line scripts, checkpoint paths, and result-generation dependencies. The mapping below defines the intended project structure.

---

## 2. Model-to-directory mapping

| Method used in the report | Implementation location | Description |
|---|---|---|
| **HOG + SVM** | `src/traditional/` | Traditional baseline using HOG features and a linear SVM. |
| **HOG + Colour + SVM** | `src/traditional/` | Improved traditional model combining HOG and colour features. |
| **ResNet-18 Scratch** | `src/models/`, `src/training/`, and `src/evaluation/predict_resnet18_scratch*.py` | ResNet-18 trained without pretrained ImageNet weights. |
| **ResNet-18 Frozen** | `src/deep_learning/` with `--mode frozen` | ImageNet-pretrained ResNet-18 with the backbone frozen and only the final classifier trained. |
| **ResNet-18 Fine-tuned** | `src/deep_learning/` with `--mode finetune` | ImageNet-pretrained ResNet-18 with the full network fine-tuned. |

### Important naming clarification

`src/deep_learning/` does **not** represent all deep-learning code in the repository. It specifically contains the shared **transfer-learning pipeline** for both the Frozen and Fine-tuned experiments.

The Scratch implementation is stored separately because it uses a custom model definition and a different training pipeline:

```text
src/models/resnet18_scratch.py
src/training/train_resnet18_scratch.py
src/training/train_resnet18_scratch_aligned.py
src/evaluation/predict_resnet18_scratch.py
src/evaluation/predict_resnet18_scratch_aligned.py
```

---

## 3. Final and preliminary experiment names

Some experiments have both an original version and an `aligned` version.

| Repository experiment name | Interpretation | Recommended use |
|---|---|---|
| `hog_svm` | Initial HOG-only traditional baseline | Baseline / preliminary comparison |
| `hog_color_C0.1` | Selected HOG + colour SVM configuration | Final traditional model |
| `resnet18_scratch` | Initial Scratch ResNet-18 experiment | Preliminary result |
| `resnet18_scratch_aligned` | Scratch ResNet-18 using the aligned experimental protocol | Final Scratch result |
| `resnet18_frozen_aligned` | Frozen-backbone transfer-learning experiment | Final Frozen result |
| `resnet18_finetune_aligned` | Full-network transfer-learning experiment | Final Fine-tuned result |

For the main comparison in the report, the recommended experiments are:

```text
hog_color_C0.1
resnet18_scratch_aligned
resnet18_frozen_aligned
resnet18_finetune_aligned
```

The word `aligned` indicates that the corresponding experiments were adjusted to use a more comparable training and evaluation protocol. Existing filenames are retained because they are referenced by scripts and result files.

---

## 4. Repository structure

```text
COMP9517-Group-Project/
├── configs/                    # HOG/SVM feature and hyperparameter configurations
├── data/                       # Raw-data placeholders and dataset documentation
├── data_metadata/              # Shared class mapping and train/validation/test splits
├── src/
│   ├── data/                   # Shared dataset preparation, loading, and validation
│   ├── traditional/            # HOG/colour feature extraction, SVM training, and evaluation
│   ├── models/                 # Scratch ResNet-18 model definition
│   ├── training/               # Scratch ResNet-18 training scripts
│   ├── deep_learning/          # Frozen and Fine-tuned transfer-learning pipeline
│   ├── evaluation/             # Shared metrics and prediction/evaluation utilities
│   └── utils/                  # Reserved shared utilities
├── results/
│   ├── predictions/            # Standardised test-set prediction CSV files
│   ├── evaluation/             # Shared metrics, timings, and confusion-matrix arrays
│   ├── training/               # Scratch training histories
│   ├── logs/                   # Frozen and Fine-tuned training histories and summaries
│   ├── tables/                 # Traditional-model validation and hyperparameter-search results
│   ├── figures/                # Final report figures organised by model or comparison
│   └── models/                 # Placeholder for model artefacts; large checkpoints are not tracked
├── report/                     # Final report material
└── presentation/               # Presentation and video material
```

---

## 5. Result-directory mapping

The result directories reflect the original implementation streams. Files are not moved because several scripts depend on the current paths.

| Result directory | Contents | Models covered |
|---|---|---|
| `results/predictions/` | Standardised test predictions | All submitted models |
| `results/evaluation/` | Final metrics, timing records, and confusion matrices | All submitted models |
| `results/training/` | Per-epoch training history | Scratch models |
| `results/logs/` | Per-epoch history and summaries | Frozen and Fine-tuned models |
| `results/tables/` | Validation runs, feature ablations, and SVM hyperparameter searches | Traditional HOG/SVM models |
| `results/figures/` | Generated figures used in the report or presentation | Organised by model after generation |
| `results/models/` | Placeholder for trained model artefacts | Checkpoints are excluded from Git because of size |

### Why both `results/training/` and `results/logs/` exist

Both directories contain training-process records, but they were created by different implementation streams:

- Scratch scripts write history files to `results/training/`.
- Frozen and Fine-tuned scripts write history files to `results/logs/`.

They serve similar purposes. They remain separate only to preserve existing path dependencies.

### Why `results/tables/` mainly contains HOG/SVM files

The traditional-model pipeline records each validation and hyperparameter-search run as a separate JSON file. These files represent SVM parameter tuning and feature-ablation experiments; they are not the final cross-model evaluation table.

Final comparable results for all models are stored in `results/evaluation/`.

---

## 6. Shared dataset split

All model streams use the shared dataset metadata under:

```text
data_metadata/train.csv
data_metadata/validation.csv
data_metadata/test.csv
data_metadata/category_to_label.json
data_metadata/label_to_category.json
```

This ensures that the final models are compared on the same selected classes and the same held-out test split.

Dataset-related utilities are located in:

```text
src/data/create_splits.py
src/data/dataset.py
src/data/extract_subset.py
src/data/validate_dataset.py
```

---

## 7. Unified final evaluation

All final prediction files use a shared CSV format and are stored in:

```text
results/predictions/
```

The common evaluation entry point is:

```text
src/evaluation/evaluate_predictions.py
```

The shared metric implementation is:

```text
src/evaluation/metrics.py
```

The unified evaluation reports:

- Top-1 accuracy
- Top-5 accuracy
- Macro precision
- Macro recall
- Macro F1-score
- Confusion matrix

Final comparable outputs are stored in:

```text
results/evaluation/
```

This means that the training implementations differ by model type, but the final test-set evaluation is standardised across all models.

---

## 8. Figures

Final report figures should be stored under `results/figures/`, using clearly named model subdirectories. For example:

```text
results/figures/
├── traditional/
├── scratch/
├── frozen/
├── finetune/
└── comparison/
```

Suggested figure types include:

- Training and validation loss curves
- Accuracy or Macro-F1 curves
- Confusion-matrix visualisations
- Traditional-model feature or hyperparameter comparisons
- Final cross-model performance comparison

The figure-directory names do not need to match the source-code directory names exactly, provided that each figure is clearly labelled.

---

## 9. Model checkpoints

Large trained model files are excluded from Git through `.gitignore`, including:

```text
*.pth
*.pt
*.ckpt
*.joblib
```

Different implementation streams currently use different local checkpoint paths:

| Model stream | Local checkpoint location |
|---|---|
| Traditional HOG/SVM | `results/models/` |
| Scratch ResNet-18 | `checkpoints/` |
| Frozen / Fine-tuned ResNet-18 | `models/` |

These locations are retained to avoid breaking existing scripts. The repository therefore contains code, prediction outputs, metrics, and compact training logs, but not large trained weights.

---

## 10. Reproducibility note

The repository prioritises preserving working experiment pipelines over renaming every file into a completely symmetrical structure. Renaming or moving the current modules would require changing imports, default paths, checkpoint references, plotting scripts, and previously generated outputs.

For this reason:

- Existing source filenames and paths are preserved.
- This README provides the canonical mapping between report terminology and repository implementation.
- Shared data splits and final evaluation outputs provide consistent cross-model comparison.
- New report figures should follow the unified `results/figures/` organisation described above.

