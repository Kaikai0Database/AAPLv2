# AAP Predictor

> **Anti-Antimicrobial Peptide (AAP) prediction** using ESM2 protein language model + CNN1D+Linear classifier.

This repository provides **inference-only** scripts to predict whether a given peptide sequence is an Anti-Antimicrobial Peptide (AAP / non-AAP) using our pre-trained models.

---

## 🧬 Model Overview

| Component | Details |
|-----------|---------|
| **Backbone** | ESM2-35M (`esm2_t12_35M_UR50D`) — auto-downloaded |
| **Classifier** | CNN1D+Linear (best), with ablation variants |
| **Input** | Amino acid sequences (FASTA or CSV) |
| **Output** | Prediction score (0–1) + binary label |

### Available Model Variants

| Model Type | Description |
|------------|-------------|
| `CNN1D+Linear` | Full model — **recommended** |
| `CNN1D+Linear_NoConv2` | Ablation: no 2nd convolution block |
| `CNN1D+Linear_NoFC1` | Ablation: no FC1 + Dropout |
| `CNN1D+Linear_NoBN` | Ablation: no Batch Normalization |

---

## 📦 Download Pre-trained Weights

The classifier weights (`cls.pt`) for each model variant are hosted on Hugging Face:

> 🤗 **[Download from Hugging Face — TODO: add link after upload]**

After downloading, place the files as follows:

```
AAP/
└── pretrained_weights/
    ├── CNN1D+Linear/
    │   └── cls.pt
    ├── CNN1D+Linear_NoConv2/
    │   └── cls.pt
    ├── CNN1D+Linear_NoFC1/
    │   └── cls.pt
    └── CNN1D+Linear_NoBN/
        └── cls.pt
```

> **Note:** The ESM2 backbone is **automatically downloaded** by the `esm` library on first run (cached to `~/.cache/torch/hub/`). No manual download required.

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, CPU is supported)

### Conda (recommended)

```bash
conda env create -f env_install/environment.yml
conda activate aap
```

### pip

```bash
pip install -r env_install/package-list.txt
```

---

## 🚀 Usage

### Input from FASTA

```bash
python predict.py \
  --input example_input/example.fasta \
  --cls_model pretrained_weights/CNN1D+Linear/cls.pt
```

You will be prompted to specify the label for sequences in the FASTA file:

```
📄 FASTA file detected: example_input/example.fasta
   Found 10 sequence(s).

   Please specify the class label for ALL sequences in this file.
   Is this file POSITIVE (AAP) or NEGATIVE (non-AAP)? Enter [ 1 / 0 / skip ]: 1
   ✅ Label set to 1 (Positive / AAP)
```

### Input from CSV

Your CSV must contain a `seq` column. A `label` column is optional.

```bash
python predict.py \
  --input my_sequences.csv \
  --cls_model pretrained_weights/CNN1D+Linear/cls.pt
```

### Full Options

```bash
python predict.py \
  --input      example_input/example.fasta \   # Input file (.fasta, .fa, or .csv)
  --cls_model  pretrained_weights/CNN1D+Linear/cls.pt \  # Classifier weights
  --model_type CNN1D+Linear \                  # Model variant (default: CNN1D+Linear)
  --esm_type   t12 \                           # ESM2 backbone (default: t12)
  --threshold  0.5 \                           # Decision threshold (default: 0.5)
  --batch_size 16 \                            # Batch size (default: 16)
  --output     prediction_result.csv           # Output CSV path
```

---

## 📄 Output Format

The output CSV contains the original columns plus:

| Column | Description |
|--------|-------------|
| `prediction_score` | Model confidence (0.0 – 1.0) |
| `predicted_label` | Binary prediction: `1` = AAP, `0` = non-AAP |

Example output:

```
seq,label,prediction_score,predicted_label
ARPAKAAATQKKVERKAPDA,1,0.8231,1
DFKLFAVYIKYR,1,0.7654,1
CELDENNTPMC,1,0.3012,0
```

---

## 📁 Repository Structure

```
AAP/
├── predict.py                  ← Main entry point (CLI)
├── mainclass/
│   ├── model.py                ← Shared model architecture
│   └── predictor.py            ← Inference engine
├── example_input/
│   └── example.fasta           ← Example FASTA input (10 positive sequences)
├── env_install/
│   ├── environment.yml         ← Conda environment
│   └── package-list.txt        ← pip package list
└── README.md
```

---

## 📖 Citation

If you find this repository or our research useful, please cite our paper:

### APA Format
Datta, S., Yu, J. C., Lin, Y. H., Cheng, Y. C., & Chen, C. T. (2024). AMPpred-Web: a web-based platform integrating physicochemical and compositional features for antimicrobial peptide prediction. *Scientific Reports*, *14*(1), 14510. https://doi.org/10.1038/s41598-024-65062-9

### BibTeX
```bibtex
@article{datta2024amppred,
  author    = {Datta, Saptashwa and Yu, Jen-Chieh and Lin, Yi-Hsiang and Cheng, Yun-Chen and Chen, Ching-Tai},
  title     = {AMPpred-Web: a web-based platform integrating physicochemical and compositional features for antimicrobial peptide prediction},
  journal   = {Scientific Reports},
  year      = {2024},
  volume    = {14},
  number    = {1},
  pages     = {14510},
  doi       = {10.1038/s41598-024-65062-9},
  url       = {https://doi.org/10.1038/s41598-024-65062-9},
  publisher = {Nature Publishing Group}
}
```

---

## 📬 Contact

For questions or issues, please open a GitHub Issue or contact the authors.
