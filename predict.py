"""
predict.py
----------
Main entry point for AAP (Anti-Antimicrobial Peptide) prediction.

Supports both FASTA and CSV input formats.
- FASTA: You will be prompted to specify whether sequences are positive (1) or negative (0).
- CSV  : Must contain a 'seq' column. If 'label' column is absent, predictions are still made.

Usage examples
--------------
# Predict from FASTA (will prompt for label)
python predict.py --input example_input/example.fasta --cls_model pretrained_weights/CNN1D+Linear/cls.pt

# Predict from CSV
python predict.py --input my_sequences.csv --cls_model pretrained_weights/CNN1D+Linear/cls.pt

# Specify model type and output path
python predict.py --input example_input/example.fasta \\
                  --cls_model pretrained_weights/CNN1D+Linear/cls.pt \\
                  --model_type CNN1D+Linear \\
                  --esm_type t12 \\
                  --output results/my_prediction.csv
"""

import argparse
import os
import sys
import pandas as pd

from mainclass.predictor import AAPPredictor

# ──────────────────────────────────────────────
# FASTA Parser
# ──────────────────────────────────────────────

def parse_fasta(filepath: str) -> list[dict]:
    """
    Parse a FASTA file into a list of {'id': ..., 'seq': ...} dicts.
    Handles multi-line sequences and skips blank lines.
    """
    records = []
    current_id = None
    current_seq = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_id is not None:
                    records.append({'id': current_id, 'seq': ''.join(current_seq)})
                current_id = line[1:]  # remove '>'
                current_seq = []
            else:
                current_seq.append(line)

    # Don't forget the last record
    if current_id is not None and current_seq:
        records.append({'id': current_id, 'seq': ''.join(current_seq)})

    return records


def fasta_to_dataframe(filepath: str) -> pd.DataFrame:
    """
    Convert a FASTA file to a DataFrame with columns: id, seq, label.
    Prompts the user to specify the label (pos=1 / neg=0 / skip).
    """
    records = parse_fasta(filepath)
    if not records:
        print(f"❌ No sequences found in: {filepath}")
        sys.exit(1)

    print(f"\n📄 FASTA file detected: {filepath}")
    print(f"   Found {len(records)} sequence(s).")
    print()
    print("   Please specify the class label for ALL sequences in this file.")
    print("   This label is used for result tracking only — it does not affect the prediction score.")
    print()

    while True:
        raw = input("   Is this file POSITIVE (AMP) or NEGATIVE (non-AMP)? Enter [ 1 / 0 / skip ]: ").strip().lower()
        if raw == '1':
            label = 1
            print(f"   ✅ Label set to 1 (Positive / AMP)\n")
            break
        elif raw == '0':
            label = 0
            print(f"   ✅ Label set to 0 (Negative / non-AMP)\n")
            break
        elif raw == 'skip':
            label = -1
            print(f"   ⚠️  Label skipped. 'label' column will be set to -1.\n")
            break
        else:
            print("   ⚠️  Invalid input. Please enter 1, 0, or skip.")

    df = pd.DataFrame(records)
    df['label'] = label
    return df


# ──────────────────────────────────────────────
# CSV Loader
# ──────────────────────────────────────────────

def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and validate required columns."""
    df = pd.read_csv(filepath)
    if 'seq' not in df.columns:
        print(f"❌ CSV file must contain a 'seq' column. Found columns: {list(df.columns)}")
        sys.exit(1)
    if 'label' not in df.columns:
        print("   ⚠️  No 'label' column found in CSV. Predictions will be made without ground-truth labels.")
        df['label'] = -1
    return df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

SUPPORTED_MODELS = [
    'CNN1D+Linear', 'CNN1D+Linear_NoConv2', 'CNN1D+Linear_NoFC1',
    'CNN1D+Linear_NoBN', 'Linear', 'CNN1D', 'CNN2D', 'CNN2D+Linear',
    'CNN2D+LSTM', 'LSTM'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="AAP Predictor — Anti-Antimicrobial Peptide prediction using ESM2 + CNN1D+Linear",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to input file. Accepted formats:\n'
             '  .fasta / .fa  — FASTA format (will prompt for label)\n'
             '  .csv          — CSV with a "seq" column (and optionally "label")'
    )
    parser.add_argument(
        '--cls_model', '-c', required=True,
        help='Path to classifier weights (cls.pt).\n'
             'Download from Hugging Face: see README.md for the link.'
    )
    parser.add_argument(
        '--model_type', '-m', default='CNN1D+Linear',
        choices=SUPPORTED_MODELS,
        help=f'Model architecture type (default: CNN1D+Linear).\nChoices: {SUPPORTED_MODELS}'
    )
    parser.add_argument(
        '--esm_type', '-e', default='t12',
        choices=['t6', 't12', 't30', 't33'],
        help='ESM2 backbone variant (default: t12 = ESM2-35M).\n'
             'The model will be auto-downloaded on first use.'
    )
    parser.add_argument(
        '--padding_num', type=int, default=70,
        help='Max token length for padding (default: 70).'
    )
    parser.add_argument(
        '--batch_size', type=int, default=16,
        help='Inference batch size (default: 16).'
    )
    parser.add_argument(
        '--threshold', type=float, default=0.5,
        help='Decision threshold for binary prediction (default: 0.5).'
    )
    parser.add_argument(
        '--output', '-o', default='prediction_result.csv',
        help='Output CSV path (default: prediction_result.csv).'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. Load input ──────────────────────────────────────
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.fasta', '.fa'):
        df = fasta_to_dataframe(args.input)
    elif ext == '.csv':
        df = load_csv(args.input)
    else:
        print(f"❌ Unsupported file format: '{ext}'. Please use .fasta, .fa, or .csv")
        sys.exit(1)

    print(f"📊 Input loaded: {len(df)} sequence(s)\n")

    # ── 2. Load predictor ──────────────────────────────────
    predictor = AAPPredictor(
        cls_path=args.cls_model,
        model_type=args.model_type,
        esm_type=args.esm_type,
        padding_num=args.padding_num,
        batch_size=args.batch_size,
        threshold=args.threshold,
    )

    # ── 3. Run inference ───────────────────────────────────
    print("\n🚀 Running inference...")
    result_df = predictor.predict(df)

    # ── 4. Print summary ───────────────────────────────────
    total = len(result_df)
    n_pos = (result_df['predicted_label'] == 1).sum()
    n_neg = (result_df['predicted_label'] == 0).sum()
    print(f"\n{'─'*50}")
    print(f"  Prediction Summary")
    print(f"{'─'*50}")
    print(f"  Total sequences  : {total}")
    print(f"  Predicted AMP    : {n_pos}  ({100*n_pos/total:.1f}%)")
    print(f"  Predicted non-AMP: {n_neg}  ({100*n_neg/total:.1f}%)")
    print(f"  Threshold used   : {args.threshold}")
    print(f"{'─'*50}\n")

    # ── 5. Save output ─────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    result_df.to_csv(args.output, index=False)
    print(f"✅ Results saved to: {args.output}\n")


if __name__ == '__main__':
    main()
