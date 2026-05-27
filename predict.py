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

def parseFasta(filepath: str) -> list[dict]:
    """
    Parse a FASTA file into a list of {'id': ..., 'seq': ...} dicts.
    Handles multi-line sequences and skips blank lines.
    """
    records = []
    currentId = None
    currentSeq = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if currentId is not None:
                    records.append({'id': currentId, 'seq': ''.join(currentSeq)})
                currentId = line[1:]  # remove '>'
                currentSeq = []
            else:
                currentSeq.append(line)

    # Don't forget the last record
    if currentId is not None and currentSeq:
        records.append({'id': currentId, 'seq': ''.join(currentSeq)})

    return records


def fastaToDataframe(filepath: str) -> pd.DataFrame:
    """
    Convert a FASTA file to a DataFrame with columns: id, seq, label.
    Prompts the user to specify the label (pos=1 / neg=0 / skip).
    """
    records = parseFasta(filepath)
    if not records:
        print(f"[Error] No sequences found in: {filepath}")
        sys.exit(1)

    print(f"\n[Info] FASTA file detected: {filepath}")
    print(f"   Found {len(records)} sequence(s).")
    print()
    print("   Please specify the class label for ALL sequences in this file.")
    print("   This label is used for result tracking only -- it does not affect the prediction score.")
    print()

    while True:
        raw = input("   Is this file POSITIVE (AMP) or NEGATIVE (non-AMP)? Enter [ 1 / 0 / skip ]: ").strip().lower()
        if raw == '1':
            label = 1
            print(f"   [OK] Label set to 1 (Positive / AMP)\n")
            break
        elif raw == '0':
            label = 0
            print(f"   [OK] Label set to 0 (Negative / non-AMP)\n")
            break
        elif raw == 'skip':
            label = -1
            print(f"   [Warning] Label skipped. 'label' column will be set to -1.\n")
            break
        else:
            print("   [Warning] Invalid input. Please enter 1, 0, or skip.")

    df = pd.DataFrame(records)
    df['label'] = label
    return df


# ──────────────────────────────────────────────
# CSV Loader
# ──────────────────────────────────────────────

def loadCsv(filepath: str) -> pd.DataFrame:
    """Load a CSV file and validate required columns."""
    df = pd.read_csv(filepath)
    if 'seq' not in df.columns:
        print(f"[Error] CSV file must contain a 'seq' column. Found columns: {list(df.columns)}")
        sys.exit(1)
    if 'label' not in df.columns:
        print("   [Warning] No 'label' column found in CSV. Predictions will be made without ground-truth labels.")
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


def parseArgs():
    parser = argparse.ArgumentParser(
        description="AAP Predictor -- Anti-Antimicrobial Peptide prediction using ESM2 + CNN1D+Linear",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        '--input', '-i', required=True,
        help='Path to input file. Accepted formats:\n'
             '  .fasta / .fa  -- FASTA format (will prompt for label)\n'
             '  .csv          -- CSV with a "seq" column (and optionally "label")'
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
    args = parseArgs()

    # -- 1. Load input --------------------------------------------------
    ext = os.path.splitext(args.input)[1].lower()
    if ext in ('.fasta', '.fa'):
        df = fastaToDataframe(args.input)
    elif ext == '.csv':
        df = loadCsv(args.input)
    else:
        print(f"[Error] Unsupported file format: '{ext}'. Please use .fasta, .fa, or .csv")
        sys.exit(1)

    print(f"[Info] Input loaded: {len(df)} sequence(s)\n")

    # -- 2. Load predictor ----------------------------------------------
    predictor = AAPPredictor(
        clsPath=args.cls_model,
        modelType=args.model_type,
        esmType=args.esm_type,
        paddingNum=args.padding_num,
        batchSize=args.batch_size,
        threshold=args.threshold,
    )

    # -- 3. Run inference -----------------------------------------------
    print("\n[Info] Running inference...")
    resultDf = predictor.predict(df)

    # -- 4. Print summary -----------------------------------------------
    total = len(resultDf)
    nPos = (resultDf['predicted_label'] == 1).sum()
    nNeg = (resultDf['predicted_label'] == 0).sum()
    print(f"\n{'─'*50}")
    print(f"  Prediction Summary")
    print(f"{'─'*50}")
    print(f"  Total sequences  : {total}")
    print(f"  Predicted AMP    : {nPos}  ({100*nPos/total:.1f}%)")
    print(f"  Predicted non-AMP: {nNeg}  ({100*nNeg/total:.1f}%)")
    print(f"  Threshold used   : {args.threshold}")
    print(f"{'─'*50}\n")

    # -- 5. Save output -------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    resultDf.to_csv(args.output, index=False)
    print(f"[OK] Results saved to: {args.output}\n")


if __name__ == '__main__':
    main()
