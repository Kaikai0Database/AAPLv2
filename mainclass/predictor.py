"""
predictor.py
------------
Pure inference engine for AAP prediction.
Does NOT depend on loss.xlsx or any training artifacts.
Loads cls.pt directly from a user-specified path.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from mainclass.model import multiClassifier, ProteinDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# ESM2 config table
ESM_CONFIG = {
    't6':  {'fn': 'esm2_t6_8M_UR50D',   'layer': 6,  'dim': 320},
    't12': {'fn': 'esm2_t12_35M_UR50D',  'layer': 12, 'dim': 480},
    't30': {'fn': 'esm2_t30_150M_UR50D', 'layer': 30, 'dim': 640},
    't33': {'fn': 'esm2_t33_650M_UR50D', 'layer': 33, 'dim': 1280},
}


class AAPPredictor:
    """
    Loads a pre-trained ESM2 backbone (auto-downloaded) and a classifier checkpoint (cls.pt),
    then runs inference on a DataFrame containing a 'seq' column.

    Parameters
    ----------
    cls_path : str
        Path to the classifier weights file (cls.pt).
    model_type : str
        Architecture type. One of:
        'CNN1D+Linear', 'CNN1D+Linear_NoConv2', 'CNN1D+Linear_NoFC1',
        'CNN1D+Linear_NoBN', 'Linear', 'CNN1D', 'CNN2D', 'CNN2D+Linear', 'CNN2D+LSTM', 'LSTM'
    esm_type : str
        ESM2 variant to use as backbone. One of: 't6', 't12', 't30', 't33'. Default: 't12'.
    padding_num : int
        Max sequence length (number of tokens) after padding. Default: 70.
    batch_size : int
        Inference batch size. Default: 16.
    threshold : float
        Decision threshold for binary classification. Default: 0.5.
    """

    SUPPORTED_MODELS = [
        'CNN1D+Linear', 'CNN1D+Linear_NoConv2', 'CNN1D+Linear_NoFC1',
        'CNN1D+Linear_NoBN', 'Linear', 'CNN1D', 'CNN2D', 'CNN2D+Linear',
        'CNN2D+LSTM', 'LSTM'
    ]

    def __init__(
        self,
        cls_path: str,
        model_type: str = 'CNN1D+Linear',
        esm_type: str = 't12',
        padding_num: int = 70,
        batch_size: int = 16,
        threshold: float = 0.5,
    ):
        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        if esm_type not in ESM_CONFIG:
            raise ValueError(f"Unsupported esm_type '{esm_type}'. Choose from: {list(ESM_CONFIG.keys())}")
        if not os.path.isfile(cls_path):
            raise FileNotFoundError(f"Classifier weights not found: {cls_path}")

        self.model_type = model_type
        self.esm_type = esm_type
        self.padding_num = padding_num
        self.batch_size = batch_size
        self.threshold = threshold

        cfg = ESM_CONFIG[esm_type]
        self.esm_layer = cfg['layer']
        self.input_dim = cfg['dim']

        print(f"🔄 Loading ESM2 backbone ({esm_type}) — auto-downloading if not cached...")
        import esm as esm_lib
        esm_fn = getattr(esm_lib.pretrained, cfg['fn'])
        self.esm_model, alphabet = esm_fn()
        self.esm_model = self.esm_model.to(device)
        self.esm_model.eval()
        self.batch_converter = alphabet.get_batch_converter()
        print(f"✅ ESM2 backbone loaded.")

        print(f"🔄 Loading classifier: {cls_path}")
        self.classifier = multiClassifier(
            model_type=model_type,
            input_dim=self.input_dim,
            num_labels=1,
            num_layers=2,
            kernel_size=(3, 3),
            dropout=0.6,
        ).to(device)
        self.classifier.load_state_dict(
            torch.load(cls_path, map_location=device)
        )
        self.classifier.eval()
        print(f"✅ Classifier loaded ({model_type}).")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run inference on a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'seq' column with amino acid sequences.
            Optionally contains a 'label' column (used for reporting only).

        Returns
        -------
        pd.DataFrame
            Original df with added columns:
            - prediction_score : float  (sigmoid output, 0–1)
            - predicted_label  : int    (0 or 1 based on threshold)
        """
        has_label = 'label' in df.columns

        # Fill label column with -1 if not present (inference-only mode)
        work_df = df.copy()
        if not has_label:
            work_df['label'] = -1

        dataset = ProteinDataset(
            work_df,
            self.batch_converter,
            padding=True,
            paddingNumber=self.padding_num,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)

        all_scores = []
        all_seqs = []

        with torch.no_grad():
            for seq_strs, batch_tokens, _labels in loader:
                batch_tokens = torch.squeeze(batch_tokens, 1).to(device)
                results = self.esm_model(batch_tokens, repr_layers=[self.esm_layer])
                token_repr = results["representations"][self.esm_layer]

                out = self.classifier(token_repr, return_features=False)

                # Unified output handling (logits or probabilities depending on model type)
                if isinstance(out, tuple):
                    logits, _ = out
                    scores = torch.sigmoid(logits).view(-1)
                else:
                    # Models that already apply sigmoid return probabilities directly;
                    # models without sigmoid (CNN1D+Linear family, Linear) return logits.
                    # Apply sigmoid defensively — idempotent if already in [0,1].
                    scores = torch.sigmoid(out).view(-1)

                all_scores.extend(scores.cpu().numpy().tolist())
                all_seqs.extend(seq_strs)

        result_df = df.copy()
        result_df['prediction_score'] = [round(s, 4) for s in all_scores]
        result_df['predicted_label'] = [1 if s >= self.threshold else 0 for s in all_scores]
        return result_df
