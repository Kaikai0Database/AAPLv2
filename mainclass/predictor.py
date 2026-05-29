"""
predictor.py
------------
Pure inference engine for AAP prediction.
Does NOT depend on loss.xlsx or any training artifacts.
Loads cls.pt directly from a user-specified path.
"""

import torch
from torch.utils.data import DataLoader
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
    clsPath : str
        Path to the classifier weights file (cls.pt).
    modelType : str
        Architecture type. One of:
        'CNN1D+Linear', 'CNN1D+Linear_NoConv2', 'CNN1D+Linear_NoFC1',
        'CNN1D+Linear_NoBN', 'Linear', 'CNN1D', 'CNN2D', 'CNN2D+Linear', 'CNN2D+LSTM', 'LSTM'
    esmType : str
        ESM2 variant to use as backbone. One of: 't6', 't12', 't30', 't33'. Default: 't12'.
    paddingNum : int
        Max sequence length (number of tokens) after padding. Default: 70.
    batchSize : int
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
        clsPath: str,
        esmPath: str = None,
        modelType: str = 'CNN1D+Linear',
        esmType: str = 't12',
        paddingNum: int = 70,
        batchSize: int = 16,
        threshold: float = 0.5,
    ):
        if modelType not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported modelType '{modelType}'. "
                f"Choose from: {self.SUPPORTED_MODELS}"
            )
        if esmType not in ESM_CONFIG:
            raise ValueError(f"Unsupported esmType '{esmType}'. Choose from: {list(ESM_CONFIG.keys())}")
        if not os.path.isfile(clsPath):
            raise FileNotFoundError(f"Classifier weights not found: {clsPath}")

        self.modelType = modelType
        self.esmType = esmType
        self.paddingNum = paddingNum
        self.batchSize = batchSize
        self.threshold = threshold

        cfg = ESM_CONFIG[esmType]
        self.esmLayer = cfg['layer']
        self.inputDim = cfg['dim']

        print(f"[Info] Loading ESM2 backbone ({esmType}) -- auto-downloading if not cached...")
        import esm as esmLib
        esmFn = getattr(esmLib.pretrained, cfg['fn'])
        self.esmModel, alphabet = esmFn()
        if esmPath and esmPath.lower() != 'none':
            if not os.path.isfile(esmPath):
                raise FileNotFoundError(f"ESM weights not found at: {esmPath}")
            print(f"[Info] Loading fine-tuned ESM weights: {esmPath}")
            self.esmModel.load_state_dict(
                torch.load(esmPath, map_location=device)
            )
            print("[OK] Fine-tuned ESM weights loaded.")
        else:
            print("[Info] Using raw pretrained ESM2 backbone (no fine-tuning weights loaded).")
        self.esmModel = self.esmModel.to(device)
        self.esmModel.eval()
        self.batchConverter = alphabet.get_batch_converter()
        print(f"[OK] ESM2 backbone loaded.")

        print(f"[Info] Loading classifier: {clsPath}")
        self.classifier = multiClassifier(
            modelType=modelType,
            inputDim=self.inputDim,
            numLabels=1,
            numLayers=2,
            kernelSize=(3, 3),
            dropout=0.6,
        ).to(device)
        self.classifier.load_state_dict(
            torch.load(clsPath, map_location=device)
        )
        self.classifier.eval()
        print(f"[OK] Classifier loaded ({modelType}).")

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
            - prediction_score : float  (sigmoid output, 0-1)
            - predicted_label  : int    (0 or 1 based on threshold)
        """
        hasLabel = 'label' in df.columns

        # Fill label column with -1 if not present (inference-only mode)
        workDf = df.copy()
        if not hasLabel:
            workDf['label'] = -1

        dataset = ProteinDataset(
            workDf,
            self.batchConverter,
            padding=True,
            paddingNumber=self.paddingNum,
        )
        loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False, drop_last=False)

        allScores = []
        allSeqs = []

        with torch.no_grad():
            for seqStrs, batchTokens, _labels in loader:
                batchTokens = torch.squeeze(batchTokens, 1).to(device)
                results = self.esmModel(batchTokens, repr_layers=[self.esmLayer])
                tokenRepr = results["representations"][self.esmLayer]

                out = self.classifier(tokenRepr, returnFeatures=False)

                # Unified output handling (logits or probabilities depending on model type)
                if isinstance(out, tuple):
                    logits, _ = out
                    scores = torch.sigmoid(logits).view(-1)
                else:
                    # Models that already apply sigmoid return probabilities directly;
                    # models without sigmoid (CNN1D+Linear family, Linear) return logits.
                    # Apply sigmoid defensively -- idempotent if already in [0,1].
                    scores = torch.sigmoid(out).view(-1)

                allScores.extend(scores.cpu().numpy().tolist())
                allSeqs.extend(seqStrs)

        resultDf = df.copy()
        resultDf['prediction_score'] = [round(s, 4) for s in allScores]
        resultDf['predicted_label'] = [1 if s >= self.threshold else 0 for s in allScores]
        return resultDf
