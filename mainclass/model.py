"""
model.py
--------
Shared model architecture for AAP prediction.
Contains ProteinDataset and multiClassifier used by both training and inference.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as torf


class ProteinDataset(Dataset):
    """Tokenizes protein sequences using ESM2 batch converter and applies padding."""

    def __init__(self, df, batchConverter, padding, paddingNumber):
        self.df = df
        self.batchConverter = batchConverter
        self.padding = padding
        self.paddingNum = paddingNumber

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        seqStr = item['seq']
        label = torch.tensor(int(item['label'])).float()
        _, _, tokens = self.batchConverter([(self.df.index[idx], seqStr)])
        if self.padding:
            if tokens.shape[1] > self.paddingNum:
                tokens = tokens[:, :self.paddingNum]
            tokens = torf.pad(tokens, (0, self.paddingNum - tokens.shape[1]))
        return seqStr, tokens, label


class multiClassifier(nn.Module):
    """
    Multi-architecture classifier head on top of ESM2 embeddings.

    Supported modelType values:
        - 'CNN1D+Linear'          : Full model (best performance)
        - 'CNN1D+Linear_NoConv2'  : Ablation - remove 2nd conv block
        - 'CNN1D+Linear_NoFC1'    : Ablation - remove FC1 + Dropout
        - 'CNN1D+Linear_NoBN'     : Ablation - remove all BatchNorm layers
        - 'Linear'                : Simple linear head
        - 'CNN1D'                 : 1D CNN only
        - 'CNN2D'                 : 2D CNN
        - 'CNN2D+Linear'          : 2D CNN + Linear head
        - 'CNN2D+LSTM'            : 2D CNN + BiLSTM
        - 'LSTM'                  : BiLSTM only

    NOTE: nn.Module layer attribute names (self.conv1, self.bn1, etc.) are intentionally
    kept in their original form because they are state_dict keys — renaming them would
    break loading of pre-trained .pt checkpoint files.
    """

    def __init__(self, modelType, inputDim, numLabels, numLayers=2, kernelSize=(3, 3), dropout=0.6):
        super(multiClassifier, self).__init__()
        self.modelType = modelType

        if self.modelType == "LSTM":
            self.lstm1 = nn.LSTM(input_size=inputDim, hidden_size=16, bidirectional=True,
                                 batch_first=True, num_layers=numLayers, dropout=dropout)
            self.bn = nn.BatchNorm1d(32)
            self.dropout = nn.Dropout(dropout)
            self.tanh = nn.Tanh()
            self.fc1 = nn.Linear(32, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, numLabels)
            self.sigmoid = nn.Sigmoid()

        elif self.modelType == "CNN1D":
            self.conv1 = nn.Conv1d(in_channels=inputDim, out_channels=16, kernel_size=3, stride=2)
            self.conv2 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3, stride=1)
            self.bn2 = nn.BatchNorm1d(16)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(16, numLabels)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.sigmoid = nn.Sigmoid()

        elif self.modelType == "CNN2D":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=kernelSize, padding='same')
            self.bn1 = nn.BatchNorm2d(8)
            self.relu = nn.ReLU()
            self.dropout1 = nn.Dropout(dropout)
            self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=kernelSize, padding='same')
            self.bn2 = nn.BatchNorm2d(16)
            self.dropout2 = nn.Dropout(dropout)
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16, 512)
            self.bn = nn.BatchNorm1d(512)
            self.fc_dropout1 = nn.Dropout(0.5)
            self.fc2 = nn.Linear(512, numLabels)
            self.sigmoid = nn.Sigmoid()

        elif self.modelType == "CNN2D+LSTM":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding='same')
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
            self.bn2 = nn.BatchNorm2d(128)
            self.pool = nn.MaxPool2d(kernel_size=(2, 2))
            cnnOutWidth = inputDim // 2
            self.lstm_input_features = 128 * cnnOutWidth
            self.feature_proj = nn.Linear(self.lstm_input_features, 256)
            self.relu = nn.ReLU()
            self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=numLayers, batch_first=True)
            self.bn_lstm = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(dropout)
            self.tanh = nn.Tanh()
            self.fc = nn.Linear(64, numLabels)
            self.sigmoid = nn.Sigmoid()

        elif self.modelType == "Linear":
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(inputDim, 16)
            self.bn1 = nn.BatchNorm1d(16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, numLabels)
            self.sigmoid = nn.Sigmoid()
            self.dropout = nn.Dropout(dropout)

        elif self.modelType == "CNN2D+Linear":
            self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding='same')
            self.bn1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding='same')
            self.bn2 = nn.BatchNorm2d(128)
            self.global_pool = nn.AdaptiveMaxPool2d((1, 1))
            self.flatten = nn.Flatten()
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)
            self.fc1 = nn.Linear(128, 64)
            self.bn_fc = nn.BatchNorm1d(64)
            self.fc2 = nn.Linear(64, numLabels)
            self.sigmoid = nn.Sigmoid()

        if "CNN1D+Linear" in self.modelType:
            self.global_pool = nn.AdaptiveMaxPool1d(1)
            self.flatten = nn.Flatten()
            self.sigmoid = nn.Sigmoid()
            if self.modelType == "CNN1D+Linear":
                self.conv1 = nn.Conv1d(in_channels=inputDim, out_channels=128, kernel_size=3, padding='same')
                self.bn1 = nn.BatchNorm1d(128)
                self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
                self.bn2 = nn.BatchNorm1d(256)
                self.fc1 = nn.Linear(256, 64)
                self.bn_fc = nn.BatchNorm1d(64)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, numLabels)
            elif self.modelType == "CNN1D+Linear_NoConv2":
                self.conv1 = nn.Conv1d(in_channels=inputDim, out_channels=128, kernel_size=3, padding='same')
                self.bn1 = nn.BatchNorm1d(128)
                self.fc1 = nn.Linear(128, 64)
                self.bn_fc = nn.BatchNorm1d(64)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, numLabels)
            elif self.modelType == "CNN1D+Linear_NoFC1":
                self.conv1 = nn.Conv1d(in_channels=inputDim, out_channels=128, kernel_size=3, padding='same')
                self.bn1 = nn.BatchNorm1d(128)
                self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
                self.bn2 = nn.BatchNorm1d(256)
                self.fc2 = nn.Linear(256, numLabels)
            elif self.modelType == "CNN1D+Linear_NoBN":
                self.conv1 = nn.Conv1d(in_channels=inputDim, out_channels=128, kernel_size=3, padding='same')
                self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding='same')
                self.fc1 = nn.Linear(256, 64)
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(64, numLabels)

    def forward(self, x, returnFeatures=False):
        features = {}

        if self.modelType == "LSTM":
            lstmOut, _ = self.lstm1(x)
            lstmOut = lstmOut[:, -1, :]
            lstmOut = self.bn(lstmOut)
            lstmOut = self.tanh(lstmOut)
            lstmOut = self.dropout(lstmOut)
            output = self.fc1(lstmOut)
            output = self.relu(output)
            output = self.fc2(output)
            return self.sigmoid(output)

        elif self.modelType == "CNN1D":
            x = x.transpose(1, 2)
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            return self.sigmoid(self.fc(x))

        elif self.modelType == "CNN2D":
            x = x.unsqueeze(1)
            x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
            x = self.pool(x)
            x = self.dropout2(self.relu(self.bn2(self.conv2(x))))
            x = self.pool(x)
            x = self.flatten(self.global_pool(x))
            x = self.fc_dropout1(self.relu(self.bn(self.fc1(x))))
            return self.sigmoid(self.fc2(x))

        elif self.modelType == "CNN2D+LSTM":
            x = x.unsqueeze(1)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.pool(x)
            x = x.permute(0, 2, 1, 3)
            B, Time, C, Feat = x.size()
            x = self.relu(self.feature_proj(x.reshape(B, Time, C * Feat)))
            lstmOut, _ = self.lstm(x)
            lstmOut = self.dropout(self.tanh(self.bn_lstm(lstmOut[:, -1, :])))
            return self.sigmoid(self.fc(lstmOut))

        elif self.modelType == "Linear":
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            x = self.relu(self.bn1(self.fc1(x)))
            if returnFeatures:
                features['fc1_out'] = x.clone().detach()
            logits = self.fc2(x)
            if returnFeatures:
                features['logits'] = logits.clone().detach()
                features['prediction_score'] = self.sigmoid(logits).clone().detach()
                return logits, features
            return logits

        elif self.modelType == "CNN2D+Linear":
            x = x.unsqueeze(1)
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.pool1(x)
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.flatten(self.global_pool(x))
            x = self.dropout(self.relu(self.bn_fc(self.fc1(x))))
            return self.sigmoid(self.fc2(x))

        if "CNN1D+Linear" in self.modelType:
            x = x.transpose(1, 2)
            if self.modelType == "CNN1D+Linear":
                x = torch.relu(self.bn1(self.conv1(x)))
                if returnFeatures: features['conv1_out'] = x.clone().detach()
                x = torch.relu(self.bn2(self.conv2(x)))
                if returnFeatures: features['conv2_out'] = x.clone().detach()
                x = self.dropout(torch.relu(self.bn_fc(self.fc1(self.flatten(self.global_pool(x))))))
                if returnFeatures: features['fc1_out'] = x.clone().detach()
                logits = self.fc2(x)
            elif self.modelType == "CNN1D+Linear_NoConv2":
                x = torch.relu(self.bn1(self.conv1(x)))
                if returnFeatures: features['conv1_out'] = x.clone().detach()
                x = self.dropout(torch.relu(self.bn_fc(self.fc1(self.flatten(self.global_pool(x))))))
                if returnFeatures: features['fc1_out'] = x.clone().detach()
                logits = self.fc2(x)
            elif self.modelType == "CNN1D+Linear_NoFC1":
                x = torch.relu(self.bn1(self.conv1(x)))
                if returnFeatures: features['conv1_out'] = x.clone().detach()
                x = torch.relu(self.bn2(self.conv2(x)))
                if returnFeatures: features['conv2_out'] = x.clone().detach()
                logits = self.fc2(self.flatten(self.global_pool(x)))
            elif self.modelType == "CNN1D+Linear_NoBN":
                x = torch.relu(self.conv1(x))
                if returnFeatures: features['conv1_out'] = x.clone().detach()
                x = torch.relu(self.conv2(x))
                if returnFeatures: features['conv2_out'] = x.clone().detach()
                x = self.dropout(torch.relu(self.fc1(self.flatten(self.global_pool(x)))))
                if returnFeatures: features['fc1_out'] = x.clone().detach()
                logits = self.fc2(x)

            if returnFeatures:
                features['logits'] = logits.clone().detach()
                features['prediction_score'] = self.sigmoid(logits).clone().detach()
                return logits, features
            return logits
