import joblib
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path
from typing import List, Optional

class CNNNet(nn.Module):
    def __init__(self, params: dict, input_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(32 * (input_dim // 4), 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        return self.fc2(x)

class ModelManager:
    def __init__(self, models_dir: str = "./joblib_models"):
        self.models_dir = Path(models_dir)
        self._loaded = {}
    
    def check_status(self) -> dict:
        status = {'ready': False, 'total_models': 0, 'by_algorithm': {}}
        if not self.models_dir.exists():
            return status
        
        for f in self.models_dir.glob("*.joblib"):
            status['total_models'] += 1
            algo = 'SVR' if 'SVR' in f.name else 'XGB'
            status['by_algorithm'][algo] = status['by_algorithm'].get(algo, 0) + 1
        
        for f in self.models_dir.glob("*.pth"):
            status['total_models'] += 1
            algo = 'DNN' if 'DNN' in f.name else 'CNN'
            status['by_algorithm'][algo] = status['by_algorithm'].get(algo, 0) + 1
        
        status['ready'] = status['total_models'] > 0
        return status
    
    def get_models(self, algorithm: str = None) -> List[dict]:
        models = []
        for f in self.models_dir.glob("*.joblib"):
            if algorithm and algorithm not in f.name:
                continue
            models.append({
                'model_name': f.stem,
                'model_path': str(f),
                'ML_algorithm': 'SVR' if 'SVR' in f.name else 'XGB'
            })
        for f in self.models_dir.glob("*.pth"):
            if algorithm and algorithm not in f.name:
                continue
            models.append({
                'model_name': f.stem,
                'model_path': str(f),
                'ML_algorithm': 'DNN' if 'DNN' in f.name else 'CNN'
            })
        return models
    
    def load_joblib_model(self, path: str):
        key = f"joblib_{path}"
        if key in self._loaded:
            return self._loaded[key]
        try:
            model = joblib.load(path)
            self._loaded[key] = model
            return model
        except:
            return None
    
    def load_torch_model(self, path: str, input_dim: int = 2048):
        key = f"torch_{path}"
        if key in self._loaded:
            return self._loaded[key]
        try:
            model = CNNNet({}, input_dim)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
            model.eval()
            self._loaded[key] = model
            return model
        except:
            return None