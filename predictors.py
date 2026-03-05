import numpy as np
import torch
import torch.nn as nn
from typing import Optional

class BasePredictor:
    def predict(self, features: np.ndarray) -> Optional[float]:
        raise NotImplementedError

class SVRPredictor(BasePredictor):
    def __init__(self, model):
        self.model = model
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        try:
            features = np.asarray(features, dtype=np.float64).reshape(1, -1)
            return float(self.model.predict(features)[0])
        except:
            return None

class XGBPredictor(BasePredictor):
    def __init__(self, model):
        self.model = model
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        try:
            features = np.asarray(features, dtype=np.float64).reshape(1, -1)
            return float(self.model.predict(features)[0])
        except:
            return None

class TorchPredictor(BasePredictor):
    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
    
    def predict(self, features: np.ndarray) -> Optional[float]:
        try:
            features = np.asarray(features, dtype=np.float32).squeeze()
            if features.ndim == 1:
                input_tensor = torch.from_numpy(features).unsqueeze(0).unsqueeze(0)
            else:
                input_tensor = torch.from_numpy(features).unsqueeze(0)
            
            with torch.no_grad():
                pred = self.model(input_tensor.float())
            return float(pred.item())
        except:
            return None

class PredictorManager:
    def __init__(self):
        self._predictors = {}
    
    def load_predictor(self, model_info: dict, manager) -> Optional[BasePredictor]:
        key = model_info.get('model_path', '')
        if key in self._predictors:
            return self._predictors[key]
        
        algo = model_info.get('ML_algorithm', '')
        path = model_info.get('model_path', '')
        
        try:
            if algo in ['SVR', 'XGB']:
                model = manager.load_joblib_model(path)
                if model:
                    pred = SVRPredictor(model) if algo == 'SVR' else XGBPredictor(model)
                    self._predictors[key] = pred
                    return pred
            elif algo in ['DNN', 'CNN']:
                model = manager.load_torch_model(path)
                if model:
                    pred = TorchPredictor(model)
                    self._predictors[key] = pred
                    return pred
        except:
            pass
        return None