import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from typing import Tuple, Optional

class FeatureGenerator:
    def __init__(self, radius: int = 2, nBits: int = 2048):
        self.radius = radius
        self.nBits = nBits
    
    def generate_rdkit_fingerprint(self, smiles: str) -> Tuple[Optional[np.ndarray], dict]:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None, {'error': 'Invalid'}
            
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.nBits)
            arr = np.zeros((1,), dtype=np.int8)
            Chem.DataStructs.ConvertToNumpyArray(fp, arr)
            features = arr.reshape(1, -1).astype(np.float32)
            
            return features, {
                'type': 'ECFP4',
                'dim': features.shape[1],
                'atoms': mol.GetNumAtoms()
            }
        except Exception as e:
            return None, {'error': str(e)}