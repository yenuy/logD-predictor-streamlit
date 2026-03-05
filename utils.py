from rdkit import Chem
from typing import Tuple

def validate_smiles(smiles: str) -> Tuple[bool, str]:
    if not smiles or not isinstance(smiles, str):
        return False, "空字符串"
    
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return False, "无法解析"
        n_atoms = mol.GetNumAtoms()
        if n_atoms == 0:
            return False, "无原子"
        return True, f"{n_atoms} 个原子"
    except Exception as e:
        return False, str(e)