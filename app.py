"""
logD Predictor - Streamlit Cloud 版本
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import os
import time
import urllib.request
import ssl

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# 页面配置
st.set_page_config(
    page_title="logD Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 模型配置
MODELS_DIR = Path("./joblib_models")

# CSS 样式
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
    }
    .download-box {
        background-color: #e3f2fd;
        border-left: 5px solid #2196f3;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# 导入模块
try:
    from src.model_manager import ModelManager
    from src.feature_generator import FeatureGenerator
    from src.predictors import PredictorManager
    from src.utils import validate_smiles
except ImportError as e:
    st.error(f"模块导入失败：{e}")

# 缓存函数
@st.cache_resource
def get_model_manager():
    return ModelManager(str(MODELS_DIR))

@st.cache_resource
def get_feature_generator():
    return FeatureGenerator()

@st.cache_resource
def get_predictor_manager():
    return PredictorManager()

# 检查模型
def check_models():
    if not MODELS_DIR.exists():
        return False
    files = list(MODELS_DIR.glob("*.joblib")) + list(MODELS_DIR.glob("*.pth"))
    return len(files) > 0

# 侧边栏
with st.sidebar:
    st.title("⚙️ 配置")
    
    models_ready = check_models()
    
    if models_ready:
        st.success("✅ 模型已就绪")
        if st.button("🗑️ 清除缓存"):
            import shutil
            if MODELS_DIR.exists():
                shutil.rmtree(MODELS_DIR)
            st.cache_resource.clear()
            st.rerun()
    else:
        st.warning("⚠️ 模型未找到")
        st.info("""
        **首次使用需要下载模型：**
        
        1. 从 SourceForge 下载 joblib_models.rar
        2. 解压到 joblib_models/ 目录
        3. 刷新页面
        
        [下载链接](https://sourceforge.net/projects/logd-predictor/files/joblib_models.rar/download)
        """)
    
    st.divider()
    
    st.subheader("🔮 预测配置")
    rep_type = st.radio("分子表示", ["RDKit ECFP4"], index=0)
    use_svr = st.checkbox("SVR", value=True)
    use_xgb = st.checkbox("XGBoost", value=True)
    max_models = st.slider("最大模型数", 1, 10, 3)

# 主界面
st.markdown('<h1 class="main-header">🧪 logD Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666;">预测化合物的 CHI logD 值</p>', unsafe_allow_html=True)

if not models_ready:
    st.markdown("""
    <div class="download-box">
        <h3>⚠️ 首次使用需要下载模型文件</h3>
        <p>请在左侧边栏查看下载说明</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# 选项卡
tab1, tab2 = st.tabs(["📝 单分子预测", "📊 批量预测"])

with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        examples = {
            "阿司匹林": "CC(=O)Oc1ccccc1C(=O)O",
            "咖啡因": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
            "苯": "c1ccccc1",
            "乙醇": "CCO"
        }
        selected = st.selectbox("选择示例", list(examples.keys()))
        smiles = st.text_area("SMILES", value=examples[selected], height=80)
        
        if smiles:
            valid, msg = validate_smiles(smiles)
            if valid:
                st.success(f"✅ 有效：{msg}")
            else:
                st.error(f"❌ 无效：{msg}")
    
    with col2:
        st.subheader("结构")
        if smiles:
            valid, _ = validate_smiles(smiles)
            if valid:
                from rdkit import Chem
                from rdkit.Chem import Draw
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    img = Draw.MolToImage(mol, size=(250, 250))
                    st.image(img)
    
    if st.button("🔮 预测", type="primary", use_container_width=True):
        if smiles:
            valid, _ = validate_smiles(smiles)
            if valid:
                with st.spinner("处理中..."):
                    try:
                        fg = get_feature_generator()
                        features, _ = fg.generate_rdkit_fingerprint(smiles)
                        
                        if features is not None:
                            manager = get_model_manager()
                            pm = get_predictor_manager()
                            
                            results = []
                            for algo in ['SVR', 'XGB']:
                                models = manager.get_models(algo)[:max_models]
                                for m in models:
                                    pred = pm.load_predictor(m, manager)
                                    if pred:
                                        val = pred.predict(features)
                                        if val is not None:
                                            results.append({
                                                '算法': algo,
                                                '模型': m.get('model_name', 'Unknown'),
                                                'logD': round(val, 3)
                                            })
                            
                            if results:
                                df = pd.DataFrame(results)
                                st.success(f"✅ 完成 {len(results)} 个预测")
                                st.dataframe(df, use_container_width=True)
                                
                                import plotly.express as px
                                fig = px.bar(df, x='算法', y='logD', color='算法')
                                st.plotly_chart(fig, use_container_width=True)
                                
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button("📥 下载", csv, "result.csv", "text/csv")
                            else:
                                st.warning("⚠️ 无结果")
                    except Exception as e:
                        st.error(f"错误：{e}")

with tab2:
    uploaded = st.file_uploader("上传 CSV", type=['csv'])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.dataframe(df.head())
            smiles_col = st.selectbox("SMILES 列", df.columns.tolist())
            
            if st.button("🚀 批量预测"):
                results = []
                fg = get_feature_generator()
                manager = get_model_manager()
                pm = get_predictor_manager()
                
                models = manager.get_models('SVR')
                if models:
                    pred = pm.load_predictor(models[0], manager)
                    for i, row in df.iterrows():
                        smiles = str(row[smiles_col])
                        valid, _ = validate_smiles(smiles)
                        if valid:
                            features, _ = fg.generate_rdkit_fingerprint(smiles)
                            if features is not None and pred:
                                val = pred.predict(features)
                                if val is not None:
                                    results.append({
                                        'Row': i+1,
                                        'SMILES': smiles,
                                        'logD': round(val, 3)
                                    })
                
                if results:
                    df_r = pd.DataFrame(results)
                    st.success(f"✅ {len(results)} 个成功")
                    csv = df_r.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载", csv, "batch_result.csv", "text/csv")
        except Exception as e:
            st.error(f"错误：{e}")

# 页脚
st.divider()
st.markdown('<p style="text-align: center; color: #999;">logD Predictor v1.0 | Streamlit Cloud</p>', unsafe_allow_html=True)
