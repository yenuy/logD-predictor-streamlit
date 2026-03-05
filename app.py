"""
logD Predictor - Streamlit 版本
修复模型加载和路径问题
"""
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
import sys
import os
import time
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="logD Predictor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

# 模型配置 - 使用绝对路径
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "joblib_models"

# 确保模型目录存在
MODELS_DIR.mkdir(exist_ok=True)

# 导入模块
try:
    # 添加 src 到路径
    src_dir = BASE_DIR / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    
    from src.model_manager import ModelManager
    from src.feature_generator import FeatureGenerator
    from src.predictors import PredictorManager
    from src.utils import validate_smiles
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"❌ 模块导入失败：{e}")
    IMPORT_SUCCESS = False
    st.info("💡 请确保 src 目录存在且包含必要的模块文件")

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

# 检查模型 - 修复版本
def check_models():
    """检查模型文件是否存在"""
    if not MODELS_DIR.exists():
        return False, "模型目录不存在"
    
    # 查找所有支持的模型文件
    joblib_files = list(MODELS_DIR.glob("*.joblib"))
    pth_files = list(MODELS_DIR.glob("*.pth"))
    pkl_files = list(MODELS_DIR.glob("*.pkl"))
    
    total_files = len(joblib_files) + len(pth_files) + len(pkl_files)
    
    if total_files == 0:
        return False, "未找到模型文件 (.joblib, .pth, .pkl)"
    
    return True, f"找到 {total_files} 个模型文件"

# 侧边栏
with st.sidebar:
    st.title("⚙️ 配置")
    
    # 显示当前路径信息（用于调试）
    with st.expander("📁 路径信息"):
        st.write(f"**当前目录**: `{BASE_DIR}`")
        st.write(f"**模型目录**: `{MODELS_DIR}`")
        st.write(f"**模型目录存在**: {MODELS_DIR.exists()}")
        
        if MODELS_DIR.exists():
            files = list(MODELS_DIR.iterdir())
            st.write(f"**文件数量**: {len(files)}")
            if files:
                st.write("**文件列表**:")
                for f in files[:10]:  # 只显示前10个
                    st.write(f"  - {f.name}")
    
    # 检查模型
    models_ready, msg = check_models()
    
    if models_ready:
        st.success(f"✅ {msg}")
        if st.button("🗑️ 清除缓存"):
            import shutil
            import gc
            st.cache_resource.clear()
            gc.collect()
            st.rerun()
    else:
        st.warning(f"⚠️ {msg}")
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
st.markdown('<p class="main-header">🧪 logD Predictor</p>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem;">预测化合物的 CHI logD 值</p>', unsafe_allow_html=True)

# 如果没有导入成功，显示错误并停止
if not IMPORT_SUCCESS:
    st.error("❌ 无法导入必要的模块，请检查 src 目录结构")
    st.stop()

# 如果模型未就绪，显示提示并停止
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
                try:
                    from rdkit import Chem
                    from rdkit.Chem import Draw
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        img = Draw.MolToImage(mol, size=(250, 250))
                        st.image(img)
                except Exception as e:
                    st.error(f"无法生成结构图：{e}")
    
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
                                                'logD': round(float(val), 3)
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
                        else:
                            st.error("❌ 特征生成失败")
                    except Exception as e:
                        st.error(f"❌ 错误：{e}")
                        st.exception(e)

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
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
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
                                        'logD': round(float(val), 3)
                                    })
                        
                        progress_bar.progress((i + 1) / len(df))
                        status_text.text(f"处理中：{i+1}/{len(df)}")
                    
                    progress_bar.empty()
                    status_text.empty()
                
                if results:
                    df_r = pd.DataFrame(results)
                    st.success(f"✅ {len(results)} 个成功")
                    st.dataframe(df_r, use_container_width=True)
                    csv = df_r.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下载", csv, "batch_result.csv", "text/csv")
                else:
                    st.warning("⚠️ 无结果")
        except Exception as e:
            st.error(f"❌ 错误：{e}")
            st.exception(e)

# 页脚
st.divider()
st.markdown('<p style="text-align: center; color: #666;">logD Predictor v1.0 | Streamlit</p>', unsafe_allow_html=True)
