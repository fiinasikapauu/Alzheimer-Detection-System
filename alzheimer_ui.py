"""
Alzheimer UI - Modern Streamlit User Interface
Berisi semua komponen UI untuk aplikasi deteksi dini Alzheimer dengan desain modern
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import webbrowser
import threading
import time
import socket

# Import ML Engine
from alzheimer_ml_engine import (
    AlzheimerMLEngine, 
    download_csv, 
    create_confusion_matrix_plot, 
    create_roc_curve_plot, 
    create_feature_importance_plot
)

# Load CSS from external file
def load_custom_css():
    """Load custom CSS from external file"""
    import os
    
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    css_file_path = os.path.join(current_dir, "styles.css")
    
    try:
        # Read CSS file
        with open(css_file_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        
        # Apply CSS to Streamlit
        st.markdown(f"""
        <style>
        {css_content}
        </style>
        """, unsafe_allow_html=True)
        
    except FileNotFoundError:
        # Fallback: Use basic inline CSS if file not found
        st.markdown(""" 
        <style>
        /* Fallback CSS */
        :root {
            --primary-bg: #1e1e1e;
            --secondary-bg: #2d2d2d;
            --accent-bg: #3d3d3d;
            --text-primary: #ffffff;
            --text-secondary: #b0b0b0;
            --accent-color: #4a9eff;
            --success-color: #4caf50;
            --warning-color: #ff9800;
            --error-color: #f44336;
        }
        
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }
        
        .section-spacing {
            margin: 2rem 0;
        }
        
        .content-wrapper {
            padding: 1rem;
            margin-bottom: 1.5rem;
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.warning("File styles.css tidak ditemukan. Menggunakan CSS fallback.")

# ==================== BROWSER AUTO-OPEN ====================
def get_available_port(start_port=8503):
    port = start_port
    while port < start_port + 100:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(('localhost', port))
            sock.close()
            return port
        except OSError:
            port += 1
    return start_port

def open_browser(port):
    time.sleep(2)
    url = f"http://localhost:{port}"
    print(f"\nAplikasi Deteksi Dini Alzheimer sedang berjalan di: {url}")
    print(f"Browser akan terbuka otomatis dalam 2 detik...")
    print(f"Atau buka manual: {url}\n")
    webbrowser.open(url)

# Auto-open browser (hanya sekali)
if 'browser_opened' not in st.session_state:
    port = get_available_port()
    threading.Thread(target=open_browser, args=(port,), daemon=True).start()
    st.session_state.browser_opened = True

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="Alzheimer Detection System", 
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=None
)

# Load custom CSS
load_custom_css()

# ==================== SESSION STATE INITIALIZATION ====================
if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = AlzheimerMLEngine()
if 'dataset_loaded' not in st.session_state:
    st.session_state.dataset_loaded = False
if 'features_selected' not in st.session_state:
    st.session_state.features_selected = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Input Dataset"

# Navigation function
def create_navigation():
    """Create modern sidebar navigation"""
    with st.sidebar:
        st.markdown('<div class="sidebar-header">Alzheimer Detection by Kelompok 10</div>', unsafe_allow_html=True)
        
        # Navigation items
        nav_items = [
            {"name": "Input Dataset", "icon": "", "key": "input"},
            {"name": "Preprocess Data", "icon": "", "key": "preprocess"},
            {"name": "Analisis Data", "icon": "", "key": "analysis"},
            {"name": "Visualisasi Data", "icon": "", "key": "visualization"},
            {"name": "Prediction", "icon": "", "key": "prediction"}
        ]
        
        st.markdown("### Navigation")
        
        for item in nav_items:
            is_active = st.session_state.current_page == item["name"]
            active_class = "active" if is_active else ""
            
            button_text = f"{item['name']}" if item['icon'] else item['name']
            if st.button(button_text, key=item['key'], use_container_width=True):
                st.session_state.current_page = item["name"]
                st.rerun()
        

def create_upload_section(title, description, file_key, file_types=['csv']):
    """Create modern file upload section"""
    st.markdown(f"""
    <div class="upload-section">
        <div class="upload-icon"></div>
        <div class="upload-text">{title}</div>
        <div class="upload-subtext">{description}</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        f"Choose {file_types[0].upper()} file",
        type=file_types,
        key=file_key,
        label_visibility="collapsed"
    )
    
    return uploaded_file

# Create navigation
create_navigation()

# ==================== MAIN CONTENT ====================
# Main header
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; color: var(--text-primary);">Alzheimer Detection System</h1>
    <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">
        Advanced Machine Learning Pipeline for Early Alzheimer's Detection using SVM
    </p>
</div>
""", unsafe_allow_html=True)

# Page routing based on navigation
if st.session_state.current_page == "Input Dataset":
    # ==================== INPUT DATASET PAGE ====================
    st.markdown("""
    <div class="step-header">
        <h1 style="margin: 0; color: var(--text-primary);">Input Dataset</h1>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
            Upload your CSV dataset to begin the Alzheimer detection analysis
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([3, 2], gap="large")
    
    with col1:
        st.markdown("""
        <div class="content-wrapper">
            <h3 style="color: var(--accent-color); margin-top: 0;">Upload Your Dataset</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = create_upload_section(
            "Drag and drop your CSV file here", 
            "Maximum file size: 200MB • Format: CSV only",
            "main_dataset"
        )
        
        if uploaded_file:
            with st.spinner('Memuat dataset...'):
                result = st.session_state.ml_engine.load_dataset(uploaded_file)
                
                if result['success']:
                    st.success(f"Dataset loaded successfully!")
                    st.session_state.dataset_loaded = True
                    df = result['data']
                    
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    
                    # Dataset metrics in a clean layout
                    st.markdown("""
                    <div class="content-wrapper">
                        <h4 style="color: var(--accent-color); margin-top: 0;">Dataset Overview</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.metric("Total Rows", f"{df.shape[0]:,}")
                    with metric_col2:
                        st.metric("Total Columns", f"{df.shape[1]:,}")
                    with metric_col3:
                        st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
                    
                    st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
                    
                    # Dataset preview with better organization
                    with st.expander("**Dataset Preview** (First 10 rows)", expanded=True):
                        st.dataframe(df.head(10), use_container_width=True, height=300)
                    
                    # Column information in a separate expander
                    with st.expander("**Column Details**"):
                        col_info_df = pd.DataFrame({
                            'Column Name': df.columns,
                            'Data Type': df.dtypes.astype(str),
                            'Non-Null': df.count(),
                            'Missing': df.isnull().sum(),
                            'Missing %': (df.isnull().sum() / len(df) * 100).round(1)
                        })
                        st.dataframe(col_info_df, use_container_width=True)
                else:
                    st.error(f"Error loading dataset: {result['message']}")
        
        # Show example structure when no file is uploaded
        if not uploaded_file:
            st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="content-wrapper">
                <h4 style="color: var(--accent-color); margin-top: 0;">Expected Dataset Format</h4>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="info-card">
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Your CSV file should contain the following columns:
                </p>
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="background-color: var(--accent-bg);">
                            <th style="padding: 12px; border: 1px solid var(--accent-color); text-align: left;">Column</th>
                            <th style="padding: 12px; border: 1px solid var(--accent-color); text-align: left;">Description</th>
                            <th style="padding: 12px; border: 1px solid var(--accent-color); text-align: left;">Range</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);"><strong>MMSE</strong></td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">Cognitive assessment score</td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">0-30</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);"><strong>ADL</strong></td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">Activities of Daily Living</td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">0-10</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);"><strong>FunctionalAssessment</strong></td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">Functional capability score</td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">0-10</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);"><strong>AlcoholConsumption</strong></td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">Weekly alcohol units</td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">0-20</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);"><strong>Diagnosis</strong></td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">Target variable (0=Normal, 1=Alzheimer)</td>
                            <td style="padding: 10px; border: 1px solid var(--accent-bg);">0 or 1</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-wrapper">
            <h3 style="color: var(--accent-color); margin-top: 0;">Requirements</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # File Requirements
        with st.container():
            st.markdown("#### File Requirements")
            st.write("• **Format:** CSV file only")
            st.write("• **Size:** Maximum 200MB")
            st.write("• **Encoding:** UTF-8 recommended")
            st.write("• **Rows:** Minimum 10 samples")
            
            st.markdown("#### Required Columns")
            st.write("• **Diagnosis** - Target variable")
            st.write("• **MMSE** - Cognitive score")
            st.write("• **ADL** - Daily living activities")
            st.write("• **FunctionalAssessment** - Function score")
            st.write("• **AlcoholConsumption** - Alcohol units")
        
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        # Action buttons with better spacing
        st.markdown("""
        <div class="content-wrapper">
            <h3 style="color: var(--accent-color); margin-top: 0;">Next Steps</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.dataset_loaded:
            st.success("**Dataset Ready** - Your dataset has been loaded successfully")
            
            if st.button("**Continue to Preprocessing**", type="primary", use_container_width=True):
                st.session_state.current_page = "Preprocess Data"
                st.rerun()
        else:
            st.info("**Waiting for Dataset** - Please upload your CSV file to continue")
            st.button("**Upload Dataset First**", disabled=True, use_container_width=True)

elif st.session_state.current_page == "Preprocess Data":
    # ==================== PREPROCESS DATA PAGE ====================
    if not st.session_state.dataset_loaded:
        st.warning("**Dataset Required** - Please upload a dataset first before proceeding to preprocessing.")
        
        if st.button("**Back to Input Dataset**", use_container_width=True):
            st.session_state.current_page = "Input Dataset"
            st.rerun()
    else:
        st.markdown("""
        <div class="step-header">
            <h1 style="margin: 0; color: var(--text-primary);">Data Preprocessing</h1>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
                Feature selection and data preprocessing for machine learning
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        df = st.session_state.ml_engine.df
        
        # Feature Selection Section
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("### Feature Selection")
            
            # Fitur yang ditentukan sistem
            target_col = 'Diagnosis'
            selected_features = ['MMSE', 'ADL', 'FunctionalAssessment', 'AlcoholConsumption']
            
            # Validasi ketersediaan kolom
            missing_cols = []
            if target_col not in df.columns:
                missing_cols.append(f"Target: {target_col}")
            
            for feature in selected_features:
                if feature not in df.columns:
                    missing_cols.append(f"Feature: {feature}")
            
            if missing_cols:
                st.error("**Required columns not found in dataset:**")
                for col in missing_cols:
                    st.write(f"- {col}")
                st.write("**Available columns in dataset:**")
                st.write(list(df.columns))
            else:
                st.success("**All required columns found!**")
                
                # Target and Features info
                st.markdown("#### Target Variable")
                st.info(f"**{target_col}** - Alzheimer diagnosis labels")
                if target_col in df.columns:
                    target_info = df[target_col].value_counts()
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("Classes", len(target_info))
                    with col_b:
                        st.metric("Total Samples", target_info.sum())
                
                st.markdown("#### Training Features")
                for i, feature in enumerate(selected_features, 1):
                    if feature in df.columns:
                        st.write(f"{i}. **{feature}**")
                    else:
                        st.write(f"{i}. **{feature}** (missing)")
                
                # Confirm button
                if st.button("**Confirm Feature Selection**", type="primary", use_container_width=True):
                    available_features = [f for f in selected_features if f in df.columns]
                    
                    if len(available_features) >= 1:
                        with st.spinner('Processing selected features...'):
                            result = st.session_state.ml_engine.select_features(target_col, available_features)
                            
                            if result['success']:
                                st.success(f"{result['message']}")
                                st.session_state.features_selected = True
                                st.rerun()
                            else:
                                st.error(f"{result['message']}")
                    else:
                        st.error("No valid features available for training.")
        
        with col2:
            st.markdown("### Data Overview")
            
            if st.session_state.features_selected:
                # Get descriptive stats
                stats_result = st.session_state.ml_engine.get_descriptive_stats()
                
                if stats_result['success']:
                    stats = stats_result['stats']
                    
                    st.markdown("#### Missing Values")
                    missing_df = pd.DataFrame({
                        'Column': stats['missing_values'].index,
                        'Missing': stats['missing_values'].values,
                        'Percentage': (stats['missing_values'].values / len(df) * 100).round(1)
                    })
                    st.dataframe(missing_df, use_container_width=True)
                    
                    # Data types
                    types_result = st.session_state.ml_engine.identify_data_types()
                    if types_result['success']:
                        st.markdown("#### Data Types")
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.metric("Numeric", len(types_result['numeric_cols']))
                        with col_y:
                            st.metric("Categorical", len(types_result['categorical_cols']))
            else:
                st.info("Select features first to see statistics.")
        
        # Preprocessing Pipeline Section
        if st.session_state.features_selected:
            st.markdown("---")
            st.markdown("### Preprocessing Pipeline")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown("#### Configuration")
                numeric_impute = st.selectbox("Numeric imputation strategy", ['median', 'mean', 'most_frequent'])
                encode_categoricals = st.checkbox("Encode categorical features", value=True)
                
                if st.button("**Build Preprocessing Pipeline**", use_container_width=True):
                    with st.spinner('Building preprocessing pipeline...'):
                        prep_result = st.session_state.ml_engine.build_preprocessing_pipeline(
                            numeric_impute=numeric_impute,
                            encode_categoricals=encode_categoricals
                        )
                        
                        if prep_result['success']:
                            st.success(f"{prep_result['message']}")
                            
                            # Handle different data types for features
                            numeric_count = prep_result.get('numeric_features', 0)
                            categorical_count = prep_result.get('categorical_features', 0)
                            
                            if isinstance(numeric_count, (list, tuple)):
                                numeric_count = len(numeric_count)
                            if isinstance(categorical_count, (list, tuple)):
                                categorical_count = len(categorical_count)
                            
                            st.write(f"• Numeric features: {numeric_count}")
                            st.write(f"• Categorical features: {categorical_count}")
                        else:
                            st.error(f"{prep_result['message']}")
            
            with col2:
                st.markdown("#### Next Step")
                if st.button("**Proceed to Analysis**", type="primary", use_container_width=True):
                    st.session_state.current_page = "Analisis Data"
                    st.rerun()

elif st.session_state.current_page == "Analisis Data":
    # ==================== ANALISIS DATA PAGE ====================
    if not st.session_state.features_selected:
        st.warning("**Preprocessing Required** - Please complete preprocessing first.")
        if st.button("**Back to Preprocessing**", use_container_width=True):
            st.session_state.current_page = "Preprocess Data"
            st.rerun()
    else:
        st.markdown("""
        <div class="step-header">
            <h1 style="margin: 0; color: var(--text-primary);">Data Analysis & Training</h1>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
                Configure and train the SVM model for Alzheimer detection
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        # Training Configuration
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("### Training Configuration")
            
            # Training parameters in organized sections
            st.markdown("#### Model Parameters")
            param_col1, param_col2 = st.columns(2)
            
            with param_col1:
                grid_search = st.checkbox("Use GridSearch optimization", help="Better results but slower")
                test_size = st.slider("Test data proportion", 0.1, 0.5, 0.2, 0.05)
            
            with param_col2:
                random_state = st.number_input("Random state", value=42, step=1)
                cv_folds = st.number_input("CV Folds", value=3, min_value=2, max_value=10)
            
            # Data Processing Steps
            st.markdown("---")
            st.markdown("### Data Processing")
            
            if st.button("**Step 1: Encode Target & Split Data**", use_container_width=True):
                with st.spinner('Encoding target and splitting data...'):
                    # Encode target
                    encode_result = st.session_state.ml_engine.encode_target_labels()
                    
                    if encode_result['success']:
                        st.success(f"Target encoding completed")
                        
                        # Train-test split
                        split_result = st.session_state.ml_engine.train_test_split_data(
                            test_size=test_size,
                            random_state=int(random_state)
                        )
                        
                        if split_result['success']:
                            st.success(f"Data split completed")
                            
                            # Check class distribution
                            dist_result = st.session_state.ml_engine.check_class_distribution()
                            
                            if dist_result['success']:
                                st.markdown("#### Data Distribution")
                                dist_col1, dist_col2 = st.columns(2)
                                
                                with dist_col1:
                                    st.markdown("**Training Set:**")
                                    for key, value in dist_result['train_distribution'].items():
                                        st.write(f"• {key}: {value}")
                                
                                with dist_col2:
                                    st.markdown("**Testing Set:**")
                                    for key, value in dist_result['test_distribution'].items():
                                        st.write(f"• {key}: {value}")
                        else:
                            st.error(f"{split_result['message']}")
                    else:
                        st.error(f"{encode_result['message']}")
            
            st.markdown("---")
            st.markdown("### Model Training")
            
            if st.button("**Step 2: Train SVM Model**", type="primary", use_container_width=True):
                with st.spinner('Creating SVM pipeline and training model...'):
                    # Create SVM pipeline
                    pipeline_result = st.session_state.ml_engine.create_svm_pipeline()
                    
                    if pipeline_result['success']:
                        st.success(f"SVM pipeline created")
                        
                        # Train model
                        train_result = st.session_state.ml_engine.train_model_with_gridsearch(
                            use_gridsearch=grid_search,
                            cv=cv_folds
                        )
                        
                        if train_result['success']:
                            st.success(f"Model training completed!")
                            
                            if train_result['best_params']:
                                with st.expander("**Best Parameters (GridSearch)**"):
                                    for param, value in train_result['best_params'].items():
                                        st.write(f"• **{param}**: {value}")
                            
                            st.session_state.model_trained = True
                            st.balloons()
                        else:
                            st.error(f"{train_result['message']}")
                    else:
                        st.error(f"{pipeline_result['message']}")
        
        with col2:
            st.markdown("### Data Insights")
            
            # Visualization options
            show_corr = st.checkbox("Show correlation matrix")
            show_class_dist = st.checkbox("Show class distribution", value=True)
            
            # Show descriptive statistics
            stats_result = st.session_state.ml_engine.get_descriptive_stats()
            if stats_result['success']:
                stats = stats_result['stats']
                df = st.session_state.ml_engine.df
                
                # Target distribution visualization
                if show_class_dist:
                    target_col = 'Diagnosis'
                    if target_col in df.columns:
                        try:
                            fig = px.histogram(df, x=target_col, title='Class Distribution')
                            fig.update_layout(height=250)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.write("**Target Distribution:**")
                            st.write(stats['target_distribution'])
                
                # Correlation matrix
                if show_corr:
                    corr_result = st.session_state.ml_engine.get_correlation_matrix()
                    if corr_result['success']:
                        corr_matrix = corr_result['correlation_matrix']
                        fig_corr = px.imshow(corr_matrix, title='Feature Correlation')
                        fig_corr.update_layout(height=250)
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning(corr_result['message'])
            
            # Navigation
            st.markdown("---")
            st.markdown("### Next Step")
            if st.session_state.model_trained:
                if st.button("**View Results**", type="primary", use_container_width=True):
                    st.session_state.current_page = "Visualisasi Data"
                    st.rerun()
            else:
                st.info("Complete model training to proceed")

elif st.session_state.current_page == "Visualisasi Data":
    # ==================== VISUALISASI DATA PAGE ====================
    if not st.session_state.model_trained:
        st.warning("**Model Training Required** - Please complete model training first.")
        if st.button("**Back to Analysis**", use_container_width=True):
            st.session_state.current_page = "Analisis Data"
            st.rerun()
    else:
        st.markdown("""
        <div class="step-header">
            <h1 style="margin: 0; color: var(--text-primary);">Results & Visualization</h1>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
                Model performance evaluation and prediction results
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        # Model Evaluation Results
        st.markdown("### Model Performance")
        
        # Predict and evaluate
        eval_result = st.session_state.ml_engine.predict_and_evaluate()
        
        if eval_result['success']:
            # Main metrics using st.metric
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                st.metric("Accuracy", f"{eval_result['accuracy']:.3f}")
            
            with metric_col2:
                report = eval_result['classification_report']
                f1_score = report['weighted avg']['f1-score'] if 'weighted avg' in report else 0
                st.metric("F1-Score", f"{f1_score:.3f}")
            
            with metric_col3:
                auc_score = eval_result['roc_data']['auc'] if eval_result['roc_data'] else 0
                st.metric("AUC Score", f"{auc_score:.3f}")
            
            with metric_col4:
                st.metric("Classes", len(eval_result['labels']))
            
            # Visualizations
            st.markdown("---")
            st.markdown("### Model Visualizations")
            
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.markdown("#### Confusion Matrix")
                cm_fig = create_confusion_matrix_plot(eval_result['confusion_matrix'], eval_result['labels'])
                st.plotly_chart(cm_fig, use_container_width=True)
            
            with viz_col2:
                if eval_result['roc_data']:
                    st.markdown("#### ROC Curve")
                    roc_fig = create_roc_curve_plot(
                        eval_result['roc_data']['fpr'],
                        eval_result['roc_data']['tpr'],
                        eval_result['roc_data']['auc']
                    )
                    st.plotly_chart(roc_fig, use_container_width=True)
                else:
                    st.info('ROC Curve only available for binary classification')
            
            # Classification Report
            with st.expander("**Detailed Classification Report**"):
                report_df = pd.DataFrame(eval_result['classification_report']).transpose()
                st.dataframe(report_df, use_container_width=True)
        
        # Feature Importance
        st.markdown("---")
        st.markdown("### Feature Importance")
        
        importance_col1, importance_col2 = st.columns([3, 1])
        
        with importance_col1:
            if st.button("**Calculate Feature Importance**", use_container_width=True):
                with st.spinner('Calculating feature importance...'):
                    imp_result = st.session_state.ml_engine.calculate_feature_importance()
                    
                    if imp_result['success']:
                        st.success(f"Feature importance calculated")
                        
                        imp_fig = create_feature_importance_plot(imp_result['importance_df'])
                        st.plotly_chart(imp_fig, use_container_width=True)
                    else:
                        st.warning(f"{imp_result['message']}")
        
        with importance_col2:
            if 'imp_result' in locals() and imp_result['success']:
                st.markdown("#### Top Features")
                st.dataframe(imp_result['importance_df'].head(5), use_container_width=True)
        
        # Export Results
        st.markdown("---")
        st.markdown("### Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("**Generate Results**", use_container_width=True):
                with st.spinner('Creating prediction results...'):
                    pred_result = st.session_state.ml_engine.create_prediction_results()
                    
                    if pred_result['success']:
                        st.success(f"Results generated successfully")
                        st.dataframe(pred_result['sample_preview'], use_container_width=True)
                    else:
                        st.error(f"{pred_result['message']}")
        
        with col2:
            if 'pred_result' in locals() and pred_result['success']:
                csv_data = download_csv(pred_result['results_df'], 'alzheimer_predictions.csv')
                st.download_button(
                    '**Download CSV**',
                    data=csv_data,
                    file_name='alzheimer_predictions.csv',
                    mime='text/csv',
                    use_container_width=True
                )
            
            # Navigation to Prediction
            st.markdown("#### Next Step")
            if st.button("**Make Predictions**", type="primary", use_container_width=True):
                st.session_state.current_page = "Prediction"
                st.rerun()

elif st.session_state.current_page == "Prediction":
    # ==================== PREDICTION PAGE ====================
    if not st.session_state.model_trained:
        st.warning("**Model Training Required** - Please complete model training first.")
        if st.button("**Back to Analysis**", use_container_width=True):
            st.session_state.current_page = "Analisis Data"
            st.rerun()
    else:
        st.markdown("""
        <div class="step-header">
            <h1 style="margin: 0; color: var(--text-primary);">Prediction Interface</h1>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">
                Enter patient data to predict Alzheimer's disease risk
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-spacing"></div>', unsafe_allow_html=True)
        
        # Instructions
        st.info("**Instructions:** Enter the feature values below to get an Alzheimer's risk prediction with confidence score.")
        
        # Main prediction interface
        col1, col2 = st.columns([2, 1], gap="large")
        
        with col1:
            st.markdown("### Patient Data Input")
            
            # Prediction Form
            with st.form('prediction_form'):
                input_data = {}
                df = st.session_state.ml_engine.df
                
                # Input fields in organized layout
                input_col1, input_col2 = st.columns(2)
                
                with input_col1:
                    st.markdown("#### Cognitive Assessment")
                    mmse = st.number_input(
                        'MMSE Score (0-30)', 
                        min_value=0.0, max_value=30.0,
                        value=float(df['MMSE'].median()) if 'MMSE' in df.columns else 25.0,
                        help="Mini-Mental State Examination - Cognitive assessment score"
                    )
                    input_data['MMSE'] = mmse
                    
                    st.markdown("#### Functional Capability")
                    functional = st.number_input(
                        'Functional Assessment (0-10)', 
                        min_value=0.0, max_value=10.0,
                        value=float(df['FunctionalAssessment'].median()) if 'FunctionalAssessment' in df.columns else 7.0,
                        help="Overall functional capability assessment"
                    )
                    input_data['FunctionalAssessment'] = functional
                
                with input_col2:
                    st.markdown("#### Daily Living")
                    adl = st.number_input(
                        'ADL Score (0-10)', 
                        min_value=0.0, max_value=10.0,
                        value=float(df['ADL'].median()) if 'ADL' in df.columns else 8.0,
                        help="Activities of Daily Living capability"
                    )
                    input_data['ADL'] = adl
                    
                    st.markdown("#### Lifestyle Factor")
                    alcohol = st.number_input(
                        'Alcohol Consumption (0-20)', 
                        min_value=0.0, max_value=20.0,
                        value=float(df['AlcoholConsumption'].median()) if 'AlcoholConsumption' in df.columns else 5.0,
                        help="Weekly alcohol consumption units"
                    )
                    input_data['AlcoholConsumption'] = alcohol
                
                # Prediction button
                submit_prediction = st.form_submit_button('**Analyze Risk**', type="primary", use_container_width=True)
                
                if submit_prediction:
                    with st.spinner('Analyzing patient data...'):
                        pred_result = st.session_state.ml_engine.predict_single_sample(input_data)
                        
                        if pred_result['success']:
                            diagnosis = pred_result['diagnosis']
                            confidence = pred_result['confidence']
                            
                            # Show prediction results
                            st.markdown("---")
                            st.markdown("### Risk Assessment Results")
                            
                            # Main result
                            if "Positif" in str(diagnosis) or "1" in str(diagnosis):
                                st.error(f"**HIGH RISK** - {diagnosis}")
                            else:
                                st.success(f"**LOW RISK** - {diagnosis}")
                            
                            # Confidence and probabilities
                            result_col1, result_col2 = st.columns(2)
                            
                            with result_col1:
                                st.metric("Confidence Level", f"{confidence:.1f}%")
                            
                            with result_col2:
                                if pred_result['probabilities']:
                                    st.markdown("**Risk Breakdown:**")
                                    prob_data = pred_result['probabilities']
                                    
                                    for label, prob in prob_data.items():
                                        st.write(f"• **{label}**: {prob:.1f}%")
                                        st.progress(prob / 100.0)
                        else:
                            st.error(f"{pred_result['message']}")
        
        with col2:
            st.markdown("### Feature Guide")
            
            # Feature descriptions
            with st.container():
                st.markdown("#### MMSE (0-30)")
                st.write("Mini-Mental State Examination - Cognitive assessment")
                st.caption("Lower scores indicate cognitive impairment")
                
                st.markdown("#### ADL (0-10)")
                st.write("Activities of Daily Living - Independence measure")
                st.caption("Lower scores indicate more dependency")
                
                st.markdown("#### Functional Assessment (0-10)")
                st.write("Overall functional capability assessment")
                st.caption("Lower scores indicate functional decline")
                
                st.markdown("#### Alcohol Consumption (0-20)")
                st.write("Weekly alcohol consumption in units")
                st.caption("Higher values may indicate risk factors")
            
            # Dataset reference values
            if st.session_state.dataset_loaded:
                st.markdown("---")
                st.markdown("#### Dataset Reference")
                
                df = st.session_state.ml_engine.df
                
                # Create reference metrics
                if 'MMSE' in df.columns:
                    st.metric("MMSE Average", f"{df['MMSE'].mean():.1f}")
                if 'ADL' in df.columns:
                    st.metric("ADL Average", f"{df['ADL'].mean():.1f}")
                if 'FunctionalAssessment' in df.columns:
                    st.metric("Functional Avg", f"{df['FunctionalAssessment'].mean():.1f}")
                if 'AlcoholConsumption' in df.columns:
                    st.metric("Alcohol Avg", f"{df['AlcoholConsumption'].mean():.1f}")

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("""
<div class="info-card">
    <h3 style="margin-top: 0;">Machine Learning Pipeline Overview</h3>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
        <div>
            <h4 style="color: var(--accent-color); margin-bottom: 0.5rem;">Data Processing</h4>
            <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                <li>Dataset Upload & Validation</li>
                <li>Feature Selection & Engineering</li>
                <li>Data Preprocessing & Cleaning</li>
                <li>Train-Test Split with Stratification</li>
            </ul>
        </div>
        <div>
            <h4 style="color: var(--accent-color); margin-bottom: 0.5rem;">Model Training</h4>
            <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                <li>SVM Pipeline with GridSearch</li>
                <li>Hyperparameter Optimization</li>
                <li>Cross-Validation</li>
                <li>Model Evaluation & Metrics</li>
            </ul>
        </div>
        <div>
            <h4 style="color: var(--accent-color); margin-bottom: 0.5rem;">Analysis & Results</h4>
            <ul style="color: var(--text-secondary); font-size: 0.9rem;">
                <li>Confusion Matrix & ROC Curve</li>
                <li>Feature Importance Analysis</li>
                <li>Prediction Results Export</li>
                <li>Interactive Manual Prediction</li>
            </ul>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-top: 2rem; color: var(--text-secondary);">
    <p><strong>Alzheimer Detection System</strong> - Advanced ML Pipeline for Early Detection</p>
    <p style="font-size: 0.8rem;">Built with Streamlit • Powered by scikit-learn • Enhanced with modern UI/UX</p>
</div>
""", unsafe_allow_html=True)
