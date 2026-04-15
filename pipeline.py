import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import IsolationForest, RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import DBSCAN, OPTICS, KMeans
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, confusion_matrix
import scipy.stats as stats
import warnings

warnings.filterwarnings('ignore')

# ----------------- PAGE CONFIG AND AESTHETICS -----------------
st.set_page_config(page_title="Advanced ML Pipeline Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
        background-color: #ffffff;
        padding: 10px 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        border-radius: 5px;
        padding: 10px 15px;
        background-color: #f8f9fa;
        color: #333;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
        color: #0056b3;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0056b3 !important;
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("Automated Machine Learning Pipeline Dashboard")
st.markdown("A highly interactive and aesthetically pleasing step-by-step pipeline from data loading to hyperparameter tuning.")

# ----------------- SESSION STATE INIT -----------------
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'raw_df' not in st.session_state:
    st.session_state['raw_df'] = None
if 'problem_type' not in st.session_state:
    st.session_state['problem_type'] = 'Classification'
if 'target_feature' not in st.session_state:
    st.session_state['target_feature'] = None
if 'outliers_mask' not in st.session_state:
    st.session_state['outliers_mask'] = None
if 'X_train' not in st.session_state:
    st.session_state['X_train'] = None
if 'X_test' not in st.session_state:
    st.session_state['X_test'] = None
if 'y_train' not in st.session_state:
    st.session_state['y_train'] = None
if 'y_test' not in st.session_state:
    st.session_state['y_test'] = None
if 'selected_features' not in st.session_state:
    st.session_state['selected_features'] = []
if 'action_log' not in st.session_state:
    st.session_state['action_log'] = ["✅ Pipeline initialized."]

# ----------------- TABS -----------------
tabs = st.tabs([
    "1️⃣ Problem Type", 
    "2️⃣ Input Data", 
    "3️⃣ EDA", 
    "4️⃣ Data Engineering", 
    "5️⃣ Feature Selection", 
    "6️⃣ Data Split", 
    "7️⃣ Model Selection", 
    "8️⃣ Training & Validation", 
    "9️⃣ Tuning",
    "🔟 Prediction"
])

# ----------------- SIDEBAR LOG -----------------
with st.sidebar:
    st.header("📋 Pipeline Activity Log")
    if st.session_state.get('df') is not None and st.session_state.get('raw_df') is not None:
        st.markdown(f"**Original Data Shape:** `{st.session_state['raw_df'].shape}`")
        st.markdown(f"**Current Data Shape:** `{st.session_state['df'].shape}`")
        if st.session_state.get('target_feature'):
            st.markdown(f"**Target Feature:** `{st.session_state['target_feature']}`")
        
        csv = st.session_state['df'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Current Data State",
            data=csv,
            file_name='pipeline_data_state.csv',
            mime='text/csv',
        )
        st.markdown("---")
    
    st.subheader("Action History")
    for log in reversed(st.session_state['action_log']):
        st.markdown(f"- {log}")

# ----------------- STEP 1: Problem Definition -----------------
with tabs[0]:
    st.header("Step 1: Define Problem")
    st.markdown("Select the type of machine learning problem you want to solve:")
    st.session_state['problem_type'] = st.radio("Problem Type", ["Classification", "Regression"], 
                                                index=0 if st.session_state['problem_type'] == 'Classification' else 1,
                                                horizontal=True)
    st.success(f"Configured for: **{st.session_state['problem_type']}**")

# ----------------- STEP 2: Input Data -----------------
with tabs[1]:
    st.header("Step 2: Data Loading & Visualization")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    
    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            if st.session_state['raw_df'] is None or not st.session_state['raw_df'].equals(temp_df):
                st.session_state['raw_df'] = temp_df.copy()
                st.session_state['df'] = temp_df.copy()
                st.success("File uploaded successfully!")
        except Exception as e:
            st.error(f"Error reading dataset: {e}")
            
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Data Preview")
            st.dataframe(df.head(10), width='stretch')
        with col2:
            st.subheader("Select Target Variable")
            target_col = st.selectbox("Target Feature", df.columns.tolist())
            st.session_state['target_feature'] = target_col
            st.session_state['selected_features'] = [c for c in df.columns if c != target_col]
        
        st.markdown("---")
        st.subheader("Data Shape Visualization using PCA")
        st.markdown("Select numerical features to visualize the 2D representation of the dataset:")
        selected_pca_features = st.multiselect("Select Features for PCA", num_cols, default=num_cols[:3] if len(num_cols) >= 3 else num_cols)
        
        if len(selected_pca_features) >= 2:
            pca_data = df[selected_pca_features].dropna()
            if not pca_data.empty:
                pca = PCA(n_components=2)
                scaled_data = StandardScaler().fit_transform(pca_data)
                components = pca.fit_transform(scaled_data)
                
                fig = px.scatter(
                    x=components[:, 0], 
                    y=components[:, 1], 
                    color=df.loc[pca_data.index, target_col] if target_col in df.columns else None,
                    title="2D PCA Visualization",
                    labels={'x': 'Principal Component 1', 'y': 'Principal Component 2'},
                    color_discrete_sequence=px.colors.qualitative.Plotly
                )
                st.plotly_chart(fig, width='stretch')
            else:
                st.warning("Selected features contain too many missing values for PCA.")
        else:
            st.info("Select at least 2 numerical features to view PCA.")

# ----------------- STEP 3: Exploratory Data Analysis -----------------
with tabs[2]:
    st.header("Step 3: Exploratory Data Analysis (EDA)")
    if st.session_state['df'] is not None:
        df = st.session_state['df']
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Summary Statistics")
            st.dataframe(df.describe().T, width='stretch')
        
        with col2:
            st.subheader("Data Info & Missing Values")
            missing_df = pd.DataFrame({
                'Data Type': df.dtypes.astype(str),
                'Missing Values': df.isnull().sum(),
                'Missing %': (df.isnull().sum() / len(df)) * 100
            })
            st.dataframe(missing_df, width='stretch')
            
        st.subheader("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if not num_df.empty and len(num_df.columns) > 1:
            corr = num_df.corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r", title="Feature Correlation")
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("Not enough numerical columns to generate correlation heatmap.")
            
        if st.session_state.get('target_feature') and st.session_state['target_feature'] in df.columns:
            target_col = st.session_state['target_feature']
            st.markdown("---")
            st.subheader(f"Target Distribution: `{target_col}`")
            
            if st.session_state['problem_type'] == 'Classification':
                fig_target = px.histogram(df, x=target_col, color=target_col, title="Target Class Distribution")
                st.plotly_chart(fig_target, width='stretch')
            else:
                fig_target = px.histogram(df, x=target_col, title="Target Value Distribution", marginal="box")
                st.plotly_chart(fig_target, width='stretch')
                
            st.subheader("Feature vs Target Relationship")
            # Filter out target column to prevent internal DuplicateError in Plotly
            num_cols_only = [col for col in df.select_dtypes(include=[np.number]).columns.tolist() if col != target_col]
            if num_cols_only:
                selected_dist_feature = st.selectbox("Select a feature to see its distribution relative to the Target:", num_cols_only)
                if st.session_state['problem_type'] == 'Classification':
                    fig_dist = px.box(df, x=target_col, y=selected_dist_feature, color=target_col, title=f"Distribution of {selected_dist_feature} by {target_col}")
                else:
                    fig_dist = px.scatter(df, x=selected_dist_feature, y=target_col, trendline="ols", title=f"{target_col} vs {selected_dist_feature}")
                st.plotly_chart(fig_dist, width='stretch')
    else:
        st.info("Please upload data in Step 2.")


# ----------------- STEP 4: Data Engineering and Cleaning -----------------
with tabs[3]:
    st.header("Step 4: Data Cleaning & Outlier Management")
    if st.session_state['df'] is not None:
        df_clean = st.session_state['df'].copy()
        
        st.subheader("Missing Value Imputation")
        missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
        if missing_cols:
            st.write(f"Columns with missing values: {', '.join(missing_cols)}")
            impute_method = st.selectbox("Imputation Method", ["None", "Mean", "Median", "Mode"])
            
            if st.button("Apply Imputation") and impute_method != "None":
                for col in missing_cols:
                    if impute_method == "Mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].mean(), inplace=True)
                    elif impute_method == "Median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    elif impute_method == "Mode":
                        df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
                st.session_state['df'] = df_clean
                st.session_state['action_log'].append(f"🔧 Imputed missing values in {len(missing_cols)} columns using **{impute_method}**")
                st.success("Missing values imputed.")
                st.rerun()
        else:
            st.write("No missing values found in the target variables.")

        st.markdown("---")
        st.subheader("Outlier Detection")
        outlier_method = st.selectbox("Method", ["IQR", "Isolation Forest", "DBSCAN", "OPTICS"])
        detect_outliers = st.button("Detect Outliers")
        
        num_clean_df = df_clean.select_dtypes(include=[np.number]).dropna()
        if not num_clean_df.empty:
            if detect_outliers:
                outliers = np.zeros(len(num_clean_df), dtype=bool)
                X_scaled = StandardScaler().fit_transform(num_clean_df)
                
                if outlier_method == "IQR":
                    Q1 = num_clean_df.quantile(0.25)
                    Q3 = num_clean_df.quantile(0.75)
                    IQR = Q3 - Q1
                    outliers_df = ((num_clean_df < (Q1 - 1.5 * IQR)) | (num_clean_df > (Q3 + 1.5 * IQR))).any(axis=1)
                    outliers = outliers_df.values
                elif outlier_method == "Isolation Forest":
                    clf = IsolationForest(contamination=0.05, random_state=42)
                    preds = clf.fit_predict(X_scaled)
                    outliers = preds == -1
                elif outlier_method == "DBSCAN":
                    clf = DBSCAN(eps=0.5, min_samples=5)
                    preds = clf.fit_predict(X_scaled)
                    outliers = preds == -1
                elif outlier_method == "OPTICS":
                    clf = OPTICS(min_samples=5)
                    preds = clf.fit_predict(X_scaled)
                    outliers = preds == -1
                
                # Expand outlier mask to original DataFrame dimensions (NaNs considered non-outliers for index mapping)
                full_outliers = pd.Series(False, index=df_clean.index)
                full_outliers.loc[num_clean_df.index] = outliers
                
                st.session_state['outliers_mask'] = full_outliers
                st.write(f"Detected **{full_outliers.sum()}** outliers using {outlier_method}.")
            
            if st.session_state['outliers_mask'] is not None:
                full_outliers = st.session_state['outliers_mask']
                if full_outliers.sum() > 0:
                    st.write(f"Currently tracking **{full_outliers.sum()}** outliers.")
                    
                    if len(num_clean_df.columns) >= 2:
                        pca = PCA(n_components=2)
                        components = pca.fit_transform(StandardScaler().fit_transform(num_clean_df))
                        pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
                        pca_df['Outlier'] = full_outliers.loc[num_clean_df.index].values
                        
                        fig = px.scatter(pca_df, x='PC1', y='PC2', color='Outlier',
                                       color_discrete_map={True: 'red', False: 'blue'},
                                       title="Outliers Highlighted in 2D Space (PCA)")
                        st.plotly_chart(fig, width='stretch')

                    if st.button("Delete Detected Outliers"):
                        df_no_outliers = df_clean[~full_outliers].reset_index(drop=True)
                        st.session_state['df'] = df_no_outliers
                        st.session_state['outliers_mask'] = None
                        st.session_state['action_log'].append(f"🗑️ Removed **{full_outliers.sum()}** outliers.")
                        st.success(f"Removed {full_outliers.sum()} outliers.")
                        st.rerun()
                elif st.session_state['outliers_mask'].sum() == 0 and detect_outliers:
                    st.info("No outliers detected with the current method.")
        else:
            st.warning("Cannot detect outliers: Dataset must contain numerical features without missing values.")
    else:
        st.info("Please upload data in Step 2.")

# ----------------- STEP 5: Feature Selection -----------------
with tabs[4]:
    st.header("Step 5: Feature Selection")
    if st.session_state['df'] is not None and st.session_state['target_feature'] is not None:
        df = st.session_state['df'].dropna() # Feature selection requires no NaN
        target_col = st.session_state['target_feature']
        
        if target_col not in df.columns:
            st.warning("Target column is missing.")
        else:
            num_df = df.select_dtypes(include=[np.number])
            if target_col in num_df.columns:
                X = num_df.drop(columns=[target_col])
                y = num_df[target_col]
            else:
                X = num_df
                # encode target for information gain if it's categorical
                y = LabelEncoder().fit_transform(df[target_col])
                
            if not X.empty:
                fs_method = st.selectbox("Select Feature Selection Method", ["Variance Threshold", "Correlation", "Information Gain"])
                
                if st.button("Calculate Feature Importance"):
                    if fs_method == "Variance Threshold":
                        selector = VarianceThreshold(threshold=0.01)
                        try:
                            selector.fit(X)
                            st.write(pd.DataFrame({'Feature': X.columns, 'Variance': selector.variances_}).sort_values('Variance', ascending=False))
                        except ValueError as e:
                            st.error(f"Variance threshold error: {e}")
                            
                    elif fs_method == "Correlation":
                        if pd.api.types.is_numeric_dtype(df[target_col]):
                            corr = X.corrwith(df[target_col]).abs().sort_values(ascending=False)
                            st.write(pd.DataFrame({'Feature': corr.index, 'Correlation with Target': corr.values}))
                        else:
                            st.warning("Correlation requires a numerical target variable.")
                            
                    elif fs_method == "Information Gain":
                        try:
                            if st.session_state['problem_type'] == 'Classification':
                                y_encoded = LabelEncoder().fit_transform(y)
                                if len(np.unique(y_encoded)) > 0.5 * len(y_encoded):
                                    st.warning("⚠️ Target has many unique values. Ensure you selected a valid target for Classification or switch to Regression.")
                                scores = mutual_info_classif(X, y_encoded)
                            else:
                                scores = mutual_info_regression(X, y)
                            st.write(pd.DataFrame({'Feature': X.columns, 'Information Gain': scores}).sort_values('Information Gain', ascending=False))
                        except Exception as e:
                            st.error(f"Information Gain calculation failed: {e}. Are you sure '{target_col}' is the appropriate target for {st.session_state['problem_type']}?")
                
                # Manual feature selection update
                st.subheader("Select Features for Model Training")
                selected = st.multiselect("Final Features to Keep", X.columns.tolist(), default=X.columns.tolist())
                if st.button("Update Selected Features"):
                    st.session_state['selected_features'] = selected
                    st.session_state['action_log'].append(f"✨ Selected **{len(selected)}** features for training.")
                    st.success(f"Updated! Using {len(selected)} features.")
            else:
                st.warning("Requires numeric features without NaNs to run automated feature selection.")
    else:
        st.info("Proceed to step 2 to upload data and select a target.")

# ----------------- STEP 6: Data Split -----------------
with tabs[5]:
    st.header("Step 6: Train / Test Split")
    if st.session_state['df'] is not None and st.session_state['target_feature'] is not None:
        df = st.session_state['df'].dropna()
        target_col = st.session_state['target_feature']
        selected_cols = st.session_state['selected_features']
        
        if len(selected_cols) == 0:
            st.warning("No features selected. Please select features in Step 5.")
        else:
            test_size = st.slider("Test Size Proportion", 0.05, 0.50, 0.20, step=0.05)
            
            if st.button("Perform Data Split"):
                try:
                    X = df[selected_cols]
                    y = df[target_col]
                    
                    # Ensure numerical encoding if strings
                    X = pd.get_dummies(X, drop_first=True)
                    
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['action_log'].append(f"✂️ Split data into Training ({X_train.shape[0]} rows) and Testing ({X_test.shape[0]} rows).")
                    st.success("Data successfully split!")
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Training Set Size", f"{X_train.shape[0]} rows")
                    col2.metric("Testing Set Size", f"{X_test.shape[0]} rows")
                    
                except Exception as e:
                    st.error(f"Error during splitting: {e}")
    else:
         st.info("Missing data or target variable.")

# ----------------- STEP 7: Model Selection -----------------
with tabs[6]:
    st.header("Step 7: Model Selection")
    
    models = []
    if st.session_state['problem_type'] == 'Classification':
         models = ["Logistic Regression", "SVM", "Random Forest"]
    else:
         models = ["Linear Regression", "SVM", "Random Forest"]
    
    # KMeans is requested in prompt, maybe just for showcasing clustering 
    models.append("KMeans (Clustering)")
    
    selected_model_name = st.selectbox("Choose a Model:", models)
    
    kernel_opt = "rbf"
    if selected_model_name == "SVM":
        kernel_opt = st.selectbox("SVM Kernel Option:", ["linear", "poly", "rbf", "sigmoid"])
        
    st.session_state['selected_model_name'] = selected_model_name
    st.session_state['svm_kernel'] = kernel_opt
    st.success(f"Selected Model: **{selected_model_name}**")

# ----------------- STEP 8: Model Training and KFold Validation -----------------
with tabs[7]:
    st.header("Step 8: Model Training & Validation")
    if st.session_state['X_train'] is not None:
        
        st.subheader("K-Fold Cross Validation Settings")
        kfolds = st.number_input("Enter value for K (Folds):", min_value=2, max_value=20, value=5)
        
        if st.button("Train Model & Evaluate"):
            X_train, y_train = st.session_state['X_train'], st.session_state['y_train']
            X_test, y_test = st.session_state['X_test'], st.session_state['y_test']
            
            problem = st.session_state['problem_type']
            model_name = st.session_state.get('selected_model_name', None)
            
            # Label Encoding target if Classification
            if problem == 'Classification':
                 le = LabelEncoder()
                 le.fit(pd.concat([y_train, y_test]))
                 y_train_encoded = le.transform(y_train)
                 y_test_encoded = le.transform(y_test)
                 st.session_state['target_encoder'] = le
            else:
                 y_train_encoded = y_train
                 y_test_encoded = y_test
                 st.session_state['target_encoder'] = None

            model = None
            if model_name == "Logistic Regression": model = LogisticRegression()
            elif model_name == "Linear Regression": model = LinearRegression()
            elif model_name == "SVM" and problem == 'Classification': model = SVC(kernel=st.session_state.get('svm_kernel', 'rbf'))
            elif model_name == "SVM" and problem == 'Regression': model = SVR(kernel=st.session_state.get('svm_kernel', 'rbf'))
            elif model_name == "Random Forest" and problem == 'Classification': model = RandomForestClassifier(random_state=42)
            elif model_name == "Random Forest" and problem == 'Regression': model = RandomForestRegressor(random_state=42)
            elif model_name == "KMeans (Clustering)": model = KMeans(n_clusters=3, random_state=42)
            
            if model is not None:
                st.session_state['model_instance'] = model
                
                with st.spinner("Training model with K-Fold validation..."):
                    # K-Fold CV
                    if model_name != "KMeans (Clustering)":
                        cv_scoring = 'accuracy' if problem == 'Classification' else 'neg_mean_squared_error'
                        try:
                             cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=kfolds, scoring=cv_scoring)
                             
                             if problem == 'Regression':
                                 cv_scores = -cv_scores # Convert back to positive MSE
                             
                             st.write(f"**K-Fold CV Results (K={kfolds})**")
                             st.write(f"Mean Score: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
                             
                             # Full training
                             model.fit(X_train, y_train_encoded)
                             train_preds = model.predict(X_train)
                             test_preds = model.predict(X_test)
                             
                             # Metrics Output
                             st.subheader("Performance Metrics")
                             col1, col2 = st.columns(2)
                             
                             if problem == 'Classification':
                                 train_acc = accuracy_score(y_train_encoded, train_preds)
                                 test_acc = accuracy_score(y_test_encoded, test_preds)
                                 col1.metric("Training Accuracy", f"{train_acc:.4f}")
                                 col2.metric("Testing Accuracy", f"{test_acc:.4f}")
                                 
                                 # Overfitting Check
                                 if train_acc - test_acc > 0.1:
                                     st.warning("⚠️ High chance of OVERFITTING (Train Acc >> Test Acc)")
                                 elif train_acc < 0.6:
                                     st.error("📉 High chance of UNDERFITTING (Low Train Acc)")
                                 else:
                                     st.success("✅ Model generalized well!")
                                     
                                 st.subheader("Confusion Matrix")
                                 cm = confusion_matrix(y_test_encoded, test_preds)
                                 fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues", title="Test Data Confusion Matrix",
                                                    labels=dict(x="Predicted", y="Actual", color="Count"))
                                 st.plotly_chart(fig_cm, width='stretch')
                                     
                             else:
                                 train_rmse = np.sqrt(mean_squared_error(y_train_encoded, train_preds))
                                 test_rmse = np.sqrt(mean_squared_error(y_test_encoded, test_preds))
                                 train_r2 = r2_score(y_train_encoded, train_preds)
                                 test_r2 = r2_score(y_test_encoded, test_preds)
                                 
                                 col1.metric("Training RMSE", f"{train_rmse:.4f}")
                                 col2.metric("Testing RMSE", f"{test_rmse:.4f}")
                                 col1.metric("Training R-squared", f"{train_r2:.4f}")
                                 col2.metric("Testing R-squared", f"{test_r2:.4f}")
                                 
                                 if test_rmse > train_rmse * 1.5:
                                     st.warning("⚠️ High chance of OVERFITTING (Test Error >> Train Error)")
                                 elif train_r2 < 0.5:
                                     st.error("📉 High chance of UNDERFITTING (Low R2 Score)")
                                 else:
                                     st.success("✅ Model generalized well!")
                                     
                                 st.subheader("Actual vs Predicted")
                                 fig_reg = px.scatter(x=y_test_encoded, y=test_preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted on Test Data")
                                 if len(y_test_encoded) > 0:
                                     min_val, max_val = min(y_test_encoded), max(y_test_encoded)
                                     fig_reg.add_shape(type="line", line=dict(dash='dash', color='red'), x0=min_val, y0=min_val, x1=max_val, y1=max_val)
                                 st.plotly_chart(fig_reg, width='stretch')
                        except Exception as e:
                             st.error(f"Training failed: {e}")
                    else:
                        model.fit(X_train)
                        st.write("KMeans clustering completed. Silhouette scoring generally used.")
                        st.success("Cluster fitting done.")
            else:
                 st.error("Invalid model configuration.")
    else:
        st.info("Please split the data in Step 6 first.")


# ----------------- STEP 9: Hyperparameter Tuning -----------------
with tabs[8]:
    st.header("Step 9: Hyperparameter Tuning")
    if 'model_instance' in st.session_state and st.session_state['X_train'] is not None:
        model_name = st.session_state.get('selected_model_name', '')
        
        st.markdown(f"**Tuning parameters for {model_name}**")
        search_type = st.radio("Search Method:", ["GridSearch", "RandomSearch"])
        
        params = {}
        if model_name == "Random Forest":
            params = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            }
        elif model_name == "SVM":
            params = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto']
            }
        elif model_name in ["Logistic Regression", "Linear Regression"]:
             st.info("Standard tuning parameters limited for basic linear models in this demo.")
             params = {}
        
        if params:
            st.write("Parameter Grid:", params)
            if st.button("Start Tuning"):
                X_train, y_train = st.session_state['X_train'], st.session_state['y_train']
                base_model = st.session_state['model_instance']
                
                # Label Encoding if Classification
                if st.session_state['problem_type'] == 'Classification':
                     y_train_encoded = LabelEncoder().fit_transform(y_train)
                else:
                     y_train_encoded = y_train

                with st.spinner(f"Running {search_type}..."):
                    try:
                        if search_type == "GridSearch":
                            search = GridSearchCV(base_model, params, cv=3, n_jobs=-1)
                        else:
                            search = RandomizedSearchCV(base_model, params, cv=3, n_jobs=-1, n_iter=5)
                            
                        search.fit(X_train, y_train_encoded)
                        st.success("Tuning Complete!")
                        st.write("**Best Parameters Found:**")
                        st.json(search.best_params_)
                        st.write(f"**Best CV Score:** {search.best_score_:.4f}")
                    except Exception as e:
                        st.error(f"Error during tuning: {e}")
        else:
             st.write("No default hyperparameters to tune for this model selection.")
    else:
        st.info("Please train a model in Step 8 first.")

# ----------------- STEP 10: Prediction -----------------
with tabs[9]:
    st.header("Step 10: Prediction (Disease/Outcome)")
    if 'model_instance' in st.session_state and st.session_state['X_train'] is not None:
        st.markdown(f"**Trained Model:** {st.session_state.get('selected_model_name', 'Unknown')}")
        st.markdown("Enter feature values to predict the outcome for a new instance:")
        
        input_data = {}
        # Create columns for better layout of input fields
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, feature in enumerate(st.session_state['X_train'].columns):
            col = cols[idx % num_cols]
            # Try to grab the median or mean from the original data if available
            default_val = 0.0
            if st.session_state['df'] is not None and feature in st.session_state['df'].columns:
                 if pd.api.types.is_numeric_dtype(st.session_state['df'][feature]):
                     default_val = float(st.session_state['df'][feature].median())
                     
            with col:
                 input_data[feature] = st.number_input(f"Value for {feature}", value=default_val, format="%.4f")
            
        st.markdown("---")
        if st.button("Predict Outcome", type="primary"):
            try:
                 input_df = pd.DataFrame([input_data])
                 model = st.session_state['model_instance']
                 pred = model.predict(input_df)
                 
                 problem = st.session_state['problem_type']
                 
                 if problem == 'Classification':
                      result = pred[0]
                      if st.session_state.get('target_encoder') is not None:
                           result = st.session_state['target_encoder'].inverse_transform([result])[0]
                      st.success(f"🩺 Prediction Result: **{result}**")
                 else:
                      st.success(f"📊 Prediction Result (Value): **{pred[0]:.4f}**")
            except Exception as e:
                 st.error(f"Prediction error: {e}")
    else:
        st.info("Train a model in Step 8 to enable predictions.")
        
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Designed with ❤️ using Streamlit - Course CS-303B</p>", unsafe_allow_html=True)
