# Automated Machine Learning Pipeline Dashboard

Dive into the world of machine learning with this sleek, interactive dashboard that guides you through every step of building robust models. From raw data to hyperparameter tuning, experience the power of automated pipelines with stunning visualizations and intuitive controls.

## Overview

This project is a comprehensive mid-semester assignment for our Artificial Neural Networks course, showcasing an end-to-end machine learning workflow. Built with Streamlit, it transforms complex ML processes into an accessible, visual experience that demystifies data science.

## Key Features

### Interactive Step-by-Step Pipeline
Navigate through nine meticulously designed tabs that walk you through the entire ML lifecycle:

- **Problem Definition**: Choose between classification and regression tasks
- **Data Input**: Upload your datasets or use our curated showcase examples
- **Exploratory Data Analysis**: Uncover insights with dynamic plots and statistical summaries
- **Data Engineering**: Handle missing values, encode categories, and transform features
- **Feature Selection**: Employ advanced techniques like mutual information and variance thresholding
- **Data Splitting**: Configure train-test splits with cross-validation options
- **Model Selection**: Compare multiple algorithms including Random Forest, SVM, and Linear models
- **Training & Validation**: Monitor performance with real-time metrics and confusion matrices
- **Hyperparameter Tuning**: Optimize models using Grid Search and Randomized Search

### Advanced Capabilities
- **Outlier Detection**: Identify and handle anomalies using Isolation Forest and clustering methods
- **Dimensionality Reduction**: Visualize high-dimensional data with PCA and t-SNE
- **Clustering Analysis**: Explore data patterns with K-Means, DBSCAN, and OPTICS
- **Statistical Testing**: Perform hypothesis tests and correlation analysis
- **Model Interpretability**: Gain insights with feature importance rankings

### Visual Excellence
- **Plotly-Powered Charts**: Interactive scatter plots, histograms, and correlation matrices
- **Real-Time Updates**: See results change instantly as you adjust parameters
- **Responsive Design**: Optimized for both desktop and mobile viewing
- **Dark Mode Aesthetics**: Modern UI with carefully chosen color schemes

## Getting Started

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/aviralMadhvan24/ANN-Project.git
   cd ANN-Project
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Launch the dashboard with a single command:
```bash
streamlit run pipeline.py
```

Open your browser to `http://localhost:8501` and start exploring!

### Sample Datasets

The project includes two showcase datasets to get you started:
- **Classification**: Symptom-based dataset for disease prediction
- **Regression**: Symptom severity dataset for disease risk scoring

## Project Structure

```
├── pipeline.py              # Main Streamlit application
├── generate_datasets.py     # Script to create sample datasets
├── read_pdf.py             # PDF text extraction utility
├── showcase_classification.csv  # Sample classification data
├── showcase_regression.csv      # Sample regression data
├── Training.csv             # Medical symptom dataset
├── pdf_content.txt          # Extracted PDF content
└── README.md               # This file
```

## Technical Highlights

- **Modular Architecture**: Clean separation of concerns for easy maintenance
- **Error Handling**: Robust exception management throughout the pipeline
- **Performance Optimization**: Efficient data processing for large datasets
- **Extensible Design**: Easy to add new algorithms and preprocessing techniques
- **Session Management**: Persistent state across tab navigation

## Educational Value

This dashboard serves as an excellent learning tool for:
- Understanding ML pipeline best practices
- Exploring the impact of preprocessing on model performance
- Comparing different algorithms and their strengths
- Grasping hyperparameter optimization concepts
- Visualizing complex data relationships

## Future Enhancements

We're continuously improving the dashboard with plans for:
- Neural network integration for the ANN focus
- Advanced feature engineering techniques
- Model deployment capabilities
- Collaborative features for team projects
- Integration with popular ML platforms

## Contributing

This is a college project, but we're open to suggestions! Feel free to:
- Report bugs or request features
- Suggest improvements to the UI/UX
- Propose new algorithms or techniques

## License

This project is developed as part of an academic assignment. Please respect academic integrity guidelines.

## Acknowledgments

Built with passion for machine learning and data science education. Special thanks to our professors and the open-source community for the amazing libraries that make this possible.

---

*Transform your data into insights with just a few clicks!*"