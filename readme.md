# Credit Card Fraud Detection Simulator

## Project Overview

This project aims to develop a credit card fraud detection system using a Random Forest model. The dataset used for this project is sourced from [Kaggle's Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). The project includes data loading, exploratory data analysis (EDA), data preparation, model training, and evaluation. Additionally, a Streamlit application is created to simulate real-time fraud detection.

## Notebooks

### Notebook.ipynb

- **Data Loading and Exploration**: The dataset `creditcard.csv` is loaded and its structure and descriptive statistics are explored.
- **EDA**: Visualizations such as histograms and correlation heatmaps are created to understand the data distribution and relationships between features.
- **Data Preparation**: Techniques like SMOTE are used to balance the dataset. The data is split into training, testing, and validation sets.
- **Model Training**: A Random Forest model is trained using the training data.
- **Model Evaluation**: The model's performance is evaluated using metrics like confusion matrix, classification report, ROC curve, and Precision-Recall curve.
- **Model Saving**: The trained model is saved as `RF_good.pkl` for later use.

### Notebook3.ipynb

- **Simple Random Forest Model**: A simpler version of the Random Forest model is developed and evaluated.
- **Data Visualization**: Additional visualizations are created to further understand the data.
- **Model Training and Evaluation**: Similar steps as in `Notebook.ipynb` are followed to train and evaluate the model.

## Streamlit Application

### app.py

The Streamlit application simulates a real-time fraud detection system. It includes the following features:

- **Data Loading**: The dataset is loaded and cached to improve performance.
- **Database Query Simulation**: A function simulates querying a database by selecting a random subset of the data.
- **Model API Simulation**: A function simulates an API call to the Random Forest model to make predictions.
- **Visualization**: The application displays detected fraudulent transactions and compares them with the actual values. Metrics such as the number of detected frauds, real frauds, and prediction time are also displayed.

### How to Run the Application

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

## Project Structure

```
.
├── data
│   └── creditcard.csv
├── app.py
├── Notebook.ipynb
├── Notebook3.ipynb
├── RF_good.pkl
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- The dataset used in this project is provided by [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
- The project is inspired by the [Reproducible Machine Learning for Credit-Card Fraud Detection - Practical Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/index.html).
