import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Function to load the dataset and cache it
@st.cache_data
def load_data():
    data_path = 'data/creditcard_part.csv'
    df = pd.read_csv(data_path)
    df.index.name = 'ID'
    return df, df.head(5).drop(columns='Class')

# Simulate a database query
def simulate_db_query(num_rows):
    return raw_data.sample(num_rows)

# Simulate a model API
def simulate_model_api(data):
    model_path = 'ml_models/RF_good.pkl'
    model = joblib.load(model_path)
    X = data.drop(columns='Class')
    
    start_time = time.time()
    y_pred = model.predict(X)
    end_time = time.time()
    
    prediction_time = end_time - start_time
    data['Prediction'] = y_pred
    return data, prediction_time

def fraud_number_metric(df):
    n_predicted = df[df['Prediction Status'] == 1].__len__()
    n_real = df[df['Real Status'] == 1].__len__()
    n_real_and_predicted = df[(df['Real Status'] == 1) & df['Prediction Status'] == 1].__len__()
    ratios = []
    ratios.append(n_real_and_predicted/n_predicted)
    ratios.append(n_real_and_predicted/n_real)
    return n_predicted, n_real, ratios

def visualize_predictions(df: pd.DataFrame, prediction_time: float):
    if df.__len__() > 0:
        col1, col2, col3 = st.columns(3)

        num_predicted_frauds, num_real_frauds, ratio = fraud_number_metric(df)
        col1.metric('Detected Frauds', num_predicted_frauds, f'{100*round(ratio[0],2)}% Precision')
        col2.metric('Real Frauds', num_real_frauds, f'{100*round(ratio[1],2)}% Identified')
        col3.metric('Prediction Time', f'{prediction_time:.2f} seconds', 'for all transactions', delta_color='off')

        st.markdown('''
            To evaluate the model's predictions for these cases, I used two metrics:
                    
            -**Precision Percentage**: Percentage of correct predictions in the total predicted frauds
                    
            -**Identified Percentage**: Percentage of correct predictions in the total real frauds
        ''')

        # Show detected fraudulent transactions
        st.subheader('Detected Fraudulent Transactions')
        st.write(df.drop(columns=['Real Status', 'Prediction Status']))
        
        # Compare with real values
        st.subheader('Comparison with Real Values')
        comparison = df[['Real Status', 'Prediction Status']].replace({'Real Status':{0:'Not a Fraud',1:'Fraud'}, 'Prediction Status':{0:'Not a Fraud',1:'Fraud'}})
        
        # Apply conditional styles
        def highlight_rows(row):
            if row['Real Status'] == row['Prediction Status']:
                return ['background-color: lightgreen']*len(row)
            else:
                return ['background-color: lightcoral']*len(row)
        
        styled_comparison = comparison.style.apply(highlight_rows, axis=1)
        st.dataframe(styled_comparison)
    else:
        st.subheader('No fraudulent transactions detected')

######################################################################
# Web page structure

# Load the dataset
raw_data, head_data = load_data()

# Application title
st.title('Credit Card Fraud Detection Simulator')

# Project description
st.subheader('Explanation')

st.markdown(
    "The application aims to simulate a credit card fraud detection system. Simulating the loading of data from a transaction database and real-time data querying, it makes predictions using a machine learning model simulating an API and shows the transactions that have been classified as fraudulent. This allows users to see how a fraud detection system works and evaluate its performance."
)

# Show a part of the dataset
st.subheader('Data Preview')
st.markdown(
    "The following table shows the first 5 data points describing a set with numerical variables resulting from a PCA transformation. For confidentiality reasons, the original features and additional information about the data are not provided. The features V1, V2, ... V28 are the principal components obtained with PCA. The only features not transformed by PCA are 'Time' (time in seconds between each transaction and the first transaction) and 'Amount' (transaction amount, useful for cost-sensitive learning)."
)
st.write(head_data)

# Simulate a database query
st.subheader('Database Query Simulation')
st.text('''We simulate a random transaction drop by''')
st.code('''
    queried_data = simulate_db_query(num_rows)
    # Make predictions simulating an API
    predicted_data, prediction_time = simulate_model_api(queried_data)
    # Filter fraudulent transactions
    fraudulent_transactions = predicted_data[(predicted_data['Prediction'] == 1) | (predicted_data['Class'] == 1)] 
    fraudulent_transactions = fraudulent_transactions.rename(columns={'Class':'Real Status', 'Prediction':'Prediction Status'})
    # Visualize the results
    visualize_predictions(fraudulent_transactions, prediction_time)
''')
num_rows = st.slider('Number of rows to query', min_value=1000, max_value=25000, step=1000, value=2000)
if st.button('Run'):
    queried_data = simulate_db_query(num_rows)
    st.write(queried_data.drop(columns='Class'))

    # Make predictions simulating an API
    predicted_data, prediction_time = simulate_model_api(queried_data)

    # Filter fraudulent transactions
    fraudulent_transactions = predicted_data[(predicted_data['Prediction'] == 1) | (predicted_data['Class'] == 1)]

    fraudulent_transactions = fraudulent_transactions.rename(columns={'Class':'Real Status', 'Prediction':'Prediction Status'})

    # Visualize the results
    visualize_predictions(fraudulent_transactions, prediction_time)

with st.expander('Explanation'):
    st.markdown("""
    In this project, I developed a fraud detection system using a Random Forest model. We use the Kaggle dataset on credit card fraud available [here](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) with data from over 284800 transactions.
                
    Attached is the project repository [here]()

    ## Explanatory Summary
    
    ### 1. Data Loading and Exploration
    - **Data loading**: We load the `creditcard_part.csv` dataset and explore its structure and descriptive statistics.
    - **Null and duplicate values verification**: We confirm that there are no null values and remove duplicates.""")

    # This makes it possible to visualize the function ´raw_data.info()´
    import io
    
    buffer = io.StringIO()
    raw_data.info(buf=buffer)
    s = buffer.getvalue()
    st.code('''
        raw_data, head_data = load_data()
        print(raw_data.info())
        raw_data.describe()
    ''')
    st.text(s)
    st.write(raw_data.describe())
        

    st.markdown("""
    ### 2. Exploratory Data Analysis (EDA)
    - **Feature distribution**: We visualize the distribution of each feature using histograms.
    - **Correlation heatmap**: We generate a heatmap to visualize the correlations between the dataset features.""")
    st.code("""
        plt.figure()
        t = 1
        for i in raw_data.columns:
            plt.subplot(7,5,t)
            sns.histplot(raw_data[i], kde= True)
            plt.title(i+' Distribution')
            t+= 1
        plt.tight_layout()
        plt.show()
            
        plt.figure()
        sns.heatmap(raw_data.corr(), fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Heatmap of the Resampled Dataset')
        plt.show()
    """)
    st.image('images/Distributions.png')
    st.image('images/Correlations.png')

    st.markdown("""
    ### 3. Data Preparation
    - **Null and duplicate values removal**: We remove null values from the dataset, as well as duplicate values.
    - **Data splitting**: We split the dataset into training, test, and validation sets.
    
    Note: The simulator runs on all 284800 to try to make the experience more realistic, however, the training, testing, and validation are well executed and the metrics are calculated under the validation dataset.""")

    st.code("""
        from sklearn.model_selection import train_test_split

        raw_data.dropna(inplace=True)
        raw_data.drop_duplicates(inplace=True)

        # Split the data into training and test sets
        X_train, X_TestValidation, y_train, y_TestValidation = train_test_split(raw_data.drop(columns='Class'), raw_data['Class'], test_size=0.4, random_state=42)
        X_test, X_validation, y_test, y_validation = train_test_split(X_TestValidation, y_TestValidation, test_size=0.5, random_state=65)
    """)
    st.markdown("""
    ### 4. Model Training
    - **Random Forest Model**: We train a Random Forest model with the training data.
    - **Model evaluation**: We evaluate the model's performance using metrics such as the confusion matrix, classification report, ROC curve, and Precision-Recall curve.
        
    It is vital to choose good metrics to evaluate our model, given the nature of the data (fraud data) our goal is to correctly predict the nature of the transactions, the recommended metric is to use the PR (Precision-Recall) graph and its area under the curve, trying to maximize the latter since we are looking for high degrees of precision at high recall values. As complementary metrics, we can use the ROC & ROC AUC curve, paying special attention to the number of false negatives (trying to keep them minimal), and the specific recall for fraud cases.""")

    st.code('''
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, confusion_matrix

        raw_data.dropna(inplace=True)
        raw_data.drop_duplicates(inplace=True)

        # Split the data into training and test sets
        X_train, X_TestValidation, y_train, y_TestValidation = train_test_split(raw_data.drop(columns='Class'), raw_data['Class'], test_size=0.4, random_state=42)

        X_test, X_validation, y_test, y_validation = train_test_split(X_TestValidation, y_TestValidation, test_size=0.5, random_state=65)

        # Create the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=25)

        # Train the model
        rf_model.fit(X_train, y_train)

        # Make predictions
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        
        from sklearn.metrics import roc_curve, precision_recall_curve, auc

        # Calculate prediction probabilities
        y_pred_prob = rf_model.predict_proba(X_test)[:, 1]

        # Calculate the ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Calculate the Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_prob)
        pr_auc = auc(recall, precision)
            
        # Plot the ROC curve
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(fpr_raw, tpr_raw, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_raw)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Raw Data')
        plt.legend(loc="lower right")

        # Plot the Precision-Recall curve
        plt.subplot(1, 2, 2)
        plt.plot(recall_raw, precision_raw, color='blue', lw=2, label='Precision-Recall curve (area = %0.2f)' % pr_auc_raw)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Raw Data')
        plt.legend(loc="lower left")

        plt.tight_layout()
        plt.show()
    ''')

    st.text("""
        [[56649     5]
        [   23    69]]
                      precision    recall  f1-score   support

                   0       1.00      1.00      1.00     56654
                   1       0.93      0.75      0.83        92

            accuracy                           1.00     56746
           macro avg       0.97      0.87      0.92     56746
        weighted avg       1.00      1.00      1.00     56746
    """)
    st.image('images/Metrics.png')
