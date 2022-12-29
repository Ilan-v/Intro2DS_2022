import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import plotly.graph_objects as go
from tqdm import tqdm
np.random.seed(12)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#check for missing values in train and test data
def check_missing_values(data: pd.DataFrame)->pd.DataFrame:
    data_missing_values = data.isna().sum().to_frame().reset_index()
    data_missing_values.columns = ['column', 'missing_values']
    data_missing_values['missing_values_percentage'] = data_missing_values['missing_values'] / data.shape[0]
    data_missing_values = data_missing_values.sort_values(by='missing_values', ascending=False)
    return data_missing_values

#replace missing values with unknown
def replace_missing_values(data: pd.DataFrame)->pd.DataFrame:
    data = data.fillna('unknown')
    return data

def check_the_ratio_of_clicked_from_feature(train_data:pd.DataFrame, test_data:pd.DataFrame, features:list, new_feature_name:str)->pd.DataFrame:
    train_data = train_data.copy()
    for i, feature in enumerate(features):
        unique_categories = list(train_data[feature].unique())
        ratio_clicked_per_category = {}
        train_df_feature_indexd = train_data.set_index(feature)
        for category in tqdm(unique_categories):
            ratio_clicked_per_category[category] = train_df_feature_indexd.loc[category].clicked.mean()
        #creare new column with the ratio of clicked from each category
        train_data[new_feature_name[i]] = train_data[feature].apply(lambda x: ratio_clicked_per_category[x])
        test_data[new_feature_name[i]] = test_data[feature].apply(lambda x: ratio_clicked_per_category[x] if x in ratio_clicked_per_category else np.nan)
    return train_data, test_data

def drop_correlated_features(corr_threshold:float, data:pd.DataFrame)->pd.DataFrame:
    """Drop correlated features"""
    # drop highly correlated features (correlation > corr_threshold)
    corr_matrix = data.corr().abs()
    # select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # find index of feature columns with correlation greater than corr_threshold
    to_drop = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    # print the columns to drop
    print(f'columns to drop are: {to_drop}')
    return to_drop


def plot_roc_curve(y_test:pd.Series, y_pred_proba:pd.Series)->None:
    """Plot ROC curve"""
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 5))
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #add grid
    plt.grid()
    #add naive prediction line
    plt.plot([0, 1], [0, 1], linestyle='--')
    #add the roc curve
    plt.plot(fpr, tpr)
    plt.show()

def plot_precision_recall_curve(y_test:pd.Series, y_pred_proba:pd.Series)->None:
    """Plot precision recall curve"""
    #plot the precision recall curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 5))
    plt.title('Precision Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #add grid
    plt.grid()
    #add the precision recall curve
    plt.plot(recall, precision)
    plt.show()

def plot_confusion_matrix(y_test:pd.Series, y_pred:pd.Series)->None:
    cm = confusion_matrix(y_test, y_pred)
    #plot the confusion matrix
    plt.figure(figsize=(10, 5))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

def plot_comparison_bar_chart(best_estimator:callable, X_train:pd.DataFrame, X_test:pd.DataFrame, y_train:pd.Series, y_test:pd.Series)->None:
    #check if the model is overfitting
    #compare the train and test scores
    #calculate the train scores
    y_train_pred = best_estimator.predict(X_train)
    y_train_pred_proba = best_estimator.predict_proba(X_train)[:,1]
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_auc = roc_auc_score(y_train, y_train_pred_proba)

    #calculate the test scores
    y_test_pred = best_estimator.predict(X_test)
    y_test_pred_proba = best_estimator.predict_proba(X_test)[:,1]
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    test_auc = roc_auc_score(y_test, y_test_pred_proba)

    #plot the scores in multi bar chart for comparison between train and test scores for each metric, next to each other, with plotly express
    fig = go.Figure(data=[
        go.Bar(name='Train', x=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], y=[train_accuracy, train_precision, train_recall, train_f1, train_auc]),
        go.Bar(name='Test', x=['Accuracy', 'Precision', 'Recall', 'F1', 'AUC'], y=[test_accuracy, test_precision, test_recall, test_f1, test_auc])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.show()