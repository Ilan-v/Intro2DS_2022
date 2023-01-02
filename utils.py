import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
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

def check_the_ratio_of_clicked_from_feature(train_data:pd.DataFrame, test_data:pd.DataFrame, features:list)->pd.DataFrame:
    train_data = train_data.copy()
    for i, feature in enumerate(features):
        unique_categories = list(train_data[feature].unique())
        ratio_clicked_per_category = {}
        train_df_feature_indexd = train_data.set_index(feature)
        for category in tqdm(unique_categories):
            ratio_clicked_per_category[category] = train_df_feature_indexd.loc[category].clicked.mean()
        #creare new column with the ratio of clicked from each category
        train_data[feature]= train_data[feature].apply(lambda x: ratio_clicked_per_category[x])
        test_data[feature] = test_data[feature].apply(lambda x: ratio_clicked_per_category[x] if x in ratio_clicked_per_category else np.nan)
    return train_data, test_data

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