import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report, confusion_matrix, accuracy_score
from PIL import Image


'''
#### Loaded preprocessed data-set of the US Dollar and Australian Dollar foreign exchange.
#### The data loaded is chosen from a "Tripple Barrier Strategy" on the second by second foreign exchange data then further fit by a Random Forest Classifier. 
'''
image = st.cache(Image.open)('./data/data.jpg')
st.sidebar.image(image, caption='Machine Learning in Finance',use_column_width=True)
start_date = st.sidebar.date_input('Start Date of Data', value=pd.to_datetime('2006/01/02'), key=None)
end_date = st.sidebar.date_input('End Date of Data', value=pd.to_datetime('2017/12/12'), key=None)
Xdata = st.cache(pd.read_csv)("./data/XUSDAUD.csv",index_col = 0,header =0,parse_dates=['date_time'])
Ydata = st.cache(pd.read_csv)("./data/YUSDAUD.csv",index_col = 0,parse_dates=['date_time'])
X_train = Xdata.loc[start_date:end_date]
y_train = Ydata.loc[start_date:end_date]


if len(X_train) > 0:
    choose = st.multiselect('Select which columns of the data to show', list(X_train.columns), 'log_ret')
    chart = st.line_chart(X_train.loc[start_date:end_date][choose])
    if st.checkbox('Show dataframe'):
        st.write(X_train)


    Model_file = './data/USADRandomForestClassifier.pkl'
    rf = st.cache(pickle.load)(open(Model_file, 'rb'))


    # Performance Metrics
    y_pred_rf = rf.predict_proba(X_train)[:, 1]
    y_pred = rf.predict(X_train)

    fpr_rf, tpr_rf, _ = roc_curve(y_train, y_pred_rf)
    st.text(classification_report(y_train, y_pred))

    st.text("Confusion Matrix")
    st.text(confusion_matrix(y_train, y_pred))

    
    st.text("Accuracy")
    st.text(accuracy_score(y_train, y_pred))

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_rf, tpr_rf, label='RF')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    st.pyplot()
else:
    '''
    # There is no data in between the date ranges.
    '''
