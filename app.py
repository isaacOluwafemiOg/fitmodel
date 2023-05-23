import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import sklearn
import pickle


def train_model(model,train,target):
    X = drop.drop('is_fraud',axis=1)
    y = data['is_fraud']
    X_train,X_test,y_train,y_test= sklearn.model_selection.train_test_split(X,
    y,test_size=0.3,random_state=2)
    model = model.fit(X_train,y_train)
    tr_pred = model.predict(X_train)
    te_pred = model.predict(X_test)
    train_score = f1_score(y_train,tr_pred)
    test_score = f1_score(y_test,te_pred)
    return (model,train_score,test_score)



def main():

    
    
    st.sidebar.header('Dataset to use')
    page = st.sidebar.selectbox("Data Input", ['Default','User Upload'])
    
    
    
    st.title('Credit Card Fraud Predictor')
    
    model = st.file_uploader('The untrained model goes here',type='pkl')
    
    train = st.file_uploader('The data on which you want to train goes here',type='csv')
    target = st.text_input('Target column', 'Status')
    st.write('The currenct target is ',target)
    
    if st.button("Fit the model"):
        with st.spinner("Training the model... Please wait."):
            model,trainscore,testscore = train_model(model,train,target)
        st.write('train score:', trainscore)
        st.write('test score:', testscore)

        model_filename = "trained_model.pkl"
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)

        # Provide a download link for the trained model
        st.download_button(
            label="Download Trained Model",
            data=open(model_filename, "rb").read(),
            file_name=model_filename
        )
    
        





if __name__ == '__main__':
    main()
