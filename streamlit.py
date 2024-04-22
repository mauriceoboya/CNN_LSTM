import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense
import numpy as np
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
try:
    dataset = pd.read_excel('COVID-19 DATA.xlsx')
    column_headers = dataset.iloc[0]
    dataset.columns = column_headers
    dataset = dataset.drop([0])
    dataset = dataset.iloc[:, 0:6]
    dataset.fillna(0, inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'COVID-19 DATA.xlsx' exists.")

# Function to build and train model

def build_and_train_models(X_train, y_train):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    return model


def plot_predictions(country, X_test, y_test, model):
    predictions = model.predict(X_test)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.plot(np.arange(len(y_test)), y_test, label='Actual', marker='o')
    plt.plot(np.arange(len(predictions)), predictions, label='Predicted', marker='o')
    plt.title(f'Predictions for {country}')
    plt.xlabel('Data Index')
    plt.ylabel('Scaled Value')
    plt.legend()
    plt.grid(True)
    st.pyplot(fig)


def build_and_train_model(X_train, y_train):
    model = Sequential()
    model.add(Flatten(input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    return model
# Streamlit app
def main():
    st.title('CNN-LSTM COVID-19 Data Prediction App')
    if 'dataset' not in globals():
        return
    
    country = st.selectbox('Select Country', dataset.columns.difference(['Date']))
    
    data_country = dataset[[country]].copy()
    scaler = MinMaxScaler()
    data_country_scaled = scaler.fit_transform(data_country)
    X = data_country_scaled.reshape(-1, 1, 1) 
    y = data_country_scaled.flatten()  # Flatten the target   

    # Reshape input data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Cast input data to float32
    X_train = X_train.astype('float32')
    y_train = y_train.astype('float32')
    
    st.subheader(f'Model Evaluation for {country}')
    model = build_and_train_model(X_train, y_train)
    loss = model.evaluate(X_test, y_test, verbose=1)
    st.write(f'Test Loss: {loss}')
    
    st.subheader('Predictions')
    plot_predictions(country, X_test, y_test, model)

if __name__ == '__main__':
    main()
