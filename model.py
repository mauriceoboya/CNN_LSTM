import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, RepeatVector

# Load the dataset
try:
    dataset = pd.read_excel('/home/fibonacci/Desktop/work/COVID-19 DATA.xlsx')
    column_headers = dataset.iloc[0]
    dataset.columns = column_headers
    dataset = dataset.drop([0])
    dataset['Date'] = pd.to_datetime(dataset['Date'])
    dataset.fillna(0, inplace=True)
except FileNotFoundError:
    st.error("Dataset file not found. Please make sure 'COVID-19 DATA.xlsx' exists.")

# Remove trailing whitespaces from column names
dataset.columns = dataset.columns.str.strip()

# Define the model architecture
def build_model(input_shape):
    model = Sequential([
        Conv1D(100, kernel_size=3, activation='relu', input_shape=input_shape),
        Conv1D(100, kernel_size=3, activation='relu'),
        Conv1D(100, kernel_size=3, activation='relu'),
        MaxPooling1D(2),
        Flatten(),
        RepeatVector(30),
        LSTM(128, activation='relu', return_sequences=True),
        LSTM(128, activation='relu', return_sequences=True),
        Bidirectional(LSTM(64, activation='relu')),
        Dense(128, activation='relu'),
        Dense(1)
    ])
    return model

# Streamlit app
st.title('COVID-19 Prediction App')

# Country selection
selected_country = st.selectbox('Select a country', ['United States', 'India', 'France', 'Brazil'])

# Prepare data for the selected country
country_data = pd.DataFrame(list(dataset[selected_country]), index=dataset['Date'], columns=[selected_country])
st.write(country_data)

# Scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(country_data)

# Prepare input-output sequences
steps = 30
inputs = []
outputs = []
for i in range(len(scaled_data) - steps):
    inputs.append(scaled_data[i:i + steps])
    outputs.append(scaled_data[i + steps])

inputs = np.asarray(inputs)
outputs = np.asarray(outputs)
x_train, x_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, shuffle=False)

# Build and compile the model
input_shape = (inputs.shape[1], inputs.shape[2])  # Shape of input sequence
model = build_model(input_shape)
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=1)

# Make predictions
predicted = model.predict(x_test)
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test)

# Plot predicted vs. actual values
plt.figure(figsize=(10, 6))
plt.plot(predicted, label='Predicted')
plt.plot(y_test, label='Actual')
plt.xlabel('Time')
plt.ylabel('Cases')
plt.title('Predicted vs Actual COVID-19 Cases')
plt.legend()
st.pyplot(plt)

