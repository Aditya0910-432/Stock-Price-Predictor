import pandas_datareader as pdr
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras.models import load_model
import streamlit as st
import datetime


start = '2010-01-01'
end = '2020-12-31'

st.title('Stock Trend Predictor')
user_input = st.text_input('Enter Stock Tiker', 'AAPL')
df = pdr.DataReader(user_input, 'yahoo', start, end)
df.head()

#describe
st.subheader('date from 2010-2020')
st.write(df.describe())

#Vizulation
st.subheader('Closing Price vs Time chart ')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close, 'b')
st.pyplot(fig)

st.subheader('Closing Price vs Moving Average m100 ')
m100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100, 'r')
plt.plot(df.Close, 'b')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Moving Average m100 & m200 ')
m100 = df.Close.rolling(100).mean()
m200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(m100, 'r')
plt.plot(m200, 'g')
plt.plot(df.Close, 'b')
plt.legend()
st.pyplot(fig)

#splitting data into X-train and y_train

data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

x_train=[]
y_train=[]

data_training_array = scaler.fit_transform(data_training)

for i in range (100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Load Model
model = load_model('keras_model.h5')

#Testing Model

past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)
input_data = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])
x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler[0]
y_predicted = y_predicted*scale_factor
y_test = y_test*scale_factor

#Final Graph

fig2 = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label='original_price')
plt.plot(y_predicted, 'r', label='predicted_price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
