import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
import streamlit as st
import joblib

data = pd.read_csv('SalesStore.csv')


df = data.copy()

from sklearn.preprocessing import LabelEncoder

order_data_encode = LabelEncoder()
df['Order_Date'] = order_data_encode.fit_transform(df[['Order_Date']])

product_reference_encode = LabelEncoder()
df['Product_Reference'] = product_reference_encode.fit_transform(df[['Product_Reference']])

sub_category_encode = LabelEncoder()
df['Sub_Category'] = sub_category_encode.fit_transform(df[['Sub_Category']])

sel_cols = ['Sales','Order_Date','Postal_Code','Product_Reference','Sub_Category','Quantity','Profit']
sel_df = df[sel_cols]

# Train and Test
from sklearn.model_selection import train_test_split

x = sel_df.drop('Profit', axis = 1)
y = sel_df.Profit

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20, random_state = 7)
print(f'xtrain: {xtrain.shape}')
print(f'xtest: {xtest.shape}')
print('ytrain: {}'.format(ytrain.shape))
print('ytest: {}'.format(ytest.shape))

# MODELLING ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
lin_reg = LinearRegression()
lin_reg.fit(xtrain, ytrain)

st.markdown("<h1 style = 'color: #0C2D57; text-align: center; font-family: helvetica'>PROFIT PREDICTOR</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #F11A7B; text-align: center; font-family: cursive '>Built By ELIZABETH T.ADU<h4>", unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html= True)
st.image('pngwing.com.png')

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Project Overview</h4>", unsafe_allow_html = True)
st.write("This work aims to create a predictive model based on machine learning for the purpose of forecasting a company's success, we aim to provide insights into the factors Influencing a company’s financial success, empowering stakeholders to make informed decisions.")
st.markdown("<br>", unsafe_allow_html= True)
st.dataframe(data, use_container_width= True)
st.sidebar.image('pngwing.com (1).png', caption = 'Welcome Dear User')

sales = st.sidebar.number_input('Sales')
order_date = st.sidebar.text_input('Order_Date')
postal_code = st.sidebar.number_input('Postal_Code')
product_reference = st.sidebar.text_input('Product_Reference')
sub_category = st.sidebar.text_input('Sub_Category')
quantity = st.sidebar.number_input('Quantity')

st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<br>", unsafe_allow_html= True)

st.markdown("<h4 style = 'margin: -30px; color: green; text-align: center; font-family: helvetica '>Input Variables</h4>", unsafe_allow_html = True)

inputs = pd.DataFrame()
inputs['Sales'] = [sales]
inputs['Order_Date'] = [order_date]
inputs['Postal_Code'] = [postal_code]
inputs['Product_Reference'] = [product_reference]
inputs['Sub_Category'] = [sub_category]
inputs['Quantity'] = [quantity]

st.dataframe(inputs, use_container_width= True)

# Transforming
inputs['Order_Date'] = order_data_encode.fit_transform(inputs[['Order_Date']])
inputs['Product_Reference'] = product_reference_encode.fit_transform(inputs[['Product_Reference']])
inputs['Sub_Category'] = sub_category_encode.fit_transform(inputs[['Sub_Category']])

prediction_button = st.button('Predict Profit')
if prediction_button:
   predicted = lin_reg.predict(inputs)
   st.success(f'The Profit predicted for your business is {predicted[0].round(2)}')