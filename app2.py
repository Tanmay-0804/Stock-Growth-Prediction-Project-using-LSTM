import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd
import pandas_datareader as data
import numpy as np
from keras.models import load_model
import streamlit as st
#streamlit run app.py
st.set_page_config(layout="wide")
# st.set_page_config(page_title="Prediction App")

#to remove whitespace from top
st.markdown("""
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# local_css("style.css")
st.title('Stock Prediction')
ticker_input=st.text_input('Enter Stock Ticker','TCS.NS')

col3, col4 = st.columns((1,1))
with col3:
  # df=data.Datareader(ticker_input,'yahoo')
  # st.markdown('#')
  print("\n")
  df=yf.Ticker(ticker_input)
  df=df.history(period="max")
#   df.drop(['Dividends','Stock Splits'], inplace=True, axis=1)
  # txt = st.text_area('Text to analyze', value='It was the best of times')
  st.subheader('Data Of the Chosen Ticker')
  st.write(df.describe())

with col4:
  st.markdown("####")
  my_expander = st.expander(label='Expand : NIFTY 50 List')
  with my_expander:
    import pandas as pd
    import pickle
    URL = 'https://www1.nseindia.com/content/indices/ind_nifty50list.csv'
    tlist = pd.read_csv(URL, index_col = 'Company Name')
    tlist.drop(['Series','ISIN Code','Industry'], inplace=True, axis=1)
    st.write(tlist)
    


col1, col2 = st.columns((2,1))


with col1:
    # Add chart #1
    st.subheader('Closing Price Vs Time Chart')
    fig = plt.figure(figsize=(12,6))
    plt.plot(df.Close,'b')
    st.pyplot(fig)
    
    # df.reset_index(inplace=True)
    # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # df.drop(['Dividends','Stock Splits',"Date"], inplace=True, axis=1)
    # print(df)

    #splitting data into training and testing
    data_training= pd.DataFrame(df["Close"][0:int(len(df)*0.70)])
    data_testing= pd.DataFrame(df["Close"][int(len(df)*0.70):int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    # x_train =[]
    # y_train = []
    # for i in range(100,data_training_array.shape[0]):
    #   x_train.append(data_training_array[i-100:i])
    #   y_train.append(data_training_array[i,0])
      
    # import numpy as np
    # x_train, y_train = np.array(x_train),np.array(y_train)

    model=load_model('keras_model.h5')
    #make sure the file is in same folder as this app.py file

    #testing
    past_100_days=data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index=True)
    input_data = scaler.fit_transform(final_df)
    x_test = []
    y_test = []
    for i in range(100, input_data.shape[0]):
      x_test.append(input_data[i-100:i])
      y_test.append(input_data[i,0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scalar= scaler.scale_

    scale_factor=1/scalar[0]
    y_predicted=y_predicted*scale_factor
    y_test=y_test*scale_factor

    st.subheader("Predictions Vs Original")
    fig2 = plt.figure(figsize=(12,6))
    plt.plot(y_test,'b',label="Original Price")
    plt.plot(y_predicted,"r",label="Predicted Price")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    st.pyplot(fig2)
    
with col2:
    # Add chart #4
    st.markdown("#")
    st.subheader('Closing Price Vs Time Chart with Moving Average')
    ma100=df.Close.rolling(100).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'g')
    plt.plot(df.Close,'b')
    st.pyplot(fig)
    
    st.markdown("#")
    st.subheader('Closing Price Vs Time Chart with 100MA and 200MA')
    ma100=df.Close.rolling(100).mean()
    ma200=df.Close.rolling(200).mean()
    fig = plt.figure(figsize=(12,6))
    plt.plot(ma100,'g')
    plt.plot(ma200,'r')
    plt.plot(df.Close,'b')
    st.pyplot(fig)
