# Stock Price Prediction using LSTM

## Introduction

This project employs LSTM (Long Short-Term Memory) neural networks to predict stock prices based on historical data. The model is trained on past stock prices and is capable of forecasting future prices for a given stock.

## Getting Started

These instructions will help you set up the project and run it on your local machine for development and testing purposes.

### Prerequisites

Before running the code, ensure you have the following installed:

- Python 3.x
- pip (Python package manager)

### Installation

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/your_username/stock-price-prediction.git
    ```

2. Navigate to the project directory:

    ```bash
    cd stock-price-prediction
    ```

3. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the main Python script:

```bash
python stock_prediction.py
```

## Model Training

The model training process involves splitting the historical stock price data into training and testing sets. The training set is used to train the LSTM model by feeding it sequences of past stock prices. The model architecture consists of multiple LSTM layers, each followed by dropout layers to prevent overfitting. The final layer of the model is a dense layer that predicts the next stock price based on the learned patterns in the data.

![Untitled-1](https://github.com/swapnilgupta14/Stock-Growth-Prediction-Project-using-LSTM/assets/85231522/57434ca2-1c48-46ad-a61f-b5858fa99b66)


## Architecture

The architecture of the LSTM model consists of multiple layers of LSTM units, which are a type of recurrent neural network (RNN) specifically designed to capture long-term dependencies in sequential data. Each LSTM layer processes sequences of historical stock prices, with each LSTM unit maintaining a memory cell to store information over time.

## Dataset

The historical stock price data used for training the model is obtained from Yahoo Finance, a popular financial data provider. The data includes various features such as Open, High, Low, Close prices, and trading volume for a specific stock over a given period. The dataset is preprocessed to convert date-time values, remove irrelevant features, and normalize the data to a common scale.

![image](https://github.com/swapnilgupta14/Stock-Growth-Prediction-Project-using-LSTM/assets/85231522/27b856b7-47dd-4455-af1a-22230c38977a)


## Results

The performance of the trained model is evaluated using several metrics, including Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE). Visualizations, such as line plots comparing actual and predicted prices over time, provide intuitive insights into the model's effectiveness and help identify any patterns or trends captured by the model.

![image](https://github.com/swapnilgupta14/Stock-Growth-Prediction-Project-using-LSTM/assets/85231522/6df76044-7a7d-4d89-8a21-95f2d9771276)

