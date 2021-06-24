# Portfolio Return Prediciton with LSTM

Objective
-----------
This project is intent to be a hands-on exercise to understand the intricacy and implementation of the LSTM Neural Network. 

This project consists of two parts:
1. Monthly return prediction for a portfolio of selected stocks ('AAL', 'DIS', 'F', 'MSFT') using LSTM Neural Network. 
2. Based on the predicted returns, construct the optimal portfolio using Efficient Fronter approach.

Dataset
---------
Monthly return of each stocks in the portfolio for the past 5 years extracted using Quandl API.

Output
---------
The LSTM Neural Network returns a 12 month return prediction for each stocks.
The stock returns are then used to calculate the optimal portfolio using Efficient Frontier approach. The output is a frontier graph and a text file consisting the portfolio characteristics including Sharp Ratio, optimal return, optimal risk and weights of the corresponding stocks.
