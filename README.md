# High-Frequency Market Data: Could You Guess the Stock?

## Overview

The Stock Order Book Classification Challenge is an initiative organized by Capital Fund Management (CFM) that aims to classify stock actions based on sequences of order book updates. This project leverages quantitative trading methodologies, utilizing robust statistical analysis of terabytes of data to inform asset allocation, trading decisions, and automated order execution.

## Objective

The objective of this challenge is to identify the corresponding stock action from a sequence of stock market data updates. The dataset comprises each atomic update of the order book, providing insights on best bid and ask prices, executed trades, and orders placed or canceled. Participants will explore various features, including average spreads, typical order sizes, transaction frequencies, and transaction distributions across trading venues.

## Dataset Description

### Input Data (X)

The input data consists of sequences of 100 consecutive atomic updates of the order book, collected at random times throughout the trading day. The dataset contains:

- **20 sequences per stock per day**
- **504 days of data** (approximately 2 years)
- **24 stocks**

This results in a total of **24,240,000 lines**. The columns include:

- **obs_id**: Unique identifier for each sequence of 100 order book updates.
- **venue**: Stock exchange where the event occurred, encoded as an integer (e.g., NASDAQ, BATY).
- **action**: Type of order book event (‘A’ for addition, ‘D’ for deletion, ‘U’ for update).
- **order_id**: Unique identifier for the affected order.
- **side**: Side of the order book where the event occurred (‘A’ for ask, ‘B’ for bid).
- **price**: Price of the concerned order.
- **bid**: Best bid price.
- **ask**: Best ask price.
- **bid_size**: Volume of orders at the best bid price.
- **ask_size**: Volume of orders at the best ask price.
- **flux**: Change in the order book due to the event.
- **trade**: Boolean indicating whether the event was a sale or cancellation.

Prices are adjusted by subtracting the best bid price for the first event of each sequence from the 'price', 'bid', and 'ask' columns.

### Output Data (Y)

The labels correspond to the `eqt_code_cat`, represented as integers from 0 to 23, identifying the stock concerned. The training and test datasets are derived from two distinct time periods.

## Benchmark Description

### Model Architecture and Feature Construction

The input sequences are preprocessed to generate a tensor of size (100, 30) for each observation, where:

- **8-dimensional embeddings** for the categorical items:
  - **venue**
  - **action**
  - **trade**
- **1-dimensional inputs** for:
  - **bid**
  - **ask**
  - **price**
- **Log-transformed sizes**:
  - **log(bid_size + 1)**
  - **log(ask_size + 1)**
  - **log(flux)**

The tensor is processed using two GRU layers of dimension 64, one applied in a forward direction and the other in reverse. The outputs are concatenated to form a 128-dimensional vector, which is then processed through two dense layers:

1. The first layer condenses the 128 dimensions to 64 using a linear model and applies the SeLU activation function.
2. The second layer combines these 64 dimensions into 24 outputs, applying a softmax activation to produce class probabilities.

### Training Process

The loss function used is cross-entropy, which is standard for classification tasks. Random samples of 1,000 observations are drawn from the training data to construct batches of size (1000, 100, 30). The model is trained using stochastic gradient descent with the Adam optimizer (default Optax parameters and a learning rate of 3e-3) over **10,000 batches**.

## Requirements

- **Python 3.x**
- **Libraries**:
  - **TensorFlow** or **PyTorch**
  - **NumPy**
  - **Pandas**
  - **Scikit-learn**

## Acknowledgments

Special thanks to Capital Fund Management (CFM) for organizing this challenge and providing the dataset.
