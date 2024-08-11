# Wide and Deep Networks for Bank Marketing Data

This project is part of a lab assignment where we explored the application of wide and deep networks to the Bank Marketing dataset. The goal is to predict whether a client will subscribe to a term deposit based on various attributes. The notebook includes data preprocessing, feature engineering, model building, and evaluation using TensorFlow and Keras.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Data Preparation](#data-preparation)
4. [Modeling](#modeling)
   - [Model Building](#model-building)
   - [Model Evaluation](#model-evaluation)
   - [Comparative Analysis](#comparative-analysis)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Requirements](#requirements)
8. [How to Run the Notebook](#how-to-run-the-notebook)
9. [References](#references)

## Introduction

The primary objective of this project is to apply wide and deep networks to predict whether a client will subscribe to a term deposit in response to a bank marketing campaign. The Bank Marketing dataset from the UCI Machine Learning Repository is used for this analysis.

## Data Description

The dataset contains 16 features and 45,211 instances, which include both categorical and numerical data. After preprocessing, the final dataset consists of 15 features and 7,842 instances. The target variable is binary, indicating whether a client subscribed to a term deposit.

### Source

Moro, S., Rita, P., and Cortez, P.. (2012). Bank Marketing. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

## Data Preparation

### 1. Loading Data

Data is loaded using the `ucimlrepo` library. Features and target variables are separated for preprocessing.

### 2. Preprocessing

- Instances with missing data are removed.
- Irrelevant features are dropped based on domain knowledge.
- Label encoding is applied to the target variable.
- Numerical features are converted to floats, and categorical features are encoded.

### 3. Feature Engineering

Three crossed features are created:

- **Housing and Loan**: Clients with both loans are less likely to subscribe to a term deposit.
- **Poutcome and Previous**: Past outcomes and the number of previous contacts affect current outcomes.
- **Job and Loan**: Certain jobs combined with a loan affect the likelihood of subscribing to a term deposit.

## Modeling

### Model Building

Three models were built using different activation functions:

1. **ReLU Model**: Three layers with 50, 25, and 10 units.
2. **SELU Model**: Three layers with 50, 25, and 10 units.
3. **Sigmoid Model**: Three layers with 50, 25, and 10 units.

The models were trained using stratified k-fold cross-validation to ensure a balanced representation of classes.

### Model Evaluation

Precision was used as the primary evaluation metric. The SELU model outperformed the other models with an average precision of 0.7007, followed by the ReLU model (0.6872) and the Sigmoid model (0.6704).

### Comparative Analysis

- The SELU model showed the best performance with a 95% confidence interval that indicates a statistically significant difference from the deep-only model.
- ROC curves were plotted to compare the performance of the deep-only model and the wide-and-deep model, with the wide-and-deep model showing a higher AUC score.

## Results

- The SELU model with three layers (50, 25, 10 units) performed the best with an average precision of 0.7007.
- The wide-and-deep model outperformed the deep-only model in precision and AUC scores.
- Embedding weights were extracted, and PCA was applied to visualize clusters in the data, revealing important insights for targeting marketing campaigns.

## Conclusion

The wide-and-deep network demonstrated superior performance in predicting client subscriptions to term deposits, making it a valuable tool for marketing strategies in the banking sector.

## Requirements

To run the notebook, you'll need the following Python libraries:

- `tensorflow`
- `sklearn`
- `pandas`
- `matplotlib`
- `ucimlrepo`
- `numpy`

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## How to Run the Notebook

1. Clone the repository.
2. Install the required packages.
3. Open the Jupyter notebook wide_and_deep_networks.ipynb.
4. Run the cells in sequence to replicate the analysis.

## References

- Moro, S., Rita, P., and Cortez, P.. (2012). Bank Marketing. UCI Machine Learning Repository. [https://doi.org/10.24432/C5K306](https://doi.org/10.24432/C5K306).
