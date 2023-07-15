# balanceSheet

## Credit Card Fraud Detection Analysis

This repository contains code for analyzing credit card fraud detection using a logistic regression model. The dataset used for this analysis is stored in a CSV file called `creditcard.csv`.

### Dependencies

To run the code in this repository, the following dependencies are required:

- numpy
- pandas
- sklearn

You can install these dependencies using the following command:

```
pip install numpy pandas scikit-learn
```

### Importing the Dataset

The first step is to import the necessary dependencies and load the dataset into a Pandas DataFrame.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/creditcard.csv')

# Print first 5 rows of the dataset
credit_card_data.head()
```

### Dataset Information

After loading the dataset, you can obtain information about the dataset using the `info()` method.

```python
credit_card_data.info()
```

This will display information about the dataset, including the column names, data types, and the number of non-null values in each column.

### Data Analysis

The code also includes various data analysis steps, such as:

- Checking the number of missing values in each column
- Distribution of legitimate transactions and fraudulent transactions
- Statistical measures of the data (e.g., mean, standard deviation) for legitimate and fraudulent transactions

### Data Preprocessing

The code also includes data preprocessing steps, such as:

- Separating the data into legitimate transactions and fraudulent transactions
- Building a sample dataset with a similar distribution of normal and fraudulent transactions
- Splitting the data into features (X) and targets (Y)
- Splitting the data into training and testing datasets using the `train_test_split()` function

### Model Training

The code trains a logistic regression model using the training dataset.

```python
model = LogisticRegression()

# Training the Logistic Regression model with Training Data
model.fit(X_train, Y_train)
```

### Model Evaluation

The code evaluates the trained model using the test dataset and calculates the accuracy score.

```python
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
```

The accuracy scores on the training and test datasets are then displayed.

Please ensure that you have the `creditcard.csv` file available in the specified location and have installed the required dependencies before running the code.

Note: The code snippet provided assumes that you are running it in a Jupyter environment.
