import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

try:
    df = pd.read_csv('file.csv')
except:
    pass

try:
    df = pd.read_excel('file.xlsx')
except:
    pass


y = ## let the user choose the column for y-axis
x = ## let the user choose the column for x-axis, all x-columns 
x_essential = ## let the user choose the essential x-columns for the model, without the columns that can cause bias
bias_calculater = ## let the user deside which the most important x-column for bias calculation (sex, age, etc.)


if df[bias_calculater].dtype == 'object' or  df[bias_calculater].dtype == 'string':
    pd.get_dummies(df, columns=[bias_calculater], drop_first=True, inplace=True)
    columns_bc = df[bias_calculater].unique().tolist()
## checks if the bias_calculater column is categorical and creates dummy variables for it, then stores the unique categories in a list for later use in bias calculation.


model_0 = sm.Logit('y ~ x_essential', data=df).fit()
predictions = model_0.predict(df[x])
## making a model (the model predicts 1 or 0, yes loan of no loan) and then uses the model to make predictions based on the x-columns.


percentages_model_0 = []
for category in columns_bc:
    df_category = df[df[bias_calculater] == category]
    predictions_category = model_0.predict(df_category[x])
    percentage_1 = np.mean(predictions_category)
    percentages_model_0.append((category, percentage_1))
## this code calculates the percentage of positive predictions (by model 0) (1s) for each category in the bias_calculater column and stores the results in a list of tuples.

## Finally, you can print the percentages for each category to compare the bias in the model's predictions.
## let the user choose how to visualize the results, for example with a bar chart or a table.
## let the user know wheter or not the model is biased based on the calculated percentages, and provide suggestions for how to address any bias that is found.

## VOOR ANNY: dit is mijn eerste opzet, ik denk dat het handig is om nog een code te maken voor het tweede model, de bias moet ook nog berekent worden bij de originele datset.



