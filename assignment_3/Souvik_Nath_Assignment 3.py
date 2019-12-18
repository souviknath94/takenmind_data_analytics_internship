# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 15:49:21 2019

@author: Souvik.Nath
"""

#Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 50)

# Pandas provide you with the functionality to directly pull data into your 
# local python environment from web urls
df = pd.read_csv('https://raw.githubusercontent.com/resbaz/r-novice-gapminder-files/master/data/gapminder-FiveYearData.csv')
print('Dataframe:')
display(df.head())

# Creating the pivot table
# X-axis: year
# Y-axis: country
# Values: Life expectancy(lifExp)
df_pivot = pd.pivot_table(df, values='lifeExp', index='country', columns='year')
display(df_pivot.head())

# Plotting the heatmap
plt.figure(figsize=(20, 10))
sns_plot = sns.heatmap(df_pivot, annot=False, cmap='BuPu')
plt.show()

#Saving the heatmap as a .png file
fig = sns_plot.get_figure()
fig.savefig('output.png')
