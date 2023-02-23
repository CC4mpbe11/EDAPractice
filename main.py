# pip install -q sklearn
# pip install -q  keras

# This is a personal practice script for Exploratory Data Analysis. The analysis is to decide how valuable the data is
# for a ML tensorflow model, or what portion of the data is useful for a tensorflow ML model. This is intended to be
# used on a colab notebook.

from __future__ import absolute_import, division, print_function, unicode_literals
from six.moves import urllib
from IPython.display import clear_output
from google.colab import files
from google.colab import drive
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2_contingency
from pathlib import Path

import io
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.compat.v2.feature_column as fc
import seaborn as sns

data = files.upload()
df = pd.read_csv(io.StringIO(data['mushroom_edibility_commas.csv'].decode('utf-8')))
drive.mount('/content/drive')


def kaggle_load_initial():
    dftrain = pd.read_csv(
        'https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification?select=secondary_data.csv')
    # training data
    dfeval = pd.read_csv(
        'https://www.kaggle.com/datasets/devzohaib/mushroom-edibility-classification?select=secondary_data.csv')
    # testing data
    y_train = dftrain.pop('class')
    y_eval = dfeval.pop('class')


def save_to_csv():
    mushroom_data_fs_path = Path('/content/drive/MyDrive/Datasets/Mushroom Data/mushroom_data_fs.csv')
    mushroom_data_fs_path.parent.mkdir(parents=True, exist_ok=True)
    mushroom_data_fs.to_csv(mushroom_data_fs_path, index=False)


def load_from_csv():
    mushroom_data_fs_path = Path('/content/drive/MyDrive/Datasets/Mushroom Data/mushroom_data_fs.csv')
    mushroom_data_fs = pd.read_csv(mushroom_data_fs_path)


# EDA and Feature Selection

# How many null values in each column.
# What does null mean for each feature?
# Do nulls indicate lack of information?
# Consider dropping feature columns.
# Check using heat map if specific records with many nulls can be dropped.

# Decide on other methods of handling nulls like means/medians/modes
# Decide on making "unknown" a category to replace nulls in some feature columns.
# Consider predicting values for nulls
# Consider KNN, the Nearest neighbor algorithm for dealing with nulls

# Univariate Analysis - Histogram for metric features, and Bar Chart for nominal features. Is variance low or high?

# Correlation Analysis - Check for multicollinearity with heat map. Look for correlations.

# Bivariate Analysis - using Box Plot and Grouped Bar Chart.
# use grouped bar chart to plot each category variable against the target label.
# use box plot to analyze the degree of variance between-group compared to within-group.

# consider filter methods(based on chi-square, ANVOA and mutual information), and wrapper methods(based on forward
# selection and backward elimination)

# Correlation between
# A continuous and a categorical variable: point biserial correlation
# Two Binary Variables: Tetrachoric correlation based on contingency table analysis
# Two Ordinal categorical variables: polychoric correlation
# Two Nominal Variables: Cramers V


def check_for_nulls():
    nanlist = []
    nanlist.append(df['edibility'].isna().sum())  # 0 nulls
    nanlist.append(df['cap-diameter'].isna().sum())  # 0 nulls
    nanlist.append(df['cap-shape'].isna().sum())  # 0 nulls
    nanlist.append(df['cap-surface'].isna().sum())  # 14120 nulls
    nanlist.append(df['cap-color'].isna().sum())  # 0 nulls
    nanlist.append(df['does-bruise-or-bleed'].isna().sum())  # 0 nulls
    nanlist.append(df['gill-attachment'].isna().sum())  # 9884 nulls
    nanlist.append(df['gill-spacing'].isna().sum())  # 25063 nulls   drop this feature in the first model
    nanlist.append(df['gill-color'].isna().sum())  # 0 nulls
    nanlist.append(df['stem-height'].isna().sum())  # 0 nulls
    nanlist.append(df['stem-width'].isna().sum())  # 0 nulls
    nanlist.append(df['stem-root'].isna().sum())  # 51538 nulls   drop this feature in the first model
    nanlist.append(df['stem-surface'].isna().sum())  # 38124 nulls    drop this feature in the first model
    nanlist.append(df['stem-color'].isna().sum())  # 0 nulls
    nanlist.append(df['veil-type'].isna().sum())  # 57892 nulls    drop this feature in the first model
    nanlist.append(df['veil-color'].isna().sum())  # 53656 nulls    drop this feature in the first model
    nanlist.append(df['has-ring'].isna().sum())  # 0 nulls
    nanlist.append(df['ring-type'].isna().sum())  # 2471 nulls
    nanlist.append(df['spore-print-color'].isna().sum())  # 54715 nulls   drop this feature in the first model
    nanlist.append(df['habitat'].isna().sum())  # 0 nulls
    nanlist.append(df['season'].isna().sum())  # 0 nulls
    df['season'].value_counts().plot(kind='bar')
    # print(nanlist)



def plot_for_variance():  # !!!!! make a switch in this function to choose specific plots
    sns.scatterplot(x=df.index, y=df['cap-diameter'], hue=df[
        'edibility'])  # pattern discovered here of many mushrooms with higher cap diameter that are all edible

    sns.scatterplot(x=df.index, y=df['cap-diameter'], hue=df['habitat'])
    sns.scatterplot(x=df.index, y=df['cap-diameter'], hue=df[
        'season'])  # good variance. large cap diameter shroom are all the same season. likely all one mushroom in a
    # specific index threshold

    sns.scatterplot(x=df.index, y=df['cap-diameter'],
                    hue=df['cap-shape'])  # good variance. Same pattern as above. Likely same species of mushroom
    sns.scatterplot(x=df.index, y=df['cap-diameter'], hue=df['cap-color'])  # good variance
    sns.histplot(data=df, x='cap-diameter')

    df['edibility'].value_counts().plot(kind='bar')
    df['cap-shape'].value_counts().plot(kind='bar')  # decent variance
    df['cap-surface'].value_counts().plot(kind='hist')  # very good variance
    df['cap-color'].value_counts().plot(kind='bar')  # decent variance
    df['does-bruise-or-bleed'].value_counts().plot(kind='bar')  # poor variance
    df['gill-attachment'].value_counts().plot(kind='bar')  # good variance possibly create "unknown" category to
    # preserve information

    df['gill-color'].value_counts().plot(kind='bar')  # decent variance
    df['stem-height'].value_counts().plot(kind='hist')  # poor variance
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Mushroom stem height by category')

    df['stem-width'].value_counts().plot(kind='hist', bins=5)
    df['stem-color'].value_counts().plot(kind='bar')
    df['has-ring'].value_counts().plot(kind='bar')
    df['ring-type'].value_counts().plot(kind='bar')
    df['habitat'].value_counts().plot(kind='bar')

    sns.catplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.displot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.histplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.stripplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.boxplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.boxplot(ax=axes[0, 1], data=df, x='habitat', y='stem-height')
    sns.boxplot(ax=axes[0, 2], data=df, x='season', y='stem-height')
    sns.boxplot(ax=axes[1, 0], data=df, x='cap-shape', y='stem-height')
    sns.boxplot(ax=axes[1, 1], data=df, x='cap-color', y='stem-height')

    sns.catplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.displot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.histplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.stripplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.boxplot(ax=axes[0, 0], data=df, x='edibility', y='stem-height')
    sns.boxplot(ax=axes[0, 1], data=df, x='habitat', y='stem-height')
    sns.boxplot(ax=axes[0, 2], data=df, x='season', y='stem-height')
    sns.boxplot(ax=axes[1, 0], data=df, x='cap-shape', y='stem-height')
    sns.boxplot(ax=axes[1, 1], data=df, x='cap-color', y='stem-height')

    sns.histplot(data=df, x='stem-width')


def printValueCounts():
    print(df['cap-surface'].value_counts(dropna=False))
    # NaN    14120
    # t       8196
    # s       7608
    # y       6341
    # h       4974
    # g       4724
    # d       4432
    # e       2584
    # k       2303
    # i       2225
    # w       2150
    # l       1412
    # look for correlation with stem-surface

    print(df['gill-spacing'].value_counts(dropna=False))
    # NaN    25063
    # c      24710
    # d       7766
    # f       3530
    # null seems to signify unknown gill spacing
    # lack of variance if new category were to have been created

    print(df['stem-root'].value_counts(dropna=False))
    # NaN    51538
    # s       3177
    # b       3177
    # r       1412
    # f       1059
    # c        706
    # null seems to signify unknown stem root type
    # lack of variance if new category were to have been created

    print(df['stem-surface'].value_counts(dropna=False))
    # NaN    38124
    # s       6025
    # y       4940
    # i       4396
    # t       2644
    # g       1765
    # k       1581
    # f       1059
    # h        535
    # look for correlation with stem-surface

    print(df['veil-type'].value_counts(dropna=False))
    # NaN    57892
    # u       3177
    # null seems to signify unknown veil type
    # lack of variance if new category were to have been created

    print(df['veil-color'].value_counts(dropna=False))
    # NaN    53656
    # w       5474
    # y        527
    # n        525
    # u        353
    # k        353
    # e        181
    # null may signify a white veil... nevermind, it doesnt
    # lack of variance if new category were to have been created

    print(df['ring-type'].value_counts(dropna=False))
    # f      48361
    # NaN     2471
    # e       2435
    # z       2118
    # l       1427
    # r       1399
    # p       1265
    # g       1240
    # m        353
    # possibly create "unknown" category to preserve information

    print(df['spore-print-color'].value_counts(dropna=False))
    # NaN    54715
    # k       2118
    # p       1259
    # w       1212
    # n       1059
    # g        353
    # u        182
    # r        171
    # look for correlation with cap-color
    # lack of variance if new category were to have been created


y = df[['cap-diameter'
        'cap-shape',
        'cap-surface',
        'cap-color',
        'does-bruise-or-bleed',
        'gill-attachment',
        'gill-color',
        'stem-height',
        'stem-width',
        'stem-color',
        'has-ring',
        'ring-type',
        'habitat',
        'season']]

# 'cap-diameter', #
# 'cap-shape', #
# 'cap-surface', # new category
# 'cap-color', #
# 'does-bruise-or-bleed', #
# 'gill-attachment', # new category
# 'gill-spacing', # drop
# 'gill-color', #
# 'stem-height', #
# 'stem-width', #
# 'stem-root', # drop
# 'stem-surface', # drop
# 'stem-color', #
# 'veil-type', # drop
# 'veil-color', # drop
# 'has-ring', #
# 'ring-type', # new category
# 'spore-print-color', # drop
# 'habitat', #
# 'season' #

y["cap-surface"].fillna("n", inplace=True)
y["gill-attachment"].fillna("n", inplace=True)
y["ring-type"].fillna("n", inplace=True)

y_categorical = df[['cap-shape',
                    'cap-surface',
                    'cap-color',
                    'does-bruise-or-bleed',
                    'gill-attachment',
                    'gill-color',
                    'stem-color',
                    'has-ring',
                    'ring-type',
                    'habitat',
                    'season']]
y_categorical.head()

y.corr()
sns.heatmap(df.corr())  # cap diameter and stem width correlate.


def cramersV(var1, var2):
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape)
    return (stat / (obs * mini))


rows = []

for var1 in y_categorical:
    col = []
    for var2 in y_categorical:
        cramers = cramersV(y_categorical[var1], y_categorical[var2])
        col.append(round(cramers, 2))
    rows.append(col)

cramers_results = np.array(rows)
cramersresults = pd.DataFrame(cramers_results, columns=y_categorical.columns, index=y_categorical.columns)

data = np.random.randint(low=1, high=100, size=(11, 11))

cathm = sns.heatmap(data=cramersresults)
