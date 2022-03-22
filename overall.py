#--------------------------------------------------#
# Preparation #
#--------------------------------------------------#
import re
import string
import sys
import csv
import numpy as np
import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# For data cleaning
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag

# LDA
from gensim import corpora
from gensim.models.ldamodel import LdaModel

import pickle

from google.colab import drive
drive.mount('/content/drive')


#--------------------------------------------------#
# Category Aggregation #
#--------------------------------------------------#
app = pd.read_csv('/content/drive/My Drive/csv_files/app_data_log.csv')
bea = pd.read_csv('/content/drive/My Drive/csv_files/bea_data_log.csv')
cds = pd.read_csv('/content/drive/My Drive/csv_files/cds_data_log.csv')
gro = pd.read_csv('/content/drive/My Drive/csv_files/gro_data_log.csv')

app = app.sample(n = 153489, replace = False, weights = None, random_state = 100)
cds = cds.sample(n = 153489, replace = False, weights = None, random_state = 100)
gro = gro.sample(n = 153489, replace = False, weights = None, random_state = 100)

app = app.drop(columns = 'Unnamed: 0')
app = app.drop(columns = '"ProductMaintenance"')
app = app.drop(columns = '"ConsumerSatisfaction"')
app = app.drop(columns = '"ProductAppearance"')
app = app.drop(columns = '"BigTicketItems"')

bea = bea.drop(columns = 'Unnamed: 0')
bea = bea.drop(columns = '"Scent"')
bea = bea.drop(columns = '"Texture"')
bea = bea.drop(columns = '"ConsumerSatisfaction"')
bea = bea.drop(columns = '"ProductAppearance"')

cds = cds.drop(columns = 'Unnamed: 0')
cds = cds.drop(columns = '"AlbumDescription"')
cds = cds.drop(columns = '"RockGenre"')
cds = cds.drop(columns = '"Shows"')
cds = cds.drop(columns = '"LivePerformance"')

gro = gro.drop(columns = 'Unnamed: 0')
gro = gro.drop(columns = '"Beverages"')
gro = gro.drop(columns = '"Confectionery"')
gro = gro.drop(columns = '"Flavor"')
gro = gro.drop(columns = '"ProductAppearance"')

app['Search'] = 1
app['High-Involvement'] = 1
app['Mainstream'] = 1

bea['Search'] = 0
bea['High-Involvement'] = 1
bea['Mainstream'] = 1

cds['Search'] = 0
cds['High-Involvement'] = 1
cds['Mainstream'] = 0

gro['Search'] = 0
gro['High-Involvement'] = 0
gro['Mainstream'] = 1

aggregate = app.append(bea, ignore_index = True)
aggregate = aggregate.append(cds, ignore_index = True)
aggregate = aggregate.append(gro, ignore_index = True)
aggregate

aggregate.to_csv('/content/drive/My Drive/csv_files/aggregate.csv')
aggregate = pd.read_csv('/content/drive/My Drive/csv_files/aggregate.csv')


#--------------------------------------------------#
# Descriptive Statistics #
#--------------------------------------------------#
print(aggregate['HELPFUL'].value_counts())

# Mean
print('ReviewRating_mean:', aggregate['ReviewRating'].mean())
print('ReviewLength_mean:', aggregate['ReviewLength'].mean())
print('ReviewImages_mean:', aggregate['ReviewImages'].mean())

print('ReviewAuthorExperience_mean:', aggregate['ReviewAuthorExperience'].mean())
print('ReviewAuthorReputation_mean:', aggregate['ReviewAuthorReputation'].mean())
print('ReviewAuthorCredibility_mean:', aggregate['ReviewAuthorCredibility'].mean())
print('AverageRatingByReviewAuthor_mean:', aggregate['AverageRatingByReviewAuthor'].mean())

print('ProductAverageRating_mean:', aggregate['ProductAverageRating'].mean())
print('ProductPrice_mean:', aggregate['ProductPrice'].mean())
print('ProductTitleLength_mean:', aggregate['ProductTitleLength'].mean())
print('ProductDescriptionLength_mean:', aggregate['ProductDescriptionLength'].mean())

# Median
print('ReviewRating_median:', aggregate['ReviewRating'].median())
print('ReviewLength_median:', aggregate['ReviewLength'].median())
print('ReviewImages_median:', aggregate['ReviewImages'].median())

print('ReviewAuthorExperience_median:', aggregate['ReviewAuthorExperience'].median())
print('ReviewAuthorReputation_median:', aggregate['ReviewAuthorReputation'].median())
print('ReviewAuthorCredibility_median:', aggregate['ReviewAuthorCredibility'].median())
print('AverageRatingByReviewAuthor_median:', aggregate['AverageRatingByReviewAuthor'].median())

print('ProductAverageRating_median:', aggregate['ProductAverageRating'].median())
print('ProductPrice_median:', aggregate['ProductPrice'].median())
print('ProductTitleLength_median:', aggregate['ProductTitleLength'].median())
print('ProductDescriptionLength_median:', aggregate['ProductDescriptionLength'].median())

# SD
print('ReviewRating_std:', aggregate['ReviewRating'].std())
print('ReviewLength_std:', aggregate['ReviewLength'].std())
print('ReviewImages_std:', aggregate['ReviewImages'].std())

print('ReviewAuthorExperience_std:', aggregate['ReviewAuthorExperience'].std())
print('ReviewAuthorReputation_std:', aggregate['ReviewAuthorReputation'].std())
print('ReviewAuthorCredibility_std:', aggregate['ReviewAuthorCredibility'].std())
print('AverageRatingByReviewAuthor_std:', aggregate['AverageRatingByReviewAuthor'].std())

print('ProductAverageRating_std:', aggregate['ProductAverageRating'].std())
print('ProductPrice_std:', aggregate['ProductPrice'].std())
print('ProductTitleLength_std:', aggregate['ProductTitleLength'].std())
print('ProductDescriptionLength_std:', aggregate['ProductDescriptionLength'].std())

# Range
print('ReviewRating_min:', aggregate['ReviewRating'].min())
print('ReviewLength_min:', aggregate['ReviewLength'].min())
print('ReviewImages_min:', aggregate['ReviewImages'].min())

print('ReviewAuthorExperience_min:', aggregate['ReviewAuthorExperience'].min())
print('ReviewAuthorReputation_min:', aggregate['ReviewAuthorReputation'].min())
print('ReviewAuthorCredibility_min:', aggregate['ReviewAuthorCredibility'].min())
print('AverageRatingByReviewAuthor_min:', aggregate['AverageRatingByReviewAuthor'].min())

print('ProductAverageRating_min:', aggregate['ProductAverageRating'].min())
print('ProductPrice_min:', aggregate['ProductPrice'].min())
print('ProductTitleLength_min:', aggregate['ProductTitleLength'].min())
print('ProductDescriptionLength_min:', aggregate['ProductDescriptionLength'].min())

# Range
print('ReviewRating_max:', aggregate['ReviewRating'].max())
print('ReviewLength_max:', aggregate['ReviewLength'].max())
print('ReviewImages_max:', aggregate['ReviewImages'].max())

print('ReviewAuthorExperience_max:', aggregate['ReviewAuthorExperience'].max())
print('ReviewAuthorReputation_max:', aggregate['ReviewAuthorReputation'].max())
print('ReviewAuthorCredibility_max:', aggregate['ReviewAuthorCredibility'].max())
print('AverageRatingByReviewAuthor_max:', aggregate['AverageRatingByReviewAuthor'].max())

print('ProductAverageRating_max:', aggregate['ProductAverageRating'].max())
print('ProductPrice_max:', aggregate['ProductPrice'].max())
print('ProductTitleLength_max:', aggregate['ProductTitleLength'].max())
print('ProductDescriptionLength_max:', aggregate['ProductDescriptionLength'].max())


#--------------------------------------------------#
# Regression #
#--------------------------------------------------#
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

cols1 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder"]

x1 = aggregate[cols1]
y = aggregate["HELPFUL"]

import pandas as pd
import statsmodels.api as sm
from statsmodels.multivariate.pca import PCA

logit_model = sm.Logit(y,x1)
result1 = logit_model.fit()
print(result1.summary())

cols2 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor"]

x2 = aggregate[cols2]
y = aggregate["HELPFUL"]

logit_model = sm.Logit(y,x2)
result2 = logit_model.fit()
print(result2.summary())

cols3 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor", 
         "ProductAverageRating", "ProductRank", "ProductPrice", "ProductTitleLength", "ProductDescriptionLength"]

x3 = aggregate[cols3]
y = aggregate["HELPFUL"]

logit_model = sm.Logit(y,x3)
result3 = logit_model.fit()
print(result3.summary())

cols4 = ["VerifiedReview", "ReviewRating", "ReviewLength", "ReviewImages", "ReviewPostingOrder", 
         "ReviewAuthorExperience", "ReviewAuthorReputation", "ReviewAuthorCredibility", "AverageRatingByReviewAuthor", 
         "ProductAverageRating", "ProductRank", "ProductPrice", "ProductTitleLength", "ProductDescriptionLength", 
         'Search', 'High-Involvement', 'Mainstream']

x4 = aggregate[cols4]
y = aggregate["HELPFUL"]

logit_model = sm.Logit(y,x4)
result4 = logit_model.fit()
print(result4.summary())


#--------------------------------------------------#
# Coefficients Plot #
#--------------------------------------------------#
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def coefplot(result):
    '''
    Takes in results of OLS model and returns a plot of 
    the coefficients with 95% confidence intervals.
    
    Removes intercept, so if uncentered will return error.
    '''
    # Create dataframe of results summary 
    coef_df = pd.DataFrame(result4.summary().tables[1].data)
    
    # Add column names
    coef_df.columns = coef_df.iloc[0]

    # Drop the extra row with column labels
    coef_df=coef_df.drop(0)

    # Set index to variable names 
    coef_df = coef_df.set_index(coef_df.columns[0])

    # Change datatype from object to float
    coef_df = coef_df.astype(float)

    # Get errors; (coef - lower bound of conf interval)
    errors = coef_df['coef'] - coef_df['[0.025']
    
    # Append errors column to dataframe
    coef_df['errors'] = errors

    # Sort values by coef ascending
    coef_df = coef_df.sort_values(by=['coef'])

    ### Plot Coefficients ###

    # x-labels
    variables = list(coef_df.index.values)
    
    # Add variables column to dataframe
    coef_df['variables'] = variables

    sns.set_context("poster")

    # Define figure, axes, and plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Error bars for 95% confidence interval
    # Can increase capsize to add whiskers
    c = np.array(["mediumaquamarine","palevioletred","cornflowerblue","orange",
                  "palevioletred","mediumaquamarine","cornflowerblue",
                  "mediumaquamarine","mediumaquamarine","cornflowerblue",
                  "mediumaquamarine","orange","palevioletred","palevioletred",
                  "orange","palevioletred","orange"])
    
    coef_df.plot(x='variables', y='coef', kind='bar',
                 ax=ax, color='none', fontsize=16, 
                 ecolor=c, capsize=0,
                 yerr='errors', legend=False)
    
    # Set title & labels
    plt.title('Coefficients of Features w/ 95% Confidence Intervals',fontsize=20)
    ax.set_ylabel('Coefficients',fontsize=18)
    ax.set_xlabel('',fontsize=18)
    
    # Coefficients
    ax.scatter(x=pd.np.arange(coef_df.shape[0]), 
               marker='o', s=200, 
               y=coef_df['coef'], color=c)
    
    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='black', linewidth=2)
    
    # Add legend
    legend_elements = [Line2D([0],[0],marker='o',color='palevioletred',
                         label='Review Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='orange',
                         label='Review Author Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='mediumaquamarine',
                         label='Product Listing Characteristics',markersize=10),
                       Line2D([0],[0],marker='o',color='cornflowerblue',
                         label='Product Types',markersize=10)]
    ax.legend(handles=legend_elements, prop={'size': 14})

    return plt.show()

coefplot(result4)