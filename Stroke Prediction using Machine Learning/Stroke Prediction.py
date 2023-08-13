#!/usr/bin/env python
# coding: utf-8

# # Stroke Prediction

# ## Importing libraries

# In[42]:


import os
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ## Creating data frame 

# In[7]:


print('Current project directory:', os.getcwd())


# In[8]:


file_path = 'healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)
df.head()


# In[9]:


df.describe()


# **Elucidate:**
# 
# The last column "stroke" is the response column whereas rest of the columns are predictors. They are independent to each other but response column can be dependent on the predictors. 0 and 1 represent whether a patient had stroke or not, specifically 0 represents no stroke and represents yes stroke.

# In[10]:


df.info()


# - The shape of the data set before pre-processing is 5110 rows * 12 columns.
# - Column index represents the response and the rest represent predictors
# - There are 5 categorical features and the rest are numerical ones
# - All the columns have either values (numerical/categorical) or null replacers such as 'NaN', '', etc.. We are unable to confirm that the dataset doesn't contain any missing values.
# - Visually, we see that predictor "bmi" has 201 not applicable values. Since the number is a way lesser than 30% of the total, we delete the respective rows.

# In[12]:


print('Feature column names:', df.loc[:, df.columns != 'stroke'].columns)

print('\nTarget column labels:', df['stroke'].unique())


# In[13]:


df.isnull().sum()


# In[14]:


df = df.dropna()
#df = df.reset_index()
print('New shape:', df.shape)


# * The missing values are dropped and then the dataframe is reindex.

# ## Data pre-processing

# ### Outlier detection and handling

# In[15]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

sns.boxplot(data = df[['age', 'avg_glucose_level', 'bmi']], orient = 'v')


# * With a simple box plot, one can detect outliers.
# * An outlier really affects a model performance, so most common method of handing it is to remove it.
# * We are doing box plots for there columns such as "age", "avg_glucose_level", and "bmi". Visually we conclude that column age doesn't have an outlier but columns avg_glucose_level and bmi have many outliers which need to handled before fitting the data with a model 

# In[16]:


# define the lower and upper bounds for outliers
q1 = df['avg_glucose_level'].quantile(0.25)
q3 = df['avg_glucose_level'].quantile(0.75)

iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

df = df[(df['avg_glucose_level'] >= lower_bound) & (df['avg_glucose_level'] <= upper_bound)]
df


# 568 rows are filter based on the outlier detection criteria as these data points are placed not within the lower bound and upper bound.

# In[17]:


# define the lower and upper bounds for outliers
q1 = df['bmi'].quantile(0.25)
q3 = df['bmi'].quantile(0.75)

iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]
df


# 90 rows were removed as soon as we knew that they were outliers. The number is approximately 6.3 times smaller tha the rows of average level of gluscose column.

# In[21]:


df = df.drop(['id'], axis = 1)


# In[23]:


duplicate_rows = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows.shape)


# In[24]:


plt.style.use("seaborn")
plt.subplots_adjust(hspace=0.2)
color = 'winter'

sns.histplot(data = df, kde=True, palette = 'hls',  x = 'work_type', hue = 'ever_married', shrink = .9)


# * More than 1600 people who were not bachelor and employed privately had stroke. Around 750 bachelor people who worksed private had stroke.
# * Greater than 450 people who were not bachelor and have Government jobs has stroke. More than 100 people who were bachelors had stroke.
# * More than 500 people who were self employeed and married had stroke. Greater than 100 people who were never married had stroke. 
# * From all the above observations, we can clearly note that married people with atleast 1 stroke in their medical history is twice as the bachelor people.  
# * There was 0 case under married childern catergory. It is now evident that child marriage is not happening in the population. 
# * People who never worked didn't have stroke attack. We see that people never worked will not have stroke. 
# * People in private job type has a huge number of stroke cases when campared to Government job and self-employed people. 

# In[25]:


sns.displot(df, x = 'gender', kde=True, palette = 'hls', hue = 'ever_married', shrink = .9)


# * The density of females who are married have high chances of getting a stroke because in this sample married females had stroke with significantly larger value than married male. 
# 
# * The density of female bachelors has 600 to 700 strokes more than male bachelors.

# In[26]:


sns.displot(df, x = 'smoking_status', kde=True, palette = 'hls', hue = 'ever_married', shrink = .9)
plt.show()


# * Surprisingly, the group of people who regularly smoked had less number of stroke. Among them, bachelors were less than 40% of the total people.
# * Contradictoryly, married people who never smoked had srokes higher in number than all of smoking status. Among them, again bachelors were less than 45% of the total people.
# * We easily conclude based on the sample that person who doesn't smoke may have stroke with high chances than who smokes.

# In[27]:


le = preprocessing.LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['work_type'] = le.fit_transform(df['work_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])


# ##### Label encoding: 
# Columns such as gender, ever_married, residence_type, work_type, and smoking status are categorical by default. In order to put use them in prediction, they have to undergo label encoding transformation.

# In[28]:


sns.heatmap(df)
plt.show()


# The above heatmap plot shows columns that have high variances. Columns age, average_glucose_level, and bmi have high variances. Hence they are visually appearable.

# In[29]:


corr = df.corr()
sns.heatmap(corr, cmap= "PuBu", annot = True, annot_kws={'size': 7})
corr


# This correlation map shows degree of similarity and relationship between all the features. Variables ever_married and age have the highest correlation cofficient as 0.69. Variables stroke and residence_type have the least correlation coefficient value as 0.000023.

# In[30]:


scale = StandardScaler()

variables = df[['age', 'avg_glucose_level', 'bmi']]
scaled_variables = scale.fit_transform(variables)

df['age'] = scaled_variables[:, 0]
df['avg_glucose_level'] = scaled_variables[:, 1]
df['bmi'] = scaled_variables[:, 2]


# ## Prediction with classifiers

# In[35]:


y, X = df['stroke'], df.drop('stroke', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# In[36]:


classifier = LazyClassifier(verbose = 0, ignore_warnings = True, custom_metric = None)
models, predictions = classifier.fit(X_train, X_test, y_train, y_test)

print(models)


# Here we see that XGBClassifier performs quite good in all the performance metrics incluidng F1 score. So we take it as an ideal classifier for the classification task.

# In[37]:


ros = RandomOverSampler(sampling_strategy = 0.5)
X_resampled, y_resampled = ros.fit_resample(X, y)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size = 0.3)

X_resampled.to_csv('X_resampled.csv')
y_resampled.to_csv('y_resampled.csv')


# In[38]:


model = xgb.XGBClassifier(n_estimators = 20)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[45]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)
print("Confusion Matrix:", '\n')
print(conf_matrix)


# # END 
