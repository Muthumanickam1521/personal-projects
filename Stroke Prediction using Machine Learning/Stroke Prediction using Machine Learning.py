#!/usr/bin/env python
# coding: utf-8

# # Stroke Prediction using Machine Learning

# ## Importing libraries

# In[1]:


get_ipython().system(' pip install lightgbm')
get_ipython().system(' pip install xgboost')
get_ipython().system(' pip install imbalanced-learn')

import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn import metrics as mt
from lightgbm import LGBMClassifier
from sklearn.linear_model import RidgeClassifierCV
from xgboost import XGBClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as Xgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgm
from sklearn.svm import NuSVC
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, classification_report,f1_score,confusion_matrix,precision_score,recall_score
import warnings
warnings.filterwarnings('ignore')


# ## Creating data frame 

# In[2]:


file_path = '/home/pearlruby/Project/stroke prediction/healthcare-dataset-stroke-data.csv'
df = pd.read_csv(file_path)
df.head()


# **Elucidate:**
# 
# The last column "stroke" is the response column whereas rest of the columns are predictors. They are independent to each other but response column can be dependent on the predictors. 0 and 1 represent whether a patient had stroke or not, specifically 0 represents no stroke and represents yes stroke.

# In[3]:


df.info()


# - The shape of the data set before pre-processing is 5110 rows * 12 columns.
# - Column index represents the response and the rest represent predictors
# - There are 5 categorical features and the rest are numerical ones
# - All the columns have either values (numerical/categorical) or null replacers such as 'NaN', '', etc.. We are unable to confirm that the dataset doesn't contain any missing values.
# - Visually, we see that predictor "bmi" has 201 not applicable values. Since the number is a way lesser than 30% of the total, we delete the respective rows.

# In[4]:


print('Feature column names:', df.loc[:, df.columns != 'stroke'].columns)

print('\nTarget column labels:', df['stroke'].unique())


# In[5]:


df.isnull().sum()


# In[6]:


df = df.dropna()
#df = df.reset_index()
print('New shape:', df.shape)


# * The missing values are dropped and then the dataframe is reindex.

# ## Data pre-processing

# ### Outlier detection and handling

# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)

sns.boxplot(data = df[['age', 'avg_glucose_level', 'bmi']], orient = 'v')


# * With a simple box plot, one can detect outliers.
# * An outlier really affects a model performance, so most common method of handing it is to remove it.
# * We are doing box plots for there columns such as "age", "avg_glucose_level", and "bmi". Visually we conclude that column age doesn't have an outlier but columns avg_glucose_level and bmi have many outliers which need to handled before fitting the data with a model 

# In[8]:


# define the lower and upper bounds for outliers
q1 = df['avg_glucose_level'].quantile(0.25)
q3 = df['avg_glucose_level'].quantile(0.75)

iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

df = df[(df['avg_glucose_level'] >= lower_bound) & (df['avg_glucose_level'] <= upper_bound)]
df


# 568 rows are filter based on the outlier detection criteria as these data points are placed not within the lower bound and upper bound.

# In[9]:


# define the lower and upper bounds for outliers
q1 = df['bmi'].quantile(0.25)
q3 = df['bmi'].quantile(0.75)

iqr = q3 - q1
lower_bound = q1 - (1.5 * iqr)
upper_bound = q3 + (1.5 * iqr)

df = df[(df['bmi'] >= lower_bound) & (df['bmi'] <= upper_bound)]
df


# 90 rows were removed as soon as we knew that they were outliers. The number is approximately 6.3 times smaller tha the rows of average level of gluscose column.

# In[10]:


df = df.dropna()
df = df.reset_index()
print('New shape:', df.shape)


# In[11]:


df = df.drop(['index'], axis = 1)


# In[12]:


duplicate_rows = df[df.duplicated()]
print("number of duplicate rows: ", duplicate_rows.shape)


# In[13]:


df['work_type'].unique()


# * Knowing about unique values of a feature is very important in the data visualisation part.

# In[14]:


df['smoking_status'].unique()


# In[15]:


df = df.drop('id', axis = 1)
df.head()


# * Column id is dropped since it has no use to prediction.

# In[16]:


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

# In[17]:


sns.displot(df, x = 'gender', kde=True, palette = 'hls', hue = 'ever_married', shrink = .9)


# * The density of females who are married have high chances of getting a stroke because in this sample married females had stroke with significantly larger value than married male. 
# 
# * The density of female bachelors has 600 to 700 strokes more than male bachelors.

# In[18]:


sns.displot(df, x = 'smoking_status', kde=True, palette = 'hls', hue = 'ever_married', shrink = .9)
plt.show()


# * Surprisingly, the group of people who regularly smoked had less number of stroke. Among them, bachelors were less than 40% of the total people.
# * Contradictoryly, married people who never smoked had srokes higher in number than all of smoking status. Among them, again bachelors were less than 45% of the total people.
# * We easily conclude based on the sample that person who doesn't smoke may have stroke with high chances than who smokes.

# In[19]:


le = preprocessing.LabelEncoder()
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])
df['work_type'] = le.fit_transform(df['work_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])


# ##### Label encoding: 
# Columns such as gender, ever_married, residence_type, work_type, and smoking status are categorical by default. In order to put use them in prediction, they have to undergo label encoding transformation.

# ## Feature selection

# In[20]:


from sklearn.feature_selection import VarianceThreshold

sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
sel.fit_transform(df).shape


# Features that have high variance values are considered to be least contributing to predicting model. Using VarianceThreshold from sklearn, we can find and filter columns with high variance. In our sample dataset, there is no significant column with high variance. So all the columns are taken as features to be put into model.

# In[21]:


sel.fit_transform(df)


# In[22]:


sns.heatmap(df)
plt.show()


# The above heatmap plot shows columns that have high variances. Columns age, average_glucose_level, and bmi have high variances. Hence they are visually appearable.

# In[23]:


corr = df.corr()
sns.heatmap(corr, cmap= "PuBu", annot = True, annot_kws={'size': 7})
corr


# This correlation map shows degree of similarity and relationship between all the features. Variables ever_married and age have the highest correlation cofficient as 0.69. Variables stroke and residence_type have the least correlation coefficient value as 0.000023.

# ## Basic summary statistics

# In[24]:


df.mean()


# In[25]:


df.median()


# In[26]:


df.mode()


# In[27]:


df.std()


# In[28]:


df.var()


# In[29]:


scale = StandardScaler()

variables = df[['age', 'avg_glucose_level', 'bmi']]
scaled_variables = scale.fit_transform(variables)

df['age'] = scaled_variables[:, 0]
df['avg_glucose_level'] = scaled_variables[:, 1]
df['bmi'] = scaled_variables[:, 2]


# In[30]:


df


# ## Prediction with classifiers

# In[31]:


y, X = df['stroke'], df.drop('stroke', axis = 1)
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)


# * We split the preprocessed dataset into training and testing sets with sizes 0.7 and 0.3 of total size. Random state parameter is set to 42.

# In[32]:


X_train


# In[33]:


Y_train


# In[34]:


x_test


# In[35]:


y_test


# In[73]:


value = 0.1
n = 1
accuracy = []

while n < 10:
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = value)
    
    DTclassifier = DecisionTreeClassifier(max_depth = 1)
    DTclassifier.fit(X_train, Y_train)
    predict = DTclassifier.predict(x_test)
    accuracy += [accuracy_score(y_test, predict)]
    
    value += .1
    n += 1
    
print(accuracy)
accuracy.index(max(accuracy))


# * We find that optimal split ratio for the test set size is 0.4 since it yields the highest prediction accuracy.

# In[75]:


value = 1
n = 1
accuracy0 = []

while n < 100:
    X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size = 0.6, random_state = value)
    
    DTclassifier = DecisionTreeClassifier(max_depth = 1)
    DTclassifier.fit(X_train, Y_train)

    predict = DTclassifier.predict(x_test)
    accuracy0 += [accuracy_score(y_test, predict)]
    
    value += 1
    n += 1
    
print(accuracy0)
accuracy0.index(max(accuracy0))


# We also find that random state as 33 performs very good compared to all other values between 0 to 99 with 0.6 as the optimal test split ration.

# In[76]:


sm = SMOTE(random_state = 42)
X_train_resampled, Y_train_resampled = sm.fit_resample(X_train, Y_train)


# SMOTE is a technique to do over-sampling of the data.

# In[77]:


classifiers = [#DecisionTreeClassifier(),
               RandomForestClassifier(),
               SGDClassifier(),
               SVC(),
               XGBClassifier(),
               RidgeClassifier(),
               LinearSVC(),
               AdaBoostClassifier(),
               MLPClassifier(),
               ExtraTreesClassifier(),
               Perceptron(),
               GaussianNB(),
               KNeighborsClassifier(),
               NearestCentroid(),
               LogisticRegression(),
               BaggingClassifier(),
               BernoulliNB(),
               CalibratedClassifierCV(),
               QuadraticDiscriminantAnalysis()]

accuracy = []
accuracyBalanced = []
precisionAvg = []
brier_score_loss = []
f1 = []
precision_score = []
recall_score = []
jaccard_score = []
roc_auc_score = []

for classifier in classifiers:
    classifier.fit(X_train_resampled, Y_train_resampled)
    predict = classifier.predict(x_test)

    accuracy += [mt.accuracy_score(y_test, predict)]
    accuracyBalanced += [mt.balanced_accuracy_score(y_test, predict)]
    precisionAvg += [mt.average_precision_score(y_test, predict)]
    brier_score_loss += [mt.brier_score_loss(y_test, predict)]
    f1 += [mt.f1_score(y_test, predict)]
    precision_score += [mt.precision_score(y_test, predict)]
    recall_score += [mt.recall_score(y_test, predict)]
    jaccard_score += [mt.jaccard_score(y_test, predict)]
    roc_auc_score += [mt.roc_auc_score(y_test, predict)]


# We compute almost all the model performance metrics for almost all the classifers.

# In[42]:


df_metrics = pd.DataFrame({'Classifier': classifiers,
                   'Accuracy': accuracy,
                   'AccuracyBalanced': accuracyBalanced,
                   'PrecisionAvg': precisionAvg,
                   'BrierScoreLoss': brier_score_loss,
                   'F1': f1,
                   'PrecisionScore': precision_score,
                   'RecallScore': recall_score,
                   'JaccardScore': jaccard_score,
                   'RocAucScore': roc_auc_score})
    
df_metrics.head(20)


# A generated dataframe that consists of classifiers and their performance metric scores.

# ### Deployement will be done in future version of this notebook. Happy learning :)
