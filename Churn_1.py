#!/usr/bin/env python
# coding: utf-8

# In[1]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#!pip install dataprep
from dataprep.eda import create_report
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
#!pip install category_encoders
import category_encoders as ce
#!pip install imblearn
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
#!pip install xgboost
from xgboost import XGBClassifier

#Grid search
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import *


# In[2]:


#Train data
train = pd.read_csv('C:/Users/gdiaz/Desktop/GDA/kaggle/Churn/customer-churn-prediction-2020/train.csv')
#Test data
test = pd.read_csv('C:/Users/gdiaz/Desktop/GDA/kaggle/Churn/customer-churn-prediction-2020/test.csv')


# In[3]:


#Report from dataprep.eda
create_report(train)


# In[4]:


plt.hist(train['total_intl_calls'])
#Transform total_intl_calls to sqrt to approximate to a normal distribution
train['sqrt_total_intl_calls'] = np.sqrt(train['total_intl_calls'])
test['sqrt_total_intl_calls'] = np.sqrt(test['total_intl_calls'])


# In[5]:


plt.hist(train['sqrt_total_intl_calls'])


# In[6]:


#sns.boxplot(train.account_length, groupby=train.churn)

#test['churn2'] = np.where(test['churn']=='yes', 1, 0)
#sns.boxplot(train.account_length,groupby=train.churn2)
#train.boxplot(column = ['account_length'],by='churn2')


# In[7]:


#for col in train.columns:
#    print(col)
col_real = list([1,*range(5,19),20])
col_real = np.array(col_real)

#list(range(5,20))
#Now I have the real vectors I am going to make a boxplot of each real variable groupby Churn
for i in col_real:
    var = train.iloc[:,i]
    churn = train.churn
    sns.boxplot(x=churn,y=var,orient='v')
    plt.show()


# In[8]:


#train['churn'] = np.where(train['churn']=='yes', 1, 0)
sns.countplot(x="state", hue="churn", data=train)
plt.show()
sns.countplot(x="area_code", hue="churn" ,data=train)
plt.show()
sns.countplot(x="international_plan", hue="churn", data=train)
plt.show()
sns.countplot(x="voice_mail_plan", hue="churn", data=train)
plt.show()


# # Bivariate Analysis
# https://www.kaggle.com/dileepsahu/customer-churn-prediction-with-96-44-accuracy/notebook

# In[9]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'account_length').add_legend()
plt.title('Churn rate VS account_length')


# In[10]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'number_vmail_messages').add_legend()
plt.title('Churn rate VS number_vmail_messages')
plt.show()


# In[11]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_day_minutes').add_legend()
plt.title('Churn rate VS total day minutes')
plt.show()


# In[12]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_day_calls').add_legend()
plt.title('Churn rate VS total day calls')
plt.show()


# In[13]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_day_charge').add_legend()
plt.title('Churn rate VS total day charge')
plt.show()


# In[14]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_eve_minutes').add_legend()
plt.title('Churn rate VS total evening minutes')
plt.show()


# In[15]:


sns.FacetGrid(train, hue='churn',size=5).map(sns.distplot, 'total_eve_calls').add_legend()
plt.title('Churn rate VS total evening calls')
plt.show()


# In[16]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_night_charge').add_legend()
plt.title('Churn rate VS total night charge')
plt.show()


# In[17]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_intl_minutes').add_legend()
plt.title('Churn rate VS total international minutes')
plt.show()


# In[18]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_intl_calls').add_legend()
plt.title('Churn rate VS total international calls')
plt.show()


# In[19]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'sqrt_total_intl_calls').add_legend()
plt.title('Churn rate VS total international calls')
plt.show()


# In[20]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'total_intl_charge').add_legend()
plt.title('Churn rate VS total international charge')
plt.show()


# In[21]:


sns.FacetGrid(train, hue='churn',size=7).map(sns.distplot, 'number_customer_service_calls').add_legend()
plt.title('Churn rate VS Number of customer service calls')
plt.show()


# # Outlier detection

# In[22]:


#Extract the numerical features from the dataset
num_var = [feature for feature in train.columns if train[feature].dtypes != 'O']
for feature in num_var:
    if feature != 'churn':
        sns.boxplot(x ='churn', y = feature, data = train)
        plt.title(feature)
        plt.show()


# # Removing outliers

# In[23]:


#functions for removing outliers
def remove_outliers(train,labels):
    for label in labels:
        q1 = train[label].quantile(0.25)
        q3 = train[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        train[label] = train[label].mask(train[label]< lower_bound, train[label].median(),axis=0)
        train[label] = train[label].mask(train[label]> upper_bound, train[label].median(),axis=0)

    return train


# In[24]:


train = remove_outliers(train, num_var)


# In[25]:


#after removing the outliers we have to see the outliers
for feature in num_var:
    if feature != 'churn':
        sns.boxplot(x ='churn', y = feature, data = train)
        plt.title(feature)
        plt.show()


# In[26]:


#state feature has 51 different category so we can't converted into onehot encoder
#that is it create 51 different features so it leads to overfitting so I will use the hashing encoding for state featuer.

hash_state = ce.HashingEncoder(cols = 'state')
train = hash_state.fit_transform(train)
test = hash_state.transform(test)
#Search for the HashingEncoder


# In[27]:


# replace no to 0 and yes to 1
train.international_plan.replace(['no','yes'],[0,1],inplace = True)
train.voice_mail_plan.replace(['no','yes'],[0,1],inplace=True)
train.churn.replace(['no','yes'],[0,1],inplace = True)
test.international_plan.replace(['no','yes'],[0,1],inplace = True)
test.voice_mail_plan.replace(['no','yes'],[0,1],inplace = True)


# In[28]:


# converting the area_code to numerical variable using one-hot encoder
onehot_area = OneHotEncoder()
onehot_area.fit(train[['area_code']])

# Train
encoded_values = onehot_area.transform(train[['area_code']])
train[onehot_area.categories_[0]] = encoded_values.toarray()
train = train.drop('area_code', axis=1)

# Test
encoded_values = onehot_area.transform(test[['area_code']])
test[onehot_area.categories_[0]] = encoded_values.toarray()
test = test.drop('area_code', axis=1)


# In[29]:


# showing the imbalanced class
sns.countplot(x = 'churn', data = train)
plt.show()


# In[30]:


x = train.drop('churn',axis=1).values
y = train.churn.values
id_submission = test.id
test = test.drop('id', axis=1)
# spliting the data into test and train
x_train, x_test , y_train, y_test = train_test_split(x, y , test_size=0.3, random_state=0)


# In[31]:


print('Before upsampling count of label 0 {}'.format(sum(y_train==0)))
print('Before upsampling count of label 1 {}'.format(sum(y_train==1)))
# Minority Over Sampling Technique
sm = SMOTE(sampling_strategy = 1, random_state=1)   
x_train_s, y_train_s = sm.fit_resample(x_train, y_train.ravel())
                                         
print('After upsampling count of label 0 {}'.format(sum(y_train_s==0)))
print('After upsampling count of label 1 {}'.format(sum(y_train_s==1)))


# In[32]:


# creating the object of minmax scaler
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
test = scaler.transform(test)


# # Building the model

# In[33]:


clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.7, 
                        subsample=0.8, nthread=10, learning_rate=0.01)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print('Accuracy: ')
print('{}'.format(accuracy_score(y_test, y_pred)))
print('Classification report: ')
print('{}'.format(classification_report(y_test, y_pred)))
print('Confusion Matrix')
print('{}'.format(confusion_matrix(y_test, y_pred)))
print('Cohen kappa score: ')
print('{}'.format(cohen_kappa_score(y_test, y_pred)))


# # Submission

# In[34]:


y_pred_sub = clf.predict(test)
submit = pd.DataFrame({'id':id_submission, 'churn':y_pred_sub})
submit.head()
# replace 0 to no and 1 to yes
submit.churn.replace([0,1],['no','yes'], inplace=True)
submit.to_csv('churn_submit.csv',index=False)


# # Grid Search

# In[35]:


xgb_model = XGBClassifier()

#brute force scan for all parameters, here are the tricks
#usually max_depth is 6,7,8
#learning rate is around 0.05, but small changes may make big diff
#tuning min_child_weight subsample colsample_bytree can have 
#much fun of fighting against overfit 
#n_estimators is how many round of boosting
#finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.005,0.01,0.05,0.1,0.15,0.2], #so called `eta` value
              'max_depth': [4,5,6,7],
              'min_child_weight': [11],
              'silent': [1],
              'subsample': [0.8],
              'colsample_bytree': [0.7],
              'n_estimators': [5,6,7,200,300], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed': [1337]}


clf = GridSearchCV(xgb_model, parameters, n_jobs=5, 
                   cv=StratifiedKFold(n_splits=5, shuffle=True), 
                   scoring='roc_auc',
                   verbose=2, refit=True)

clf.fit(x_train, y_train)

#trust your CV!
#best_parameters, score, _ = max(clf.cv_results_, key=lambda x: x[1])
#print('Raw AUC score:', score)
#for param_name in sorted(best_parameters.keys()):
#    print("%s: %r" % (param_name, best_parameters[param_name]))

y_pred = clf.predict_proba(x_test)
y_pred = clf.predict(x_test)
print('Accuracy: ')
print('{}'.format(accuracy_score(y_test, y_pred)))
print('Classification report: ')
print('{}'.format(classification_report(y_test, y_pred)))
print('Confusion Matrix')
print('{}'.format(confusion_matrix(y_test, y_pred)))
print('Cohen kappa score: ')
print('{}'.format(cohen_kappa_score(y_test, y_pred)))


# In[36]:


#xgb_model = XGBClassifier()
xgb_estimator = XGBClassifier(objective='binary:logistic',
                                  seed=24,
                                  subsample=0.9,
                                  colsample_bytree=0.5)


# In[37]:


param_grid = {
    'max_depth' : [4,5,6,7],
    'objective':['binary:logistic'],
    'learning_rate': [0.005,0.01,0.05,0.1,0.15,0.2],
    'n_estimators' : [5,6,7,200,300],
    'subsample': [0.8],
    'colsample_bytree': [0.7],
    'max_depth': [4,5,6,7,8,9,10],
    'min_child_weight': [11]
}

clf_xgb_tuned = GridSearchCV(estimator=xgb_estimator,
                             param_grid=param_grid,
                             scoring='roc_auc',
                             verbose=2,
                             n_jobs=-1,
                             cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
)


# In[38]:


clf_xgb_tuned.fit(x_train,
                  y_train,
                  verbose=True,
                  early_stopping_rounds=10,
                  eval_metric='aucpr',
           eval_set=[(x_test, y_test)]
)


# In[39]:


y_pred = clf_xgb_tuned.predict(x_test)
print('Accuracy: ')
print('{}'.format(accuracy_score(y_test, y_pred)))
print('Classification report: ')
print('{}'.format(classification_report(y_test, y_pred)))
print('Confusion Matrix')
print('{}'.format(confusion_matrix(y_test, y_pred)))
print('Cohen kappa score: ')
print('{}'.format(cohen_kappa_score(y_test, y_pred)))


# In[40]:


print(classification_report(y_pred, y_test))


# In[41]:


# Submission
clf = XGBClassifier(max_depth=7, n_estimators=200, colsample_bytree=0.7, 
                        subsample=0.8, nthread=10, learning_rate=0.01)
clf.fit(x_train, y_train)


# In[42]:


y_pred_sub = clf.predict(test)


# In[43]:


submit = pd.DataFrame({'id':id_submission, 'churn':y_pred_sub})
submit.head()


# In[44]:


# replace 0 to no and 1 to yes
submit.churn.replace([0,1],['no','yes'], inplace=True)
submit.to_csv('churn_submit2.csv',index=False)


# In[ ]:




