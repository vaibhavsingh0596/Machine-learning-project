#!/usr/bin/env python
# coding: utf-8

# # Backgroung Story
# 
# Invistico is one of the Airlines' company who wants to span their wings further in airline industries. Invistico is aware that Service quality will build a brand, which results in Customer Satisfaction and Customer Loyalty. Invistico needs a Business Recommendations and insights to Retain their Customers without excessive costs for the required Business Idea.

# # Problem Statement
# 
# The main purpose of this dataset is to predict whether a future customer would be satisfied with their service given the details of the other parameters values.
# 
# Also the airlines need to know on which aspect of the services offered by them have to be emphasized more to generate more satisfied customers.

# In[2]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[3]:


flight=pd.read_csv('C:/Users/Dell/Downloads/Invistico_Airline.csv')


# In[4]:


flight.info()


# In[5]:


flight.head(n=20)


# In[6]:


flight['Seat comfort'].value_counts()


# In[7]:


flight.describe()


# In[8]:


flight.describe(include=object)


# In[9]:


flight.isnull().sum()


# In[10]:


#data cleaning and preprocessing
flight.fillna(flight['Arrival Delay in Minutes'].mean(), inplace=True)
flight.isnull().sum()


# In[11]:


flight.shape


# In[12]:


#EDA
fig,axs = plt.subplots(2,2,figsize=(10, 10))
cols=['Gender', 'Customer Type', 'Type of Travel', 'Class']
c=0
for i in range(2):
  for j in range(2):
    sns.countplot(data=flight,x=cols[c],hue='satisfaction',ax=axs[i][j])
    axs[i][j].set_title('\nCustomer Satisafaction as per {}'.format(cols[c]))
    c+=1
#EDA
fig,axs = plt.subplots(2,2,figsize=(10, 10))
cols=['Gender', 'Customer Type', 'Type of Travel', 'Class']
c=0
for i in range(2):
  for j in range(2):
    sns.countplot(data=flight,x=cols[c],hue='satisfaction',ax=axs[i][j])
    axs[i][j].set_title('\nCustomer Satisafaction as per {}'.format(cols[c]))
    c+=1


# In[13]:


sns.displot(flight,x='Age',binwidth=5,hue='satisfaction')
plt.show()


# # Countplot  and distributionplot conclusions :
# 
#     Female Customers have higher satisfaction than Male Customers.
#     Loyal Customers have higher satisfaction than Disloyal Customers.
#     Business Travel has higher customer satisfaction than Personal Travel.
#     Business Class has the highest satisfaction between the 3 airlines classes.
#     Customers of age group between 40 to 60 are more satisfied than customers of other age group.

# In[14]:


print("Gender:",flight['Gender'].unique())
print("Customer Type:",flight['Customer Type'].unique())
print("Type of Travel:",flight['Type of Travel'].unique())
print("class:",flight['Class'].unique())
print("satisfaction:",flight['satisfaction'].unique())

le=LabelEncoder()
flight['Gender']= le.fit_transform(flight['Gender']) 
flight['Customer Type']= le.fit_transform(flight['Customer Type']) 
flight['Type of Travel']= le.fit_transform(flight['Type of Travel']) 
flight['Class']= le.fit_transform(flight['Class']) 
flight['satisfaction']= le.fit_transform(flight['satisfaction']) 

print("\nGender:",flight['Gender'].unique())
print("Customer Type:",flight['Customer Type'].unique())
print("Type of Travel:",flight['Type of Travel'].unique())
print("class:",flight['Class'].unique())
print("satisfaction:",flight['satisfaction'].unique())


# In[17]:


temp1 = flight.drop('satisfaction',axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = temp1.columns
vif_data["VIF"] = [variance_inflation_factor(temp1.values, i) for i in range(len(temp1.columns))]
vif_data


# In[18]:


flight=flight.drop(['Ease of Online booking','Cleanliness','Baggage handling'],axis=1)


# In[19]:


temp1 = flight.drop('satisfaction',axis=1)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = temp1.columns
vif_data["VIF"] = [variance_inflation_factor(temp1.values, i) for i in range(len(temp1.columns))]
vif_data


# In[20]:


plt.figure(figsize=(18,10))
sns.heatmap(flight.corr(), cmap='Reds', annot=True, fmt='.2f')


# In[21]:


nonbinary_columns = [column for column in flight.columns if len(flight[column].unique()) > 3]
plt.figure(figsize=(20, 20))

for i, column in enumerate(nonbinary_columns):
    plt.subplot(4, 6, i + 1)
    sns.boxplot(data=flight[column], palette='Blues')
    plt.title(column)

plt.suptitle('Boxplots With Outliers', size=30)
plt.show()


# In[22]:


#outlier Removal using IQR

flight = flight[flight['Flight Distance']<=4000]
flight = flight[flight['On-board service']>1]
flight = flight[flight['Checkin service']>1]
flight = flight[flight['Departure Delay in Minutes']<= 50]
flight = flight[flight['Arrival Delay in Minutes']<= 50]


# In[23]:


nonbinary_columns = [column for column in flight.columns if len(flight[column].unique()) > 3]
plt.figure(figsize=(20, 20))

for i, column in enumerate(nonbinary_columns):
    plt.subplot(4, 6, i + 1)
    sns.boxplot(data=flight[column], palette='Blues')
    plt.title(column)

plt.suptitle('Boxplots Without Outliers', size=30)
plt.show()


# In[24]:


x=flight.drop('satisfaction',axis=1)
y=flight['satisfaction']
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)
x_train.shape,x_test.shape


# In[25]:


#Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,x)
lr=logit_model.fit()
lr.summary()


# # Logistic Regression Model Fitting

# In[26]:


from sklearn import metrics
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_test = logreg.predict(x_test)
y_pred_train = logreg.predict(x_train)


# In[44]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print(confusion_matrix)

sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[49]:


print(" Test Accuracy:",metrics.accuracy_score(y_test, y_pred_test))
print(" Train Accuracy:",metrics.accuracy_score(y_train, y_pred_train))

print("\nPrecision:",metrics.precision_score(y_test, y_pred_test))
print("Recall:",metrics.recall_score(y_test, y_pred_test))


# In[29]:


from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(x_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# # Decesion Tree

# In[35]:


# Create Decision Tree classifer object
Dtc = DecisionTreeClassifier()

# Train Decision Tree Classifer
Dtc= Dtc.fit(x_train,y_train)

#Predict the response for test dataset
y_pred = Dtc.predict(x_test)
y_pred1 = Dtc.predict(x_train)


# In[52]:


# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred1))

print("\nPrecision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# # RANDOM FOREST

# In[38]:


classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(x_train, y_train)
y_pred_c1 = classifier.predict(x_test)
y_pred_c2 = classifier.predict(x_train)


# In[54]:


# Model Accuracy, how often is the classifier correct?
print("Test Accuracy:",metrics.accuracy_score(y_test, y_pred_c1))
print("Train Accuracy:",metrics.accuracy_score(y_train, y_pred_c2))

print("\nPrecision:",metrics.precision_score(y_test, y_pred_c1))
print("Recall:",metrics.recall_score(y_test, y_pred_c1))


# In[47]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

result = confusion_matrix(y_test, y_pred_c1)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred_c1)
print("Classification Report:",)
print (result1)


# # Conclusion 1:
# After using Machine Learning to analyze customer satisfaction, we find that Logistic regression is the best machine learning model to predict our customer satisfaction data.
# 
# This model isn't Overfitting or Underfitting since the the accuracy differences between train and test data is almost equal.
# 

# In[83]:


# get importance
importance = logreg.coef_[0]

# summarize feature importance
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))

    # plot feature importance
plt.figure(figsize=(15,8))
plt.bar(x.columns,importance)
plt.xticks(rotation=90)
plt.show()


# # Conclusion 2:
# There are 4 services that are highly affects customer satisfaction in this Airlines data :
# 
#     Inflight Entertainment
#     online support
#     leg room service
#     seat comfort
# Invistico Airlines can choose to upgrade/investing more money and effort in those 4 services to improve their customer satisfactions.
