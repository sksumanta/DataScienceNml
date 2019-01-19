# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 23:38:55 2018

@author: Sumanta
"""

'''
The Titanic data file is 
    E:/datascienceProject/DatascienceNml/MLProjects/titanicDataAnalysis/titanicData/titanic.csv
having below fields 

 PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked

PassengerId : Serial no assigned to each passanger
Survived : Passanger servived or not at titanic desaster (1 - servived, 0 - not)
Pclass : Passanger class (1-- 1st class , 2-- 2nd class and 3-- 3rd class)
Name : Name of the Passanger
Sex : Gender of the Passanger
Age : Age of the Passanger
SibSp : No of sibling and spouses aboarded for the trip
Parch : No of parents and children aboarded for the trip
Ticket : Ticket no for the passanger 
Fare : Amount paid by the passanger
Cabin : Cabin of the passanger
Embarked : point of embarkment/aboarding point ( C  , Q , S)

The Titanic Data is a classification problem to detect survived people. 
we have two datasets, train, and test. when we build our model we should 
predict for the new passenger that he or she were been survived or dead.

We are going to perform below steps to predict the dataset.

Data extraction : we'll load the dataset and have a first look at it. 
Cleaning : we'll fill in missing values. 
Plotting : we'll create some interesting charts that will give  
    correlations and hidden insights out of the data. 
Assumptions : we'll formulate hypotheses from the charts.
Build Model : We'll transform the required variables in such a way that, 
    we should able to use a machine learning algorithm by taking the 
    trained data and build a model. So we can predict new passenger.

'''

# Step-1 Read the data from file or from database.


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

thePath = 'E:/datascienceProject/DatascienceNml/MLProjects/titanicDataAnalysis/titanicData/'
theFile = 'titanic.csv'


def readFileToDF(Path,File):
    file=os.path.join(Path, File)
    fileDF = pd.read_csv(file,delimiter=",")
    return fileDF

titanicDF= readFileToDF(thePath,theFile)

print(titanicDF.shape)
print(titanicDF.head())

# step-2 rearrangement of dataframe cloumns

# As we are going to predict the servival of pasanger so I am doing the 
# rearrangement of dataframe cloumns 

titanicDF = titanicDF.loc[:,['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]

# As PassengerId is the serial no so i changed it to row index

titanicDF=titanicDF.set_index(titanicDF.columns[0])
print(titanicDF.shape)
print(titanicDF.head())

# step-3 Take a statistical look to the value of different columns and visualize the distribution of data.

#check the null value. we can use the below code

print(titanicDF.info())  

# The info() we come to know we have 891 entries/rows, index starts 
# from 0 to 890 and each columns data-type and count of not-null values.

#The descibe() method gives to statistics for all numerical columns.
# if we  will use  the argument  include = 'all' the describe method will 
# give the output for catagorical/non numerical columns.

titanicDF.describe( include = 'all' )

'''
        	Pclass	Name	Sex	Age	 SibSp	Parch	 Ticket	 Fare	 Cabin	Embarked  	Survived
count	   891.0	   891	891	714.0	 891.0	891.0	 891	  891.0	 204	  889	      891.0

unique	nan	      891	2 nan  nan nan	681	nan	147	3	nan

top		nan Lesurer, Mr. Gustave J	male	nan  nan nan			347082 nan		B96 B98	S	nan

freq		nan 1	577	nan  nan nan			7	 nan	4	644	nan

mean	2.308641975308642	nan  nan		29.69911764705882	0.5230078563411896	0.38159371492704824		32.2042079685746			nan  nan 0.3838383838383838

std	0.8360712409770513	nan  nan		14.526497332334044	1.1027434322934275	0.8060572211299559		49.693428597180905		nan  nan 	0.4865924542648585

min	1.0			nan  nan 0.42	0.0	0.0		0.0			nan  nan     0.0

25%	2.0			nan  nan 20.125	0.0	0.0		7.9104	nan  nan		0.0

50%	3.0			nan  nan 28.0	0.0	0.0		14.4542		nan  nan 	0.0

75%	3.0			nan  nan 38.0	1.0	0.0		31.0			nan  nan    1.0

max	3.0			nan  nan 80.0	8.0	6.0		512.3292		nan  nan 	1.0

'''

# As we can see the Fare column having more standard deviation 
# so fare may be high for some a passnagers so we can plot a box plot to 
# get the visulaization of fare column distribution.

import matplotlib.pyplot as plt

plt.boxplot(titanicDF['Fare'])
plt.show()

# incase of describe we dont have catagoriwise count 
# To know how many male and female passangers are there we can use 
# value_counts() to get catagoriwise count on SEX column.

titanicDF['Sex'].value_counts()

# To get the proprotion/ratio of male and female passangers we can pass
# normalize=True in value_counts()

titanicDF['Sex'].value_counts(normalize=True)

# Similarly check the passanger class 

titanicDF['Pclass'].value_counts() 
titanicDF['Pclass'].value_counts(normalize=True)

#Visualize the count of passanger class in bar graph

ax = titanicDF['Pclass'].value_counts().plot(kind='bar' , 
                                    figsize=(15,10), 
                                    title = "count of passanger class" )

ax.set_xlabel("pclass")
ax.set_ylabel("count")

# Similarly check the passanger's servied count 

titanicDF['Survived'].value_counts()
titanicDF['Survived'].value_counts(normalize=True)

#visualization for fare column
ax = titanicDF['Fare'].plot(kind='hist',
                          figsize=(15,10),
                          title = "Passanger fare", bins = 10 )

ax.set_xlabel("passanger fare") 
ax.set_ylabel("No of passanger")

#visualization of passanger class and fare columns, I am using scatter plot.

ax = titanicDF.plot.scatter(x='Pclass' , y='Fare' ,
                            title = "Fare wrt passanger class")
ax.set_xlabel("Passanger class")
ax.set_ylabel("Fare")   # from the visualizaion we get one passanger paid 

# high fare for first class.

# To check the average age of the passanger per  'passanger class' and sex
# I could use any of this,  pivot table or groupby.

'''
# for pivot table 
titanicDF.pivot_table( index='Sex' , columns = 'Pclass' , 
                       values = 'Age' , aggfunc = 'mean'  )
 
  Pclass     1         2         3
Sex                                    
female  34.757904  28.719796  20.454211
male    41.864934  31.083944  25.730884

'''

titanicDF.groupby(['Sex','Pclass']).Age.mean().unstack() # unstack() show output in table format


# Step-4 Delete null data or imputation (fill) of null data. Also deal with outliers.  

#check the null value. we can use the below code

print(titanicDF.info())   # Embarked ,Cabin ,Age  columns has null value

# fill null value for Embarked column.
       
nullRowOfembarked  = titanicDF[titanicDF.Embarked.isnull()]
print(nullRowOfembarked)  # if there is any null value  isnull() will return true 

''' 
Output is 

PassengerId 	Pclass	Name						               Sex		   Age	  SibSp	Parch	Ticket	Fare	Cabin	Embarked	Survived
62	        1	Icard, Miss. Amelie							       female	   38.0	     0	0		113572	78.0	B28		nan 		1
830	     1	Stone, Mrs. George Nelson (Martha Evelyn)	 female	   62.0	     0	0		113572	78.0	B28		nan 		1

Form the output we can see the both passenger were travelling in 1st class with 80.0 fare and survived in the disaster. 

'''
# titanicDF.iloc[829,:]  # embark is  null

# Lets check the data for of passanger class and Embarked wrt mean fare

titanicDF.groupby(['Pclass','Embarked']).Fare.mean().unstack()

'''
Embarked           C          Q          S
Pclass                                    
1         104.718529  90.000000  70.364862
2          25.358335  12.350000  20.327439
3          11.214083  11.183393  14.644083

From the above output we can see the fare given by the passanger is close to
The passanger who aboarded from 'S' embark point.

'''
# Let's fill the passanged embark point as 'S'

titanicDF['Embarked'].fillna('S' , inplace = True)

# titanicDF.iloc[829,:] # now embark is filled with 'S'


# fill null value for Age column.

nullInAge = titanicDF[titanicDF.Age.isnull()]  # For Age 177 row having null value
#print(titanicDF[titanicDF.Age.isnull()].shape)
#nullInAge.shape

# let's apply a predictive model to fill the null value of 'Age' column
# For 'Age' column let's consider linear regression to fill null-value.

notnullData = titanicDF.dropna()
#print(notnullData.shape)
#print(notnullData.head())

#plot the not null Age data  to know the distribution

plt.boxplot(notnullData['Age'])
plt.show()

# rearrange the Pclass	, Survived,	 SibSp, Parch, Fare and 	Age columns to predict age value .

xTrainData = notnullData.loc[:,['Pclass','Survived','SibSp','Parch','Fare']]
yTrainData = notnullData.loc[:,'Age']

#xTestData = nullInAge.loc[:,['Pclass','Survived','SibSp','Parch','Fare']]
xTestData = nullInAge.loc[:,['Pclass','Survived','SibSp','Parch','Fare']]

#xTestData.shape

from sklearn.linear_model import LinearRegression
linReg  = LinearRegression()

linReg.fit( xTrainData, yTrainData)  # model train 

predictAge = pd.DataFrame( linReg.predict(xTestData) , columns=['Age'])

# fill the Age column of titanic data frame 

titanicRowIndex = titanicDF[titanicDF.Age.isnull()].index.values.astype(int)

# to loop through two list we can use zip()

for rowIndex , predAge in zip(titanicRowIndex , predictAge['Age'] ):
    titanicDF.iloc[rowIndex-1 , 3 ] = abs(predAge)

#theAge = titanicDF['Age']

    
# Step-5 Check and treatment of outliers.

# For fare column lets use the box plot to get the visualization of outliers.

plt.boxplot(titanicDF['Fare'])
plt.show()

# plot the histogram to see distribution
ax = titanicDF['Fare'].plot(kind = 'hist' , figsize=(15,10),
                          title = "Passanger fare", bins = 10 )
ax.set_xlabel('no of passanger ')
ax.set_ylabel('fare of passanger')

# check the data what is outlier for Fare column

fareOutlier = titanicDF[titanicDF['Fare'] == titanicDF['Fare'].max()]

'''
From this data we can see the passanger boarded from 'C' embark point and 
blongs to 1st class. so lets transform this to normalize the distribution
# For transformation we will use 'log transformation' as we know Fare will 
not be ngegetive, also for 0 log transformation will be  infinity so add 
one to the each value of Fare before tansformation.
'''
import numpy as np

transformFare = np.log(titanicDF['Fare']+1)

ax = transformFare.plot(kind = 'hist' , figsize =(15,10) 
                                , title="transform fare" , bins = 10)
ax.set_xlabel('no of passanger ')
ax.set_ylabel('fare of passanger')

# lets check the age column

plt.boxplot(titanicDF['Age']) 
plt.show()

# check the data what is outliers for age columns

ageOutliers = titanicDF[titanicDF['Age'] > 70]

# we show only five record is " age greater than 70 " present 

print(ageOutliers)


#Step-5 To do better analysis lets look into the dataset and create new feature
# if required.

# let's consider the age column.
import numpy as np
titanicDF['ageState'] = np.where(titanicDF['Age']<18 , 'Child' , 'Adult')
titanicDF['ageState'].value_counts()

#let's check the servival count for the new feature 'ageState'

titanicDF.groupby(['ageState','Survived']).ageState.count().unstack()


# lets check 'SibSp','Parch' column to create a new feature 

titanicDF['familySize'] = titanicDF['SibSp'] + titanicDF['Parch'] + 1 # 1 is added for person itself

titanicDF['familySize'].max() # maximum size of the family.

# let's create visualization of 'familySize'

ax = titanicDF['familySize'].plot(kind='hist', figsize=(15,10),
              title="family size ")

# let's check the servival count for the new feature 'familySize'

titanicDF.groupby(['familySize' , 'Survived']).Survived.count().unstack()


#Lets check 'name' column to create a new feature 'title'

namLis=[]
for nam, age,sex in zip(titanicDF['Name'],titanicDF['ageState'],titanicDF['Sex']):
    #print(nam, age , sex )
    nam = nam.split(",")[1].split(".")[0]
    #namLis.append(nam)
    titleLis=['Master','Miss','Mr','Mrs',' Capt']
    if nam not in titleLis:
        if age == 'Adult':
            if sex == 'male':
                namLis.append('Mr')
            else:
                namLis.append('Mrs')
        else:
            if sex == 'male':
                namLis.append('Master')
            else:
                namLis.append('Miss')
    else:
        namLis.append(nam.strip(' '))  # strip() trim the char from both side of a string

titanicDF['title'] = namLis

#print(titanicDF['title'])
titanicDF.groupby(['title']).title.unique()
titanicDF.groupby(['title']).title.count()


# Let's check how many mother survived in the disaster
# let's consider  'Parch' , 'sex' , 'ageState' and create new column 'motherOrNot' 

import numpy as np

titanicDF['motherOrNot'] = np.where( ( (titanicDF.Sex=='female') & (titanicDF.ageState=='Adult') & (titanicDF.Parch > 0) ) , 1 , 0)

titanicDF.groupby(['motherOrNot']).motherOrNot.count() # 71 mother was there

# 'Survived' count for the new feature 'motherOrNot'

titanicDF.groupby(['motherOrNot','Survived']).Survived.count().unstack()


# Let's check the 'Cabin' column/feature

titanicDF.groupby(['Cabin']).Cabin.unique()

# we can see there is different alphanumeric cabin numbner including 
# one 'T' cabin , and nan   so let's convert the  'T'  cabin to nan

import numpy as np

titanicDF.loc[titanicDF.Cabin == 'T' , 'Cabin'] = np.NaN

# Fill the nan value of cabin column to 'K' and keep only first char from
# alphanumeric value of cabin.

def getCabinChar(cabinVal):
    return np.where(pd.notnull(cabinVal) ,  str(cabinVal)[0].upper() , 'K'  )
    
titanicDF['cabinAlpha'] = titanicDF['Cabin'].map(lambda x : getCabinChar(x))

titanicDF.groupby(['cabinAlpha']).cabinAlpha.count()

# Let's check the servival of passenger per cabin

titanicDF.groupby(['Survived','cabinAlpha']).Survived.count().unstack()


# Now check the fare column 
# get the max and min fare of passanger

titanicDF['Fare'].max()
titanicDF['Fare'].min()

#Let's visualize the Fare distribution

ax = titanicDF['Fare'].plot(kind='hist', figsize=(15,10),
              title="passanger fare ")

# As passangers are given fare from 0.0 to 512 
# now categorise the passangers into different fare groups using "qcut()" .

titanicDF['FareCategorise'] = pd.qcut(titanicDF.Fare , 4 , labels = ['veryLow' , 'low' , 'high', 'veryHigh'] )

# converting numerical data to categorical data is known as discretization.

# now visualize the fare category in bar/histogram.

ax = titanicDF['FareCategorise'].value_counts().plot(kind='bar' , figsize=(15,10) ,
                                 title = "Fare catagories")

ax.set_xlabel("category of fare ")
ax.set_ylabel("frequency of fare ")

# lets check the dataframe info to check the datatype of all the features/columns

titanicDF.info()

"""  The output of info() is as below
Data columns (total 17 columns):
Pclass            891 non-null int64
Name              891 non-null object
Sex               891 non-null object
Age               891 non-null float64
SibSp             891 non-null int64
Parch             891 non-null int64
Ticket            891 non-null object
Fare              891 non-null float64
Cabin             205 non-null object
Embarked          891 non-null object
Survived          891 non-null int64
ageState          891 non-null object
familySize        891 non-null int64
title             891 non-null object
motherOrNot       891 non-null int32
cabinAlpha        891 non-null object
FareCategorise    891 non-null category
dtypes: category(1), float64(2), int32(1), int64(5), object(8)
"""

# Step-6 For the machine leaning we need to convert categorical feature 
    # to numerical feature so model can understand the data. 
    # For the conversion we can use Binary encoding, Label encoding and 
    # One-hot encoding.

#If there are two categories in a feature/column then we can 
# use binary encoding.    
    #EX- we can use binary encoding for " Sex , ageState " 

#If there is more than two category in a feature/column then we can 
# use label encoding. It is good to use label encoding when label has 
# an ordering sequence ascending or descending 'like high medium and low '
   #Ex - we can use label encoding for "Embarked , cabinAlpha , FareCategorise " 

#If there is two or more than two category and those category has not 
# order in a feature/column then we can use one-hot encoding 
# ( using pd.get_dummies() ).

    #Ex -We can use one-hot encoding for "title, Embarked , cabinAlpha , FareCategorise ,...."  

 
        ######### Binary encoding

titanicDF['encodeSex'] = np.where(titanicDF.Sex=='male' , 1 , 0) # if male it will give 1 else 0

        ######## Label encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

titanicDF['lableEncodeEmbark'] = le.fit_transform(titanicDF.Embarked)

        ######## One-hot encoding
        
titanicDF = pd.get_dummies(titanicDF,columns=['cabinAlpha', 'title' , 
                             'FareCategorise', 'ageState'])

# though 'ageState' column has two catagory we can use binary encoding , but 
# using adventage of get_dummies()  we can pass the column having two catagories.

    
# Step-7 Drop the columns which is not required in future. 
# Re-order the required columns and keep dependent column at the end.
# Save the processed data into a file 

#Drop the columns which is not required in future. 

titanicDF.drop(['Name', 'SibSp' , 'Parch' ,'Cabin' , 'Ticket'
                        , 'Sex' , 'Embarked' ] , axis = 1 , inplace = True)

titanicDF.info()
titanicDF.shape

# Re-order the required columns and keep dependent column at the end 

allColumns = [ col for col in titanicDF.columns if col != 'Survived' ]
allColumns = allColumns + ['Survived']
print(allColumns)

titanicDF = titanicDF[allColumns]
titanicDF.info() # Now we can see there is int and float data in the dataframe

# Save the processed data into a file  
# E:/datascienceProject/DatascienceNml/MLProjects/titanicDataAnalysis/titanicData/titanicOutputDataForML/

thePath = 'E:/datascienceProject/DatascienceNml/MLProjects/titanicDataAnalysis/titanicData/titanicOutputDataForML/'
theFile = 'titanicProcessed.csv'

def saveFile(Path,File):
    file=os.path.join(Path, File)
    titanicDF.to_csv(file, sep=',')
    return

saveFile(thePath,theFile )

# Step-8 Building and Evaluating the predictive model.
'''
# Always create a base line model first. The baseline model 
# doesn't use any machine learning at all. Baseline model always 
# gives the output of the majority class. 
# Then create the ML-model using algorithm and compare the accuracy
# the accuracy should be more than baseline model. 

# To check the acuracy we have to use  "accuracy, confusion metrics 
# and classification  report "
'''
# Building the model, by spliting processed data into training and test

titanicModelDF = readFileToDF(thePath,theFile)
indepentVar = titanicModelDF.iloc[:,:-1]
dependentVar = titanicModelDF.iloc[:,-1]

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(indepentVar, dependentVar, test_size = 0.2, random_state = 0)

print(Xtrain.shape ,ytrain.shape)
print(Xtest.shape , ytest.shape)

# average servived passanger
print("mean servival in train dataset" ,np.mean(ytrain))
print("mean servival in test dataset" ,np.mean(ytest))

# Baseline model

from sklearn.dummy import DummyClassifier
baseLine = DummyClassifier(strategy="most_frequent",random_state=0)
baseLine.fit(Xtrain,ytrain)

yBaselinePredict = baseLine.predict(Xtest)

#score/accuracy of baseline model

print("baseline model accuracy is  " , baseLine.score(Xtest,ytest))

# confusion metrics of baseline model



# Lets use feature normalization and optimization to the model for prediction.
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# feature normalization 

scaler = MinMaxScaler()
XtrainedScaled = scaler.fit_transform(Xtrain)

XtrainedScaled[:,0].min(),XtrainedScaled[:,0].max() 

# Normalise the test data

XtestScaled = scaler.transform(Xtest)

# feature standardization 

scaler = StandardScaler()
XtrainedScaled = scaler.fit_transform(Xtrain)

#standardize the test data
XtestScaled =  scaler.transform(Xtest)

# create the model after standardization 

from sklearn.linear_model import LogisticRegression
logisReg = LogisticRegression(random_state=0)

from sklearn.model_selection import GridSearchCV
alpha = [1.0,10.0,50.0,100.0,1000.0]
penalties= ['l1','l2']
gridParamters = {'C':alpha , 'penalty': penalties}
# cv=k will perform k-fold cross validation
optimLogisModel = GridSearchCV(logisReg , param_grid=gridParamters , cv=3) 
optimLogisModel.fit(Xtrain, ytrain)

# best_params_ give model paramerters for which we will get optimize model
optimLogisModel.best_params_  

yPredict = optimLogisModel.predict(Xtest)

# Accuracy of optimized logistic model is

from sklearn.metrics import accuracy_score
accuracy_score(ytest , yPredict)

# confusion metrics

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusionMatrixLreg = confusion_matrix(ytest , yPredict )
print(confusionMatrixLreg)


# Lets consider the Naivebayes Algorithm to perform prediction.

from sklearn.naive_bayes import GaussianNB
naivebayes = GaussianNB()
naivebayes.fit(Xtrain, ytrain)
yNbPredict = naivebayes.predict(Xtest)
     
# Accuracy of the model 

from sklearn.metrics import accuracy_score
accuracy_score(ytest, yNbPredict)

# confusion metrics

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report

confusionMatrixNb = confusion_matrix(ytest , yNbPredict )
print(confusionMatrixNb)

# classification report
print(classification_report(ytest , yNbPredict)) 



 