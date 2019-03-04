
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:27:26 2019

@author: Sumanta
"""

from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot as plt
import seaborn as sns
import datetime
import pandas as pd
import numpy as np
import time

train = pd.read_csv('E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/train.csv')

            # Arrange the columns
            
columList= [col for col in train.columns if col != 'SalePrice']
columList.append('SalePrice')  
train = train[columList]

train.shape      # shape of the dataframe

train.info()

train.describe(include='all')

test= pd.read_csv('E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/test.csv')

test.shape      # shape of the dataframe

test.info()

test.describe(include='all')

houseDf = pd.concat( (train.loc[:,'MSSubClass':'SaleCondition'],
                             test.loc[:,'MSSubClass':'SaleCondition']) )

houseDf.shape      # shape of the dataframe

houseDf.info()

houseDf.describe(include='all')

 # Statistical distribution of salePrice  

train['SalePrice'].describe() 

# Visualize the distribution of sales price
plt.figure()
plt.hist(train['SalePrice'])

plt.figure()
plt.boxplot(train['SalePrice'])

#lets check the distrution for Below continuous value columns.

columnList = ['MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
'BsmtFinSF2' , 'BsmtUnfSF','BedroomAbvGr','KitchenAbvGr' ]
columnList2 = ['1stFlrSF', 'TotalBsmtSF' , '2ndFlrSF' ,
'GrLivArea' , 'GarageArea' ,'WoodDeckSF' , 'OpenPorchSF','YearBuilt' ,'YearRemodAdd' ]

# distrution of these columns using histogram

def plotHist(col):
    ax = plt.subplots(3,3,figsize=(24,20))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(3,3,count)
        ax = train[columns].plot(kind = 'hist' , figsize=(15,10),
                           bins = 10 )
        ax.set_xlabel(columns)
    plt.show()

plotHist(columnList)

plotHist(columnList2)

# Identify relationship between two continuous/quantitative variables using Scatter plot

def plotScatter(col):    # visualize distribution of above column wrt salePrice
    ax = plt.subplots(3,3,figsize=(24,20))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(3,3,count)
        #print(columns , "\n\n" , train[columns])
        plt.scatter(train[columns],train['SalePrice'])
        plt.xlabel(columns, fontsize=9)
    plt.show()

plotScatter(columnList)

plotScatter(columnList2)

# seaborn's kdeplot, plots univariate or bivariate density estimates.
# FacetGrid drawing multiple instances of the same plot on different subsets of dataset.

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.FacetGrid(train[columns], hue="OverallQual", size=10 ).map(sns.kdeplot, "YearBuilt").add_legend()
plt.show()

# checking the distribution of 'TotRmsAbvGrd' 

plt.figure()
plt.hist(train['TotRmsAbvGrd'])

train['TotRmsAbvGrd'].value_counts()

plt.figure()
ax = train['TotRmsAbvGrd'].value_counts().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Count")
plt.show()

# 'saleprice' distribution wrt the TotRmsAbvGrd   

train.groupby(['SalePrice', 'TotRmsAbvGrd']).TotRmsAbvGrd.mean().unstack()

ax = train[['SalePrice', 'TotRmsAbvGrd']].groupby(['TotRmsAbvGrd']).mean().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Sale Price")
plt.show()

# 'saleprice' distribution wrt the OverallQual   

train.groupby(['SalePrice', 'OverallQual']).OverallQual.mean().unstack()

ax = train[['SalePrice', 'OverallQual']].groupby(['OverallQual']).mean().plot(kind="bar")
ax.set(xlabel="Over-all Quality", ylabel="Sale Price")
plt.show()

#check if, is there  any null value for the below columns
dfColumns = ['TotRmsAbvGrd','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd','OverallQual']

def countNull(cols):
    res=[]
    for c in cols:
        if pd.isnull(houseDf[c]).any() == True:
# If null value presnet then count the no of null value in that column
            res.append(c + " column has " + str(pd.isnull(houseDf[c]).sum() )+ " null values")
        else:
            res.append("no null values in " + c)
    return res
       
countNull(dfColumns)

#Lets go for the heatmap for the columns in dfColumns list 
plt.figure()
ax = sns.heatmap(pd.concat([train[dfColumns],train['SalePrice']] ,
                                   axis =1 ).corr(),annot=True,cmap='RdYlGn')
ax.xaxis.set_ticks_position('top')
plt.show()

#lets check the distrution for Below catagoriacal value columns.

colms = [
'MSZoning','Exterior1st','Exterior2nd','MasVnrType','BsmtQual',
'BsmtCond','Utilities','BsmtFinType1','BsmtFinType2','KitchenQual',
'Functional','BsmtExposure','SaleType','GarageFinish','GarageQual',
'GarageCond', 'GarageType','GarageCars','BsmtFullBath','BsmtHalfBath',
'Alley', 'FireplaceQu' , 'PoolQC','Fence','MiscFeature']

#houseCatagoricalDf = train[colms]

colSet = [colms[start::3] for start in range(3)]      # to get 3 set of graph for clear visualization

def plotBar(col):
    #plt.subplots_adjust(hspace=0.4)
    ax = plt.subplots(3,3,figsize=(30,30))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(3,3,count)
        ax = train[columns].value_counts().plot(kind='bar', figsize=(25,20))
        ax.set_xlabel(columns)
    plt.show()

#train['GarageFinish'].value_counts()
for ind in range(3):
    col=colSet[ind]
    plotBar(col)

                            #Data Preprocessing 

# Handle outliers for SalePrice 

# Lets consider GrLivArea-: ground living area square feet to analyse saleprice

plt.figure()
plt.scatter(train['GrLivArea'], train['SalePrice']) # visualize outliers
plt.xlabel("Ground LivArea")

train.sort_values(by = 'GrLivArea' , ascending=False).loc[:,
                                 ['LotArea' , 'GrLivArea','SalePrice']].head()

train.loc[[1298,523] ,['LotArea' , 'GrLivArea','SalePrice']]

# delete the outliers 

train = train.drop([1298,523], axis=0)

plt.figure()
plt.scatter(train['GrLivArea'], train['SalePrice']) # visualize outliers
plt.xlabel("Ground LivArea")

# To deal with the outliers let's transform the 'SalePrice' using log-transformation
#if sale price is zero we can handel it by adding 1

train['SalePrice'] = np.log1p(train['SalePrice'])  # log1p = log(data +1) 

#print(train['SalePrice'])

plt.figure()
plt.scatter(train['GrLivArea'],train['SalePrice'])
plt.xlabel("Ground LivArea")

                        #Fill null value
                        
# check the null value in dataframe

houseDf.info()

# nullData= houseDf[ pd.isnull(houseDf).any() ]

# Find all the numeric columns or continuous data columns
numericCol = houseDf.dtypes[houseDf.dtypes != "object"].index

#Lets check null values for below columns and fill them using statistical methods.  

countNull(numericCol)     

# find the mean and median of the numeric columns

def findMeanMedian(cols):
    for c in cols:
        print("mean of",c  , "is ", houseDf[c].mean()) 
        print("median of",c  , "is ",houseDf[c].median())  

findMeanMedian(numericCol)  
      
# lets fill  "median value"  in the null columns

for c in numericCol:
    if pd.isnull(houseDf[c]).any() == True:
        print(c)
        houseDf[c].fillna(houseDf[c].median(), inplace=True)


houseDf.info()


#compute skewness of each column ( default axis=1) to measure, asymmetry of the probability distribution 
'''
For an unimodal distribution (for each column) if we plot histogrm then 
#  positive skew indicates the tail is on right side
# negative skew commonly indicates ,tail is on left side
'''
skewedData = houseDf[numericCol].apply(lambda x : pd.DataFrame.skew(x))

skewedCols = skewedData.index

# Lets do one-hot encoding for "MSSubClass" catagorical column.

houseDf = pd.get_dummies(houseDf,columns=['MSSubClass'])
houseDf.shape


'''
Fill the null values of " 'BsmtQual'catagorical columns
 using  " ['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl'] " columns
'''

indColumns =  ['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl']

depColumn = ['BsmtQual']

sourceDf = pd.concat([houseDf[indColumns],houseDf[depColumn]] , axis=1)

sourceDf.shape
   
countNull(depColumn)   # find the number of null value present in each column


sourceDf['Foundation'].value_counts()
sourceDf['RoofMatl'].value_counts()

            # Preform label encoding for Foundation and RoofMatl
            
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()   # creating label encoder instance


sourceDf['Foundation'] = le.fit_transform(sourceDf.Foundation)
sourceDf['RoofMatl'] = le.fit_transform(sourceDf.RoofMatl)


nullInBsmtQual = sourceDf[sourceDf.BsmtQual.isnull()] 
notnullData = sourceDf.dropna()

notnullData['BsmtQual'].value_counts()


# Preform label encoding for BsmtQual

notnullData['BsmtQual'] = le.fit_transform(notnullData.BsmtQual)
        
rowIndexBsmtQual =  sourceDf[sourceDf.BsmtQual.isnull()==False].index.values.astype(int)

#fill the encoded value into the source data frame for BsmtQual column        
for rowIndex , notnullBsmtQual in zip(rowIndexBsmtQual , notnullData['BsmtQual'] ):
    sourceDf.loc[rowIndex, 'BsmtQual' ] = notnullBsmtQual
            
# heatmap 

plt.figure()
ax = sns.heatmap(notnullData.corr() ,annot=True,cmap='RdYlGn')
ax.xaxis.set_ticks_position('top')     #you will get a warning
plt.show()

xTrainData = notnullData.loc[:,['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl']]
yTrainData = notnullData.loc[:,['BsmtQual']]

xTestData = nullInBsmtQual.loc[:,['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl']]

rfBsmtQual = RandomForestClassifier()
rfBsmtQual.fit(xTrainData , yTrainData)

predictBsmtQual = pd.DataFrame(rfBsmtQual.predict(xTestData) , columns=['BsmtQual'])
        
rowNullIndexBsmtQual =  sourceDf[sourceDf.BsmtQual.isnull()].index.values.astype(int)
        
for rowIndex , predBsmtQual in zip(rowIndexBsmtQual , predictBsmtQual['BsmtQual'] ):
    sourceDf.loc[rowIndex, 'BsmtQual' ] = abs(predBsmtQual)
    
# now add notnull value of "BsmtQual" column to a new cloumn in original dataframe              
houseDf['BsmtQualRes'] = sourceDf['BsmtQual']
        

# Get all catagorical columns 
                
columnLists = houseDf.dtypes[houseDf.dtypes == "object"].index

# Fill the mode value if the column having NaN

for c in columnLists:
    # As these two columns are already filled so ignoring these two columns
    if 'BsmtQual' not in c and 'MSSubClass' not in c:
        print(c)
        #houseDf[c].fillna(houseDf[c].mode() , inplace = True )
        houseDf[c].fillna(houseDf[c].value_counts().index[0], inplace=True)
        #Once the null value is filled then label encode the column
        houseDf[c] = le.fit_transform( houseDf[c] )
        
houseDf.info()

# check any column having null value

houseDf.columns[houseDf.isna().any()].tolist() 

houseDf =  houseDf.drop(['BsmtQual'] , axis = 1)

# Save the Transformed data in a file

houseDf.to_csv("E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/processedData.csv" , sep=',')


# Find the correlation between the columns and find correlation > 70%

corrMatrix = houseDf.corr().abs()

# save the correlation matrix in a file

corrMatrix.to_csv("E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/corrMartix.csv")

# find correlation > 70%

hiCorrVar=np.where(corrMatrix>0.7)
#print(hiCorrVar)
hiCorrVar=[(corrMatrix.columns[x],corrMatrix.columns[y]) 
                        for x,y in zip(*hiCorrVar) if x!=y and x<y]

print(hiCorrVar)

corrDf = houseDf[pd.concat([ pd.DataFrame(hiCorrVar)[0],
                      pd.DataFrame(hiCorrVar)[1] ] ,axis = 0)]


#Lets go for the heatmap for the columns in dfColumns list 
plt.figure()
ax = sns.heatmap(corrDf.corr(),annot=True,cmap='RdYlGn')
ax.xaxis.set_ticks_position('top')
plt.xticks(rotation=15)
plt.show()

# From the above image we got the below columns for proceed.

columnList = ['YearBuilt','GrLivArea','GarageArea','BsmtFinType2','TotalBsmtSF',
   'Exterior1st','MSSubClass_20','MSSubClass_30','MSSubClass_40','MSSubClass_45',
   'MSSubClass_50','MSSubClass_60','MSSubClass_70','MSSubClass_75',
   'MSSubClass_80','MSSubClass_85','MSSubClass_90','MSSubClass_120',
   'MSSubClass_160','MSSubClass_180','MSSubClass_190']

# Split the data into train and test set 

xTrain= houseDf[ : train.shape[0]].loc[ :,columnList ]
xTest = houseDf[ train.shape[0] :].loc[:,columnList]
yTrain= train['SalePrice'] 

xTrain.shape
xTest.shape
yTrain.shape


#standardize the x variables/ independent variables 

from sklearn.preprocessing import StandardScaler
XStd = StandardScaler().fit_transform(xTrain.astype(np.float))

print(np.shape(XStd))

#covarient matrix 

covMatrix = np.cov(XStd.T)
print("covarient matrix \n" ,covMatrix)

#correlation matrix on Standardized data

corrMatrix = np.corrcoef(XStd.T)
print("correlation matrix of Standardized data \n", corrMatrix, end="\n\n")

# find the PCA on covarienc matrix

eigVals, eigVecs = np.linalg.eig(covMatrix)
        
print('Eigenvectors \n%s' %eigVecs, end="\n\n")
print('\nEigenvalues \n%s' %eigVals, end="\n\n")
      
# find the PCA on correlation matrix

eigVals, eigVecs = np.linalg.eig(corrMatrix)
        
print('Eigenvectors \n%s' %eigVecs, end="\n\n")
print('\nEigenvalues \n%s' %eigVals, end="\n\n")
      

# Use Decision Tree

from sklearn.tree import DecisionTreeRegressor

# Specify a number for random_state to ensure same results each run

dt = DecisionTreeRegressor(random_state=1)

dt.fit(xTrain, yTrain)

#score/accuracy 

dt.score(xTrain, yTrain)






