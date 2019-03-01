# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:27:26 2019

@author: Sumanta
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder 
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

hPriceDf = pd.read_csv("E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/train.csv")

hPriceDf.shape      # shape of the dataframe

hPriceDf.info()

# As more than 70 persent data is null in the below columns so we droped these columns

hPriceDf = hPriceDf.drop(['Alley', 'FireplaceQu' , 'PoolQC','Fence','MiscFeature' ] , axis=1)

hPriceDf.shape      # shape of the dataframe

hPriceDf.info()

describes = hPriceDf.describe(include='all')

hPriceDf['SalePrice'].describe() 

# Distribution of sales price
plt.figure()
plt.hist(hPriceDf['SalePrice'])

plt.figure()
plt.boxplot(hPriceDf['SalePrice'])

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
        ax = hPriceDf[columns].plot(kind = 'hist' , figsize=(15,10),
                           bins = 10 )
        ax.set_xlabel(columns)
    plt.show()

plotHist(columnList)

plotHist(columnList2)

# Identify relationship between two continuous/quantitative variables using Scatter plot

def plotScatter(col):
    ax = plt.subplots(3,3,figsize=(24,20))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(3,3,count)
        #print(columns , "\n\n" , hPriceDf[columns])
        plt.scatter(hPriceDf[columns],hPriceDf['SalePrice'])
        plt.xlabel(columns, fontsize=9)
    plt.show()

plotScatter(columnList)

plotScatter(columnList2)

# seaborn's kdeplot, plots univariate or bivariate density estimates.
# FacetGrid drawing multiple instances of the same plot on different subsets of dataset.

columns = ['SalePrice','OverallQual','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
sns.FacetGrid(hPriceDf[columns], hue="OverallQual", size=10 ).map(sns.kdeplot, "YearBuilt").add_legend()
plt.show()

# checking the distribution of 'TotRmsAbvGrd' and group 'TotRmsAbvGrd' and 'saleprice' 

plt.figure()
plt.hist(hPriceDf['TotRmsAbvGrd'])

hPriceDf['TotRmsAbvGrd'].value_counts()

plt.figure()
ax = hPriceDf['TotRmsAbvGrd'].value_counts().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Count")
plt.show()

hPriceDf[['SalePrice', 'TotRmsAbvGrd']].groupby(['TotRmsAbvGrd']).mean().unstack()

plt.figure()
ax = hPriceDf[['SalePrice', 'TotRmsAbvGrd']].groupby(['TotRmsAbvGrd']).mean().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Mean")
plt.show()


# bivariate relation between each pair using "pairplot" and  "hitmap"

dfColumns = ['SalePrice','TotRmsAbvGrd','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
#check is there any null value for the above columns in the dataframe


def countNull(cols):
    res=[]
    for c in cols:
        if pd.isnull(hPriceDf[c]).any() == True:
# If null value presnet then count the no of null value in that column
            res.append(c + " column has " + str(pd.isnull(hPriceDf[c]).sum() )+ " null values")
        else:
            res.append("no null values in " + c)
    return res
       
countNull(dfColumns)

'''     
# As we can not use null value column in pairplot so column list is
    
dfColumns = ['SalePrice','TotRmsAbvGrd','GarageArea','FullBath','YearBuilt','YearRemodAdd']


# it is not working we will check later

sns.pairplot(hPriceDf[dfColumns],size = 2 ,kind ='scatter')
plt.show()
'''

plt.figure()
sns.heatmap(hPriceDf[dfColumns].corr(),annot=True,cmap='RdYlGn')
plt.show()

#lets check the distrution for Below catagoriacal value columns.

colms = [
'MSZoning','Exterior1st','Exterior2nd','MasVnrType','BsmtQual',
'BsmtCond','Utilities','BsmtFinType1','BsmtFinType2','KitchenQual',
'Functional','BsmtExposure','SaleType','GarageFinish','GarageQual',
'GarageCond', 'GarageType','GarageCars','BsmtFullBath','BsmtHalfBath']

houseCatagoricalDf = hPriceDf[colms]

colSet = [colms[start::3] for start in range(3)]      # to get 3 set of graph for clear visualization

def plotBar(col):
    #plt.subplots_adjust(hspace=0.4)
    ax = plt.subplots(3,3,figsize=(30,30))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(3,3,count)
        ax = hPriceDf[columns].value_counts().plot(kind='bar', figsize=(25,20))
        ax.set_xlabel(columns)
    plt.show()

#hPriceDf['GarageFinish'].value_counts()
for ind in range(3):
    col=colSet[ind]
    plotBar(col)

                            #Data Preprocessing 

# Handle outliers for SalePrice 

# Lets consider GrLivArea-: ground living area square feet to analyse saleprice

plt.figure()
plt.scatter(hPriceDf['GrLivArea'], hPriceDf['SalePrice']) # visualize outliers
plt.xlabel("Ground LivArea")

# To deal with the outliers let's transform the 'SalePrice' using log-transformation
#if sale price is zero we can handel it by adding 1

transformSalePrice = np.log1p(hPriceDf['SalePrice'])  # log1p = log(data +1) 

#print(transformSalePrice)

plt.figure()
plt.scatter(hPriceDf['GrLivArea'],transformSalePrice)
plt.xlabel("Ground LivArea")

                        #Fill null value
                        
# check the null value in dataframe

hPriceDf.info()

nullData= hPriceDf[pd.isnull(hPriceDf).any(axis=1)]

#Lets check null values for below columns and fill them using statistical methods.  

cols = ['BsmtFinSF1','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','GarageArea','MasVnrArea','LotFrontage']

countNull(cols)     
      
#plot histogram to check the distrubution of the null columns

plotHist(cols)

# find the mean and median of the null columns

def findMeanMedian(cols):
    for c in cols:
        print("mean of",c  , "is ", hPriceDf[c].mean()) 
        print("median of",c  , "is ",hPriceDf[c].median())  


findMeanMedian(cols)  
      
# lets fill  "median value"  in the null columns

for c in cols:
    hPriceDf[c].fillna(hPriceDf[c].median() , inplace = True)

hPriceDf.info()

from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()   # creating label encoder instance



'''
Fill the null values of " 'BsmtQual'catagorical columns
 using  " ['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl'] " columns
'''

indColumns =  ['TotRmsAbvGrd','TotalBsmtSF','Foundation','RoofMatl']

depColumn = ['BsmtQual']

sourceDf = pd.concat([hPriceDf[indColumns],hPriceDf[depColumn]] , axis=1)

sourceDf.shape
   
countNull(depColumn)   # find the number of null value present in each column


sourceDf['Foundation'].value_counts()
sourceDf['RoofMatl'].value_counts()

            # Preform label encoding for Foundation and RoofMatl
le = LabelEncoder()

sourceDf['Foundation'] = le.fit_transform(sourceDf.Foundation)
sourceDf['RoofMatl'] = le.fit_transform(sourceDf.RoofMatl)


nullInBsmtQual = sourceDf[sourceDf.BsmtQual.isnull()] 
notnullData = sourceDf.dropna()

notnullData['BsmtQual'].value_counts()


# Preform label encoding for BsmtQual

notnullData['BsmtQual'] = le.fit_transform(notnullData.BsmtQual)
        
rowIndexBsmtQual =  sourceDf[sourceDf.BsmtQual.isnull()==False].index.values.astype(int)
         
for rowIndex , notnullBsmtQual in zip(rowIndexBsmtQual , notnullData['BsmtQual'] ):
    sourceDf.ix[rowIndex, 'BsmtQual' ] = notnullBsmtQual
            
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
        
rowIndexBsmtQual =  sourceDf[sourceDf.BsmtQual.isnull()].index.values.astype(int)
        
for rowIndex , predBsmtQual in zip(rowIndexBsmtQual , predictBsmtQual['BsmtQual'] ):
    sourceDf.ix[rowIndex, 'BsmtQual' ] = abs(predBsmtQual)
    
# now add notnull value of "BsmtQual" column to a new cloumn in original dataframe              
hPriceDf['BsmtQualRes'] = sourceDf['BsmtQual']
        

                # Fill mode for the null values in catagorical data
for c in colms:
    if 'BsmtQual' not in c:
        #hPriceDf[c].fillna(hPriceDf[c].mode() , inplace = True )
        hPriceDf[c].fillna(hPriceDf[c].value_counts().index[0], inplace=True)

hPriceDf.info()

# fill null value of 'Electrical' column
hPriceDf['Electrical'].fillna(hPriceDf['Electrical'].value_counts().index[0], inplace=True)

# check any column having null value

hPriceDf.columns[hPriceDf.isna().any()].tolist()   # found 'GarageYrBlt' having null value

#Fill null value of 'GarageYrBlt' considering 'GarageType','GarageArea', 'YearRemodAdd','OverallQual'

hPriceDf['YearRemodAdd'] = pd.to_datetime(hPriceDf['YearRemodAdd'], format='%Y', utc=True)










