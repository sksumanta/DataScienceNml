# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 20:27:26 2019

@author: Sumanta
"""

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

salesPriceDf = pd.read_csv("E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/sample_submission.csv")
housePriceDf = pd.read_csv("E:/datascienceNml/DataScienceInPy/HousePricesLinearRegression/Data/HouseData.csv")

HousePriceDf = pd.concat([housePriceDf,salesPriceDf['SalePrice']],axis=1)

HousePriceDf.shape      # shape of the dataframe

HousePriceDf.info()

# As more than 70 data is null in the below columns so we droped these columns
HousePriceDf = HousePriceDf.drop(['Alley', 'FireplaceQu' , 'PoolQC','Fence','MiscFeature' ] , axis=1)

HousePriceDf.shape      # shape of the dataframe

HousePriceDf.info()

describes = HousePriceDf.describe(include='all')

HousePriceDf['SalePrice'].describe() 

# Distribution of sales price

plt.hist(HousePriceDf['SalePrice'])

plt.boxplot(HousePriceDf['SalePrice'])

'''
Below columns are having continuous value.

'MSSubClass', 'LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 
'BsmtFinSF2' , 'BsmtUnfSF' , 'TotalBsmtSF' , '1stFlrSF' , '2ndFlrSF' ,
'GrLivArea' , 'GarageArea' ,'WoodDeckSF' , 'OpenPorchSF' 

so lets check the distrution of these columns.
'''

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
        ax = HousePriceDf[columns].plot(kind = 'hist' , figsize=(15,10),
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
        #print(columns , "\n\n" , HousePriceDf[columns])
        plt.scatter(HousePriceDf[columns],HousePriceDf['SalePrice'])
        plt.xlabel(columns, fontsize=9)
    plt.show()

plotScatter(columnList)

plotScatter(columnList2)


# checking the distribution of 'TotRmsAbvGrd' and group 'TotRmsAbvGrd' and 'saleprice' 

plt.hist(HousePriceDf['TotRmsAbvGrd'])

HousePriceDf['TotRmsAbvGrd'].value_counts()

ax = HousePriceDf['TotRmsAbvGrd'].value_counts().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Count")
plt.show()

HousePriceDf[['SalePrice', 'TotRmsAbvGrd']].groupby(['TotRmsAbvGrd']).mean().unstack()

ax = HousePriceDf[['SalePrice', 'TotRmsAbvGrd']].groupby(['TotRmsAbvGrd']).mean().plot(kind="bar")
ax.set(xlabel="Total Rooms Above Grade", ylabel="Mean")
plt.show()


# bivariate relation between each pair using "pairplot" and  "hitmap"

dfColumns = ['SalePrice','TotRmsAbvGrd','TotalBsmtSF','GrLivArea','GarageArea','FullBath','YearBuilt','YearRemodAdd']
#check is there any null value for the above columns in the dataframe

isNullDf = pd.DataFrame(HousePriceDf[dfColumns].isnull())
for checkNullCount in isNullDf.columns:
    valCount = isNullDf[checkNullCount].value_counts()  # TotalBsmtSF and GarageArea having null value
    print(checkNullCount , "\n" , valCount ,"\n\n")

'''     
# As we can not use null value column in pairplot so column list is
    
dfColumns = ['SalePrice','TotRmsAbvGrd','GarageArea','FullBath','YearBuilt','YearRemodAdd']


# it is not working we will check later

sns.pairplot(HousePriceDf[dfColumns],size = 2 ,kind ='scatter')
plt.show()
'''

sns.heatmap(HousePriceDf[dfColumns].corr(),annot=True,cmap='RdYlGn')
plt.show()



#Data Preprocessing 

# Handle outliers for SalePrice 

# Lets consider GrLivArea-: ground living area square feet to analyse saleprice







































