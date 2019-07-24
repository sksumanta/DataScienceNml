# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 19:20:29 2019

@author: Sumanta
"""


from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

train = pd.read_csv("E:/datascienceNml/DataScienceInPy/BigmartSales/data/Train.csv")
test = pd.read_csv("E:/datascienceNml/DataScienceInPy/BigmartSales/data/Test.csv")

train.info()
test.info()

# Create a datafram taking all columns from given train and test dataset excluding dependent column from train dataset

yTrain = train.iloc[:,-1]

train['file_lable'] = "train"
test['file_lable'] = "test"

columList= [col for col in train.columns if col != 'Item_Outlet_Sales']

bigMartDF =  pd.concat([train[columList] , test[columList]])

bigMartDF = bigMartDF.reset_index(drop=True)

bigMartDF.info()
bigMartDF.describe(include='all')

# collecting the non-numeric column and numeric column

nonNumColumns = bigMartDF.dtypes[bigMartDF.dtypes == 'object'].index
print(nonNumColumns)
numColumns = bigMartDF.dtypes[bigMartDF.dtypes != 'object'].index
print(numColumns)

for c in numColumns:
    if pd.isnull(bigMartDF[c]).any() == True:
        print(c , pd.isnull(bigMartDF[c]).sum())

for c in nonNumColumns:
    if pd.isnull(bigMartDF[c]).any() == True:
        print(c , pd.isnull(bigMartDF[c]).sum())

for c in nonNumColumns:
    print( c , "\n" , bigMartDF[c].value_counts(dropna = True) , end = "\n\n")
        
        
# Visualize the distribution of sales price
# continuous --- > hist , line , regression plot
# catagorical ---> scatter , hitmap , cluster map
    
ax=train['Item_Outlet_Sales'].plot(kind='hist')
ax.set(xlabel="Item_Outlet_Sales", ylabel="frequency")
plt.show()

plt.figure()
plt.boxplot(train['Item_Outlet_Sales'])

# explore count-plot / bar-plot for all univariant non-numeric columns 

def plotCountplt(col):   
    ax = plt.subplots(3,2,figsize=(24,20))
    count = 0
    for columns in col:
        if columns not in ('Item_Identifier','file_lable'):
            count +=1
            #print(count)
            plt.subplot(3,2,count)
            sns.countplot(bigMartDF[columns].dropna())
            plt.xlabel(columns, fontsize=9)
    plt.show()

plotCountplt(nonNumColumns)

# explore histogram / PDF / CDF (that is distplot  , kdeplot) for all univariant numeric columns 

def plotHistogram(col):   
    ax = plt.subplots(2,2,figsize=(24,20))
    count = 0
    for columns in col:
        count +=1
        plt.subplot(2,2,count)
        #sns.kdeplot(bigMartDF[columns].dropna())
        sns.distplot(bigMartDF[columns].dropna() , kde=True)
        plt.xlabel(columns, fontsize=9)
    plt.show()

plotHistogram(numColumns)

# convert catagorical value to numerical value         
from sklearn.preprocessing import LabelEncoder 
le = LabelEncoder()   # creating label encoder instance

for col in nonNumColumns:
    if col != 'Outlet_Size':
        bigMartDF[col] = le.fit_transform(bigMartDF[col])
     
notnullData = bigMartDF['Outlet_Size'].dropna().index
bigMartDF.loc[notnullData,'Outlet_Size'] = le.fit_transform(bigMartDF.loc[notnullData,'Outlet_Size'])

# fill null value for Item_Weight
bigMartDF['Item_Weight'].mean()
bigMartDF['Item_Weight'].median()

bigMartDF['Item_Weight'].fillna(bigMartDF['Item_Weight'].median() , inplace = True) 

# fill null value for  Outlet_Size

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 100, random_state = 10)

outletSizeDF = pd.DataFrame(bigMartDF.loc[:,['Outlet_Location_Type','Outlet_Type','Outlet_Size']].dropna())
xtrain = outletSizeDF.loc[:,['Outlet_Location_Type','Outlet_Type']]
ytrain = outletSizeDF['Outlet_Size']

rf.fit(xtrain, ytrain)

nullIndex=bigMartDF.loc[pd.isna(bigMartDF["Outlet_Size"]), :].index
    
xtest = bigMartDF.loc[nullIndex,['Outlet_Location_Type','Outlet_Type']]

ypredict = rf.predict(xtest)

bigMartDF.loc[nullIndex,'Outlet_Size'] = ypredict

# explore bivariant using scatterplot for train dataset
yTrain.shape
def plotScatter(col):   
    ax = plt.subplots(3,4,figsize=(24,20))
    count = 0
    for columns in col:
        if columns != 'file_lable':
            count +=1
            train.dropna()
            plt.subplot(3,4,count)
            plt.scatter(bigMartDF.loc[ :8522 , columns ], yTrain)
            plt.xlabel(columns, fontsize=9)
    plt.show()

plotScatter(columList)

# handel the outliers using p-value / Z-score ( all column shoud be numeric in the dataframe) 

bigMartStdDF = pd.concat([bigMartDF.iloc[:8523,:-1] , yTrain] , axis = 1)
from scipy import stats
import numpy as np
outLierRow_n_ColumnArray = np.abs(stats.zscore(bigMartStdDF.astype('float') ))
print(outLierRow_n_ColumnArray)  

# In Z-score result first list is row and second list is column 

# to get three outliers
threshold = 4
outlires = np.where(outLierRow_n_ColumnArray > threshold )
# from the result we can see the row no 81 and column 2 ,   row no 83 and column 3  and so on
 
rowNumber = outlires[0]
colNumber = outlires[1]

bigMartDF.drop(bigMartDF.iloc[rowNumber].index , inplace = True)
bigMartDF = bigMartDF.reset_index(drop=True)
bigMartDF.shape

yTrain.drop(yTrain.iloc[rowNumber].index , inplace = True)
yTrain = yTrain.reset_index(drop=True)
yTrain.shape

# Explore data after removing outliers

#  Explore dependent data after removing outliers
ax=yTrain.plot(kind='box')
ax.set(xlabel="Item_Outlet_Sales", ylabel="frequency")
plt.show()   

#  Explore independent and dependent data after removing outliers using scatter plot
def plotScatter(col):   
    ax = plt.subplots(3,4,figsize=(24,20))
    count = 0
    for columns in col:
        if columns != 'file_lable':
            count +=1
            train.dropna()
            plt.subplot(3,4,count)
            plt.scatter(bigMartDF.loc[ :8342 , columns ], yTrain)
            plt.xlabel(columns, fontsize=9)
    plt.show()

plotScatter(columList)

# Selecting columns / Feature selection  based on correlation (PCA) or using p-value or using RFE (Recursive Feature Elimination) 

# correlation (multi-coliniarity) and co-variance / hitmap

corrDF = pd.concat([bigMartDF.iloc[ :8343,:-1 ], yTrain] , axis =1 )
corrMatrix = corrDF.corr().abs()

# find correlation > 65%
hiCorrVar=np.where(corrDF>0.65) 

# hitmap to visualize the correlation

sns.heatmap(corrMatrix,annot=True,cmap='RdYlGn')


''''
# using RFE (Recursive Feature Elimination) 

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

model = LinearRegression()

rfe  =  RFE(model, 7) # select the 7 important Feature 

fit = rfe.fit(bigMartDF.iloc[ :8343,:-1 ], yTrain)

print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)  # Selected Features will show  'True'
print("Feature Ranking: %s" % fit.ranking_)  # Selected Features will show  1 

'''

# Using the P-value


'''
import statsmodels.api as sm 
X = sm.add_constant(np.array(bigMartDF.iloc[ :8343,:-1 ].values, dtype = float)) # add_constant adds a constant column of ones, mandatory for sm.OLS model 
Y = yTrain.values 
model = sm.OLS(Y,X).fit()
model.pvalues
'''

'''
import statsmodels.api as sm 
cols = corrDF.iloc[:,:-1].columns
SL = 0.05 # Significance Level
colsList = list(cols)

for i in range(len(colsList)):
    X = sm.add_constant(np.array(bigMartDF.iloc[ :8343,:-1 ].values, dtype = float)) # add_constant adds a constant column of ones, mandatory for sm.OLS model 
    Y = corrDF.iloc[:,-1].values 
    model = sm.OLS(Y,X).fit()
    pVal = pd.Series(model.pvalues[1:] , index = cols)
    pmax = max(pVal)
    feature_with_p_max = pVal.idxmax()
    print(feature_with_p_max)
    if feature_with_p_max in colsList:
        if(pmax > SL):
            colsList.remove(feature_with_p_max)
        else:
            break

selected_features_BE = cols
print(selected_features_BE)

'''

# standardize the independent data

from sklearn.preprocessing import StandardScaler
XStd = StandardScaler().fit_transform(bigMartDF.iloc[:,:-1].astype(np.float))
XStd = pd.DataFrame(XStd)

# Normaize the dependent data
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()

Ydepe = min_max_scaler.fit_transform(pd.DataFrame(yTrain).values.astype(np.float))

YStd = pd.DataFrame(Ydepe) 


# Explore data after standardize
ax=YStd.plot(kind='box')
ax.set(xlabel="Item_Outlet_Sales", ylabel="frequency")
plt.show()   

#  check multi-coliniarity using correlation and co-variance / hitmap for standardise data

covMatrix = np.cov(XStd.iloc[ :8343,: ].T)

#eigenvectors only define the directions of the new axis

#eigen-decomposition on the covariance matrix:

eig_vals, eig_vecs = np.linalg.eig(covMatrix)

print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)


#correlation matrix on Standardized data

CorrStdMatrix = np.corrcoef(XStd.iloc[ :8343,: ].T)
#print("correlation matrix of Standardized data \n", irisCorrStdMatrix)

#eigen-decomposition on the Standardized data of correlation matrix

eigVals , eigVects = np.linalg.eig(CorrStdMatrix)

print('Eigenvectors \n%s' %eigVects)
print('\nEigenvalues \n%s' %eigVals)

'''   
To decide which eigenvector(s) can dropped without losing too much data
we need to check the eigen values by sorting highest to lowest order 
and choose the top k eigenvectors.
'''

# Make a list of (eigenvalue, eigenvector) tuples

eigPairs = [(np.abs(eigVals[i]), eigVects[:,i]) for i in range(len(eigVals))]

#print(eigPairs[0][0]," is the eigen value  and \n\n ", eigPairs[0][1] , " is the eigen vector")
#print(eigPairs[1][0]," is the eigen value  and \n\n ", eigPairs[1][1] , " is the eigen vector")
#print(eigPairs)

# Sort the (eigenvalue, eigenvector) tuples from high to low 

eigPairs.sort(key=lambda x: x[0], reverse=True)

sortedEigenVal=[]
for i in eigPairs:
    sortedEigenVal.append(i[0])
    print(i[0],"\n================\n",i[1])

print(sortedEigenVal)    
# Explained Variance / how many component or attribute we will keep for predicton    

tot = sum(eigVals)

# cumulative sum of the variable 

valsInPercent =  [ (ev/tot)*100 for ev in sorted(eigVals, reverse=True)]
#print(valsInPercent)
cumuVals = np.cumsum(valsInPercent)
#print(cumuVals)

# PCA cumulative plot to check the variance

with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(16, 12))

    plt.bar(range(11), valsInPercent, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(11), cumuVals, where='mid',
             label='cumulative explained variance')
    plt.plot(range(11), cumuVals, 'C1o')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
#plt.savefig('PREDI2.png', format='png', dpi=1200)
plt.show()

# Let's choosing "top 7" eigenvectors as it has 
# highest eigenvalues to create Projection Matrix for PCA

projMatrix = np.hstack((eigPairs[0][1].reshape(11,1),
                      eigPairs[1][1].reshape(11,1),
                      eigPairs[2][1].reshape(11,1),
                      eigPairs[3][1].reshape(11,1),
                      eigPairs[4][1].reshape(11,1),
                      eigPairs[5][1].reshape(11,1),
                      eigPairs[6][1].reshape(11,1)
                          ))
print('Projection Matrix:\n', projMatrix)  # 11X2 matrix created

# newMatrix  = standardize_X_matrix * ProjectionMartix 

newMatrix = XStd.iloc[ :8343,: ].dot(projMatrix) 





