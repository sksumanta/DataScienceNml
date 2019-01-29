### Titanic data analysis
	The Titanic Data is a classification problem to predict new passenger is survived or died.
##### Features in the file 
    PassengerID : Serial no assigned to each passenger
    Survived : Passenger survived or not at titanic disaster (1 - survived, 0 - not)
    Pclass : Passenger class (1-- 1st class , 2-- 2nd class and 3-- 3rd class)
    Name : Name of the Passenger
    Sex : Gender of the Passenger
    Age : Age of the Passenger
    SibSp : No of sibling and spouses boarded for the trip
    Parch : No of parents and children boarded for the trip
    Ticket : Ticket no for the passenger 
    Fare : Amount paid by the passenger
    Cabin : Cabin of the passenger
    Embarked : point of embankment/boarding point ( C  , Q , S)

<!DOCTYPE html>
<html>
<body>

<h5> Sample data</h5>

<table style="width:100%">
  <tr>
    <th>PassengerId</th>
	<th>Survived</th> 
    <th>Pclass</th> 
    <th>Name</th>
	<th>Sex</th>
    <th>Age</th> 
    <th>Parch</th>
	<th>Ticket</th>
    <th>Fare</th> 
    <th>Cabin</th>
	<th>Embarked</th>
  </tr>
  <tr>
	<td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22</td><td>1</td><td>0</td><td>A/5</td> <td>21171</td><td>7.25</td><td></td><td>S</td>
  </tr>
  <tr>	
	<td>2</td><td>1</td><td>1</td><td>Cumings, Mrs. John Bradley (Florence Briggs Thayer)</td><td>female</td><td>38</td><td>1</td><td>0</td><td>PC 17599</td><td>71.2833</td><td>C85</td><td>C</td>
  </tr>
  <tr>
	<td>3</td><td>1</td><td>3</td><td>Heikkinen, Miss. Laina</td><td>female</td><td>26</td><td>0</td><td>0</td><td>STON/O2. 3101282</td><td>7.925</td><td></td><td>S</td>
  </tr>
  <tr>
	<td>4</td><td>1</td><td>1</td><td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td><td>female</td><td>35</td><td>1</td><td>0</td><td>113803</td><td>53.1</td><td>C123</td><td>S</td>
  </tr>
  <tr>	
	<td>5</td><td>0</td><td>3</td><td>Allen, Mr. William Henry</td><td>male</td><td>35</td><td>0</td><td>0</td><td>373450</td><td>8.05</td><td></td><td>S</td>
  </tr>
  <tr>
  	<td>6 </td><td>	0 </td><td>	3  </td><td>	Moran, Mr. James  </td><td>	male  </td><td>	 </td> <td>		0	 </td><td> 0  </td><td>	330877  </td><td>	8.4583	 </td><td>	 </td><td> Q  </td>
   </tr>
</table>

</body>
</html>

##### Read file 
import pandas as pd

import os

def readFileToDF(Path,File):

    file=os.path.join(Path, File)
    
    fileDF = pd.read_csv(file,delimiter=",")
    
    return fileDF
    
titanicDF= readFileToDF(thePath,theFile)

titanicDF = titanicDF.loc[:,['PassengerId','Pclass','Name','Sex','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Survived']]

###### # As PassengerId is the serial no so in below 'PassengerId' used as row index

titanicDF=titanicDF.set_index(titanicDF.columns[0])
print(titanicDF.shape)

###### # The shape of data frame "titanicDF" is (891, 11)

###### Visualize different columns data distribution.
print(titanicDF.info())  
###### # Each columns data-type and count of not-null values
<p>&lt;class 'pandas.core.frame.DataFrame'&gt;<br />Int64Index:&nbsp; &nbsp; 891 entries,&nbsp; &nbsp;1 to 891<br />Data columns&nbsp; (total 11 columns):<br />Pclass&nbsp; &nbsp;891&nbsp; non-null&nbsp; int64<br />Name&nbsp; &nbsp;891&nbsp; non-null&nbsp; object<br />Sex&nbsp; &nbsp; &nbsp; 891&nbsp; non-null&nbsp; object<br />Age&nbsp; &nbsp; &nbsp; 714&nbsp; non-null&nbsp; float64<br />SibSp&nbsp; &nbsp;891&nbsp; non-null&nbsp; int64<br />Parch&nbsp; &nbsp;891&nbsp; non-null&nbsp; int64<br />Ticket&nbsp; 891&nbsp; non-null&nbsp; object<br />Fare&nbsp; &nbsp; 891&nbsp; non-null&nbsp; float64<br />Cabin&nbsp; 206&nbsp; non-null&nbsp; object<br />Embarked&nbsp; 889 non-null&nbsp; object<br />Survived&nbsp; &nbsp; 891 non-null&nbsp; int64<br />dtypes: float64(2), int64(4), object(5)<br />memory usage:&nbsp; 66.1+ KB<br />None</p>

###### # describe statistics for all columns

titanicDF.describe( include = 'all' )

###### # statistics for all columns.
<table>
<tbody>
<tr>
<td>&nbsp;</td>
<td>Pclass</td>
<td>Name</td>
<td>Sex</td>
<td>Age</td>
<td>SibSp</td>
<td>Parch</td>
<td>Ticket</td>
<td>Fare</td>
<td>Cabin</td>
<td>Embarked</td>
<td>Survived</td>
</tr>
<tr>
<td>count</td>
<td>891.0</td>
<td>891</td>
<td>891</td>
<td>714.0</td>
<td>891.0</td>
<td>891.0</td>
<td>891</td>
<td>891.0</td>
<td>204</td>
<td>889</td>
<td>891.0</td>
</tr>
<tr>
<td>unique</td>
<td>nan</td>
<td>891</td>
<td>2</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>681</td>
<td>nan</td>
<td>147</td>
<td>3</td>
<td>nan</td>
</tr>
<tr>
<td>top</td>
<td>nan</td>
<td>Lesurer, Mr. Gustave J</td>
<td>male</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>347082 nan</td>
<td>B96 B98</td>
<td>S</td>
<td>nan</td>
</tr>
<tr>
<td>freq</td>
<td>nan</td>
<td>1</td>
<td>577</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>7</td>
<td>nan</td>
<td>4</td>
<td>644</td>
<td>nan</td>
</tr>
<tr>
<td>mean</td>
<td>2.308641975308642</td>
<td>nan</td>
<td>nan</td>
<td>29.69911764705882</td>
<td>0.5230078563411896</td>
<td>0.38159371492704824</td>
<td>nan</td>
<td>32.2042079685746</td>
<td>nan</td>
<td>nan</td>
<td>0.3838383838383838</td>
</tr>
<tr>
<td>std</td>
<td>0.8360712409770513</td>
<td>nan</td>
<td>nan</td>
<td>14.526497332334044</td>
<td>1.1027434322934275</td>
<td>0.8060572211299559</td>
<td>nan</td>
<td>49.693428597180905</td>
<td>nan</td>
<td>nan</td>
<td>0.4865924542648585</td>
</tr>
<tr>
<td>min</td>
<td>1.0</td>
<td>nan</td>
<td>nan</td>
<td>0.42</td>
<td>0.0</td>
<td>0.0</td>
<td>nan</td>
<td>0.0</td>
<td>nan</td>
<td>nan</td>
<td>0.0</td>
</tr>
<tr>
<td>25%</td>
<td>2.0</td>
<td>nan</td>
<td>nan</td>
<td>20.125</td>
<td>0.0</td>
<td>0.0</td>
<td>nan</td>
<td>7.9104</td>
<td>nan</td>
<td>nan</td>
<td>0.0</td>
</tr>
<tr>
<td>50%</td>
<td>3.0</td>
<td>nan</td>
<td>nan</td>
<td>28.0</td>
<td>0.0</td>
<td>0.0</td>
<td>nan</td>
<td>14.4542</td>
<td>nan</td>
<td>nan</td>
<td>0.0</td>
</tr>
<tr>
<td>75%</td>
<td>3.0</td>
<td>nan</td>
<td>nan</td>
<td>38.0</td>
<td>1.0</td>
<td>nan</td>
<td>0.0</td>
<td>31.0</td>
<td>nan</td>
<td>nan</td>
<td>1.0</td>
</tr>
<tr>
<td>max</td>
<td>3.0</td>
<td>nan</td>
<td>nan</td>
<td>80.0</td>
<td>8.0</td>
<td>6.0</td>
<td>nan</td>
<td>512.3292</td>
<td>nan</td>
<td>nan</td>
<td>1.0</td>
</tr>
</tbody>
</table>

###### # visulaization of fare column distribution
ax = titanicDF['Fare'].plot(kind='hist',
                          figsize=(15,10),
                           bins = 10 )

ax.set_xlabel("passanger fare") 

ax.set_ylabel("No of passanger")

![Visualization of Fare](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Fare.png)


###### # Visualize the count of passanger per class

ax = titanicDF['Pclass'].value_counts().plot(kind='bar' , 
                                    figsize=(15,10), 
                                    title = "count of passanger class" )

ax.set_xlabel("pclass")

ax.set_ylabel("count")


![passanger class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/pclass1.PNG)


###### # Visualize average age of male and female passanger per class

Psex = titanicDF.groupby(['Sex','Pclass']).Age.mean().unstack()

import seaborn as sns

ax = sns.lineplot( data=Psex)


![average age of passanger per class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/avgAge.png)


###### # visualization of fare with respect to passanger class

ax = titanicDF.plot.scatter(x='Pclass' , y='Fare' ,
                            title = "Fare wrt passanger class")

ax.set_xlabel("Passanger class")

ax.set_ylabel("Fare")

![fare wrt passanger class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Farewrtpclass.png)


###### # Delete or imputation of null data

<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  # Fill the null value for Embarked column</p>

nullRowOfembarked  = titanicDF[titanicDF.Embarked.isnull()]

print(nullRowOfembarked)  # if there is any null value  isnull() will return true 

titanicDF.groupby(['Pclass','Embarked']).Fare.mean().unstack()
<table>
<tbody>
<tr>
<td>Embarked</td>
<td>C</td>
<td>Q</td>
<td>S</td>
</tr>
<tr>
<td>Pclass</td>
</tr>
<tr>
<td>1</td>
<td>104.718529 </td>
<td>90.000000</td>
<td>70.364862</td>
</tr>
<tr>
<td>2</td>
<td>25.358335</td>
<td>12.350000</td>
<td>20.327439</td>
</tr>
<tr>
<td>3</td>
<td>11.214083</td>
<td>11.183393</td>
<td>14.644083</td>
</tr>
</tbody>
</table>

titanicDF['Embarked'].fillna('S' , inplace = True)  		 # fill the passenger embark point as 'S'

<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # fill null value for Age column</p>

nullInAge = titanicDF[titanicDF.Age.isnull()]  

notnullData = titanicDF.dropna()

xTrainData = notnullData.loc[:,['Pclass','Survived','SibSp','Parch','Fare']]

yTrainData = notnullData.loc[:,'Age']

xTestData = nullInAge.loc[:,['Pclass','Survived','SibSp','Parch','Fare']]

from sklearn.linear_model import LinearRegression

linReg  = LinearRegression()

linReg.fit( xTrainData, yTrainData)  

predictAge = pd.DataFrame( linReg.predict(xTestData) , columns=['Age'])

titanicRowIndex = titanicDF[titanicDF.Age.isnull()].index.values.astype(int)

for rowIndex , predAge in zip(titanicRowIndex , predictAge['Age'] ):

	titanicDF.iloc[rowIndex-1 , 3 ] = abs(predAge)

###### # Treatment of outliers 

<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Treatment of Fare column outlier</p>

import numpy as np	

fareOutlier = titanicDF[titanicDF['Fare'] == titanicDF['Fare'].max()]

transformFare = np.log(titanicDF['Fare']+1)

ax = transformFare.plot(kind = 'hist' , figsize =(15,10) 
                        , title="transform fare using log transformation" , bins = 10)

ax.set_xlabel('no of passanger ')

ax.set_ylabel('fare of passanger')


![fare log transformation](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Farelogtrnsform.png)


###### # Feature engineering or create new feature
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Creating ageState feature using Age column </p>

titanicDF['ageState'] = np.where(titanicDF['Age']<18 , 'Child' , 'Adult')

titanicDF['ageState'].value_counts()

titanicDF.groupby(['ageState','Survived']).ageState.count().unstack()

<table>
<tbody>
<tr>
<td>Survived</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>ageState</td>
</tr>
<tr>
<td>Adult</td>
<td>497</td>
<td>281</td>
</tr>
<tr>
<td>Child</td>
<td>52</td>
<td>61</td>
</tr>
</tbody>
</table>

<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; # Creating familySize feature using 'SibSp' and 'Parch' column </p>

titanicDF['familySize'] = titanicDF['SibSp'] + titanicDF['Parch'] + 1 

titanicDF['familySize'].max() 

titanicDF.groupby(['familySize' , 'Survived']).Survived.count().unstack()

<table>
<tbody>
<tr>
<td>Survived</td>
<td>0</td>
<td>1</td>
</tr>
<tr>
<td>familySize</td>
</tr>
<tr>
<td>1</td>
<td>374.0</td>
<td>163.0</td>
</tr>
<tr>
<td>2</td>
<td>72.0</td>
<td>89.0</td>
</tr>
<tr>
<td>3</td>
<td>43.0</td>
<td>59.0</td>
</tr>
<tr>
<td>4</td>
<td>8.0</td>
<td>21.0</td>
</tr>
<tr>
<td>5</td>
<td>12.0</td>
<td>3.0</td>
</tr>
<tr>
<td>6</td>
<td>19.0</td>
<td>3.0</td>
</tr>
<tr>
<td>7</td>
<td>8.0</td>
<td>4.0</td>
</tr>
<tr>
<td>8</td>
<td>6.0</td>
<td>NaN</td>
</tr>
<tr>
<td>11</td>
<td>7.0</td>
<td>NaN</td>
</tr>
</tbody>
</table>












