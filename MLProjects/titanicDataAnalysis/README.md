### Titanic data analysis
	The Titanic Data is a classification problem to predict passenger survival.
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
    
#####  Who were the passengers on the Titanic? (Ages,Gender,Class,..etc) 
<!DOCTYPE html>
<html>
<body>

<h5> Sample data  in Titanic data set </h5>

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
	<td>1</td><td>0</td><td>3</td><td>Braund, Mr. Owen Harris</td><td>male</td><td>22</td><td>1</td><td>0</td><td>A/5</td> <td>21171</td><td>7.25</td><td>S</td>
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

#####  Where did the passengers come from?

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
<td>104.718529</td>
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

###### # Visualize of passenger Fare

![Visualization of Fare](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/farewrtpclass.PNG)

###### # Visualize the count of passenger per class

![passenger class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/pclass1.PNG) 

###### # Visualize average age  of male and female passenger per class

![age range per class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/avgAge.png)

###### What deck/cabin were the passengers on and how does that relate to their class? Did the deck have an effect on the passengers survival rate? 

For this fill the nan value of cabin column to 'K' and keep only first upper case alphabetic char from alphanumeric value of cabin.

<table>
<tbody>
<tr>
<td>cabinAlpha</td>
<td>A</td>
<td>B</td>
<td>C</td>
<td>D</td>
<td>E</td>
<td>F</td>
<td>G</td>
<td>K</td>
</tr>
<tr>
<td>Survived</td>
</tr>
<tr>
<td>0</td>
<td>8</td>
<td>14</td>
<td>24</td>
<td>8</td>
<td>8</td>
<td>5</td>
<td>2</td>
<td>480</td>
</tr>
<tr>
<td>1</td>
<td>7</td>
<td>35</td>
<td>35</td>
<td>25</td>
<td>24</td>
<td>8</td>
<td>2</td>
<td>206</td>
</tr>
</tbody>
</table>

###### Did age state has any effect on survival rate? 

![agestate wrt servival rate](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/agestate.PNG)

###### Feature engineering of family size.

titanicDF['familySize'] = titanicDF['SibSp'] + titanicDF['Parch'] + 1 # 1 is added for person itself

titanicDF['familySize'].max()

###### # Visualize family size 

![familySize wrt servival rate](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/familySize.PNG)

###### Did family size has any effect on survival rate? 
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

###### # Visualize family size wrt servival rate

 ![familySize wrt servival rate](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/familySizewrtservive.PNG)
 
###### What factors effect passenger servival? 

Using PCA get the minimum number of features which we need to transform the training set and test set. 

<p>&nbsp; &nbsp; &nbsp;  pca = PCA(n_components = 15 ) </p>

Then create the model to predict the accuracy.

<p>&nbsp; &nbsp; &nbsp;  from sklearn.metrics import accuracy_score </p>

<p>&nbsp; &nbsp; &nbsp;  accuracy_score(ytest , yPredict) </p>
	<p>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; # accuracy score 0.8379 &nbsp;</p>
	
confusion matrix will give  true positive , true negative , false positive and false negative result

<p>&nbsp; &nbsp; &nbsp; from sklearn.metrics import confusion_matrix, accuracy_score,classification_report  </p>

<p>&nbsp; &nbsp; &nbsp;  confusionMatrixLreg = confusion_matrix(ytest , yPredict )  </p>

<table>
<tbody>
<tr>
<td>confusion Matrix</td>
</tr>
<tr>
<td>95</td>
<td>15</td>
</tr>
<tr>
<td>15</td>
<td>55</td>
</tr>
</tbody>
</table>


