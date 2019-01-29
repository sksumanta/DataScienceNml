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

![Visualization of Fare](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Fare.png)


###### # Visualize the count of passanger per class


![passanger class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/pclass1.PNG)


###### # Visualize average age of male and female passanger per class



![average age of passanger per class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/avgAge.png)


###### # visualization of fare with respect to passanger class


![fare wrt passanger class](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Farewrtpclass.png)

###### # Delete or imputation of null data

###### # Treatment of outliers 

![fare log transformation](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/Farelogtrnsform.png)

###### # categorise the passengers into different fare groups </p>

![fare category](https://github.com/sksumanta/DatascienceNml/blob/master/AllProjectImages/titanic/fareCatagory.PNG)

###### # convert categorical feature to numerical feature

###### # Drop the columns which is not required in future and Save the dataframe in an output file

###### # For prediction normalize and optimize the features 


###### # Train the model and prdeict using the trained model

<table>
<tbody>
<tr><td>confusion matrix</td></tr>
<tr><td>97</td><td> 13</td></tr>
<tr><td>16</td><td> 53</td></tr>
</tbody>
</table>
