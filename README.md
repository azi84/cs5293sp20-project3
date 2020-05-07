# Project 3 The Analyzer:
The goal of the project is to create an application that take a list of ingredients from a user and attempts to predict the type of cuisine and similar meals. Consider a chef who has a list of ingredients and would like to change the current meal without changing the ingredients. The steps to develop the application should proceed as follows.

# project  Description:
In this project We are using yummly.json data set to help the Executive Chef in understanding the large menu set better by providing a cuisine predictor.
## Github:
First I made github repository with the name of cs5293sp20-project3. This one would be use at the end of project when are done we need to push every directory and file we made in our cloud instant. All directory and file was created in the cloud SSH can use $git clone + you github url of the project repository to get accsess to github and following code would be used after we done with project to have them in our github :

$git status (at first time it is red before adding the files/directories to github)
$git add filename
$git status (turn to green after adding)
$git commit -m "your comment"
$git push origin master]

### Directory and files:
For starting to make your directory such as ##project0/docs/test... with the command:

mkdir "name of directory"

Pipfile and Pipfile.lock :
For these two file we need the other kind of command for them like :

Pipfile ==> $pipenv --python python3

Pipfile.lock ==> $pipenv install requests

Main.py:
This file is contain the main function of our codes and after we wrote our python code in project0.py we need to call them from this file.

Setup.py:
This file is contain below codes which you can write into it by using Vim setup.py command :

from setuptools import setup, find packages setup( name='project0', version='1.0', author='You Name', authour_email='your ou email', packages=find_packages(exclude=('tests', 'docs')), setup_requires=['pytest-runner'], tests_require=['pytest'] )

Setup.cfg:
This file should have at least the following text inside which for writing into it we use Vim setup.cfg command: [aliases] test=pytest

[tool:pytest] norecursedirs = .*, CVS, _darcs, {arch}, *.egg, venv

## Packages:
Following the packages were used in this projects that for having them in your SSH you need to use the ##pipenve install filename.
>       json
>       pandas
>       from sklearn.linear_model import LogisticRegression
>       from sklearn.model_selection import train_test_split
>       from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
>       from sklearn.feature_extraction import DictVectorizer
>       from sklearn.neighbors import KNeighborsClassifier
>       
# Code Description:
#### Parse dataset and Conver the text to features:
In this project we were working with json files. Which I used json.load for loading and it and with open for opening the file. Then I assgined it to data farame by using of pandas liConver the text to featuresbarary.

At first step convert my "ingredients" columns to string to make it ready for vectrozing. I used **TFidVectorizer** to convert my text into feature vector. I used n-grams for processors and english stop word.
I checked some random columns and rows and decided not to do cleaning beacuse the 'id' and 'cuisine' column do not need and stop word and n-gram was enough for 'ingrediants'
Also I created a list of cuisine for making ready for next step.
####  Train or prepare classifiers:
In this step I used LogesticRegression to train the entire dataset, also I tried kneighborsclassifier and K-means clustrig as well but based on my finall result I decided to go with LogesticRegression beacuse I got so much netter result with that at end.

After deciding to use Regression model, for fit the model according the data, I used 'ingredient as my X and worked with both 'id' and 'Cuisine'. 

The problem I have with 'id' was I coudln't work with all the data, even if I used the train_test_split with the test_size = 0.5 I had a memory problem. The only way was worked in my computer was selecting at most 3000 data. I wrote all my code for that and get the result with that too. But I thought we might need to work with the whole dataset so I submited the one I trained with 'cuisine'.

(I will submit the one for 'id' in m github for my future aceess)

#### Get input:
After fiting with LogesticRegression,
I made a simple interface to asked user to enter the ingrediant he/she has. In this part user can enter how many ingrediant he/she want and can exit by entering double enter. 
#### Prediction:

After getting input from the user we need to find the best cuisin matched by the given ingredients. 

Therefor,after transfrom the user input I used the predict methods , to predict the class labes for provided ingrediant.
Then I used the predict_probability method to get the probability estimate for that.
Then return the predicted 'cuisine' with its probabilty.

#### Select closest N recipes:

In this part first I asked the user that how many cuisine he/she wants based on her ingrediants.

I used the number they Enter as the number of the neighbors for KNeighborsClassifier. Here I used this library for two reasons.
First I need to used the number that the user enter to find the best match cusine for that so KNeighborsClassifier was the best choice. 
Second, I need to find the neghbiours of that ingrediants and number were enter based on 'id'. 

In this part I used my first X, y for fitting the classifier and get a really good result. But I decided to use the user ingrediant input as my X and the predict one as y. The problem I has was, if the number of Cuisine the user wants more that the number of the ingredints he/she enter. 

After finding the neghibours of the cuisine by the help of the .neghibors method based on the user desire ingrediants I used for loop through that and called the 'id' and the'cuisine name that was match with the ingridenats.'

![](https://i.imgur.com/Q5ThztO.png)

## Whole project view:

In this project after reading data and using the feature vector I used two type of methods two train and fit my dataset. The fisrt one was LogesticRegression that helps me two predict the type of the cuisine based on the user input. The second one was KNeighborsClassifier that was useful to find the neghibors of the user inputs based on the number of the cuisine he/she wants. Then fianl result will suggest the user different cuisine based on her/his available ingridents.
## Check the project result:

There is two way to run my code.
First you can use the cloud ssh and run it by **python3 project3.py** and then answer to the question of the interface to see the final result.

Second way is you can use jupyter notebook and run it then answer the question.

### Collaboration:
1. https://towardsdatascience.com/multi-class-text-classification-model-comparison-and-selection-5eb066197568
1. https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
1. https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
1. https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
1. https://towardsdatascience.com/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8
1. https://stackoverflow.com/questions/52262800/machine-learning-algorithm-does-not-work-after-vectorizing-a-feature-that-is-of
1. https://stackoverflow.com/questions/7378091/taking-multiple-inputs-from-user-in-python
2. https://github.com/senior-sigan/ML_Competitions/blob/master/kaggle_whats-cooking/step_1.ipynb





