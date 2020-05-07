#librarys------------------------------------------
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

#reading a file ------------------------------------

with open('yummly.json') as datafile: # open a file
    data = json.load(datafile)    # read a json file
df = pd.DataFrame(data)           # convert to datafarme

#dataframe convert to string---------------------------------------

df['ingredients'] = [', '.join(map(str, i)) for i in df['ingredients']]  # convert to string

#Vectorizing ingrediant ---------------------------------
tf= TfidfVectorizer(ngram_range=(1,3), stop_words= 'english') # using tfidf  featuer vectorizng
X = tf.fit_transform(df['ingredients'])  #fit and transform the string data
#X.shape

# Y for fo fitting-------------------------------
y = [] #list
for i in data :  # loop through jsson file
    y += [i['cuisine']] # read all cuisine data

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
#--------------------------------------------    
logistic = LogisticRegression()  # classify  and train by using the logisticRegression
logistic.fit(X, y)     # fit the xa nd y which are our ingrediant and cuisine


# Get inputs from users(interface)---------------------------------

inputs = [] # list 
print ("\nPlease tell waht ingredient you have, I suggest you a recipe.")

while True: # while loop

    lists = input('\nEnter your ingredient : ') # read the data from user to variable data

    if lists == '': # exist when ecieve double enter
        break

    else:
        inputs.append(lists) # append the the uster ingridients to inputs list 

print("\n",inputs) 

# train user inputs----------------------------------------
user_ingredient = tf.transform(inputs) # transform the user input
predict = logistic.predict(user_ingredient) #predict the user input
predict_prob =logistic.predict_proba(user_ingredient)[0] # get the predict probability fo the user input
class_predic = logistic.classes_ # check the class, that gives us all the cusine names


print ("\nThe cuisine related to your ingredients is: %s !!! (%f)" %(predict[0],predict_prob[0]*100)) # print the result 
#of predict and its probablity 

print("----------------------------------------------------------")

#Get number of recepie user need(interface)-------------------------------
Num = input("\nPlease enter the number of cuisine you want: ")  # ask the user to enter the number of the cuisine wants
print("")

#Find n and tarain to find a neghibours of the user input--------------------
knn=KNeighborsClassifier(n_neighbors=Num)  # train based on the number of the user input
knn.fit(X,y) # fitting
ids_prob,ids = knn.kneighbors(user_ingredient,int(Num)) # finding the neighbors and its probability

#prediction of neghibours-------------------------------
print ("\nClosest %d Cuisines are :" %(len (ids[0])))

for i in range(len(ids[0])): # loop through a neghbors
    print("%d(%f) ," %(df.id[ids[0][i]],ids_prob[0][i]))  # result of the predict cuisine id and it's probability
    print(df.cuisine[ids[0][i]])   # predicted cuisine
    print({df.ingredients[ids[0][i]]})  # predicted ingridiant
    print("")
  
