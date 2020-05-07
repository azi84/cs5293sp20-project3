#===== librarys
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

#reading a file ------------------------------------

with open('yummly.json') as datafile:
    data = json.load(datafile)
df = pd.DataFrame(data)

#dataframe convert to string---------------------------------------

df['ingredients'] = [', '.join(map(str, i)) for i in df['ingredients']]

#Vectorizing ingrediant ---------------------------------
tf= TfidfVectorizer(ngram_range=(1,3), stop_words= 'english')
X = tf.fit_transform(df['ingredients'])
#X.shape

# Y for fo fitting-------------------------------
y = []
for i in data :
    y += [i['cuisine']]

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 100)
#--------------------------------------------    
logistic = LogisticRegression()
logistic.fit(X, y)


# Get inputs from users(interface)---------------------------------

inputs = [] # list 
print ("\nPlease tell waht ingredient you have, I suggest you a recipe.")

while True: # 

    data = input('\nEnter your ingredient : ') # read the data from user to variable data

    if data == '':
        break

    else:
        inputs.append(data) 

print("\n",inputs) 

# train user inputs----------------------------------------
user_ingredient = tf.transform(inputs)
predict = logistic.predict(user_ingredient)[0]
predict_prob =logistic.predict_proba(user_ingredient)[0]
class_predic = logistic.classes_


print ("\nThe cuisine related to your ingredients is: %s !!! (%f)" %(predict,predict_prob[0]*100))
print("----------------------------------------------------------")

#Get number of recepie user need(interface)-------------------------------
Num = input("\nPlease enter the number of cuisine you want: ")
print("")
#Find n and tarain to find a neghibours of the user input--------------------
knn=KNeighborsClassifier(n_neighbors=Num)
knn.fit(X,y)
id_prob,ids = knn.kneighbors(user_ingredient,int(Num))
#prediction of neghibours-------------------------------
print ("\nClosest %d Cuisines are :" %(len (ids[0])))

for i in range(len(ids[0])):
    print("%d ," %(df.id[ids[0][i]]))
    print(df.cuisine[ids[0][i]])
    print({df.ingredients[ids[0][i]]})
    print("")
