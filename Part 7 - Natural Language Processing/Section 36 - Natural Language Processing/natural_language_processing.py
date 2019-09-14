# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import the  data set 
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
#quoting=3 will ignore the  dounle  quotes in  file 

#Cleaning the dataset
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[] #list 
for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i]) #remove all  exept a-z and A-Z letter 
    review=review.lower();#all  letetr  in lower case
    review=review.split() # split the  review into different word
    ps=PorterStemmer() #only keep the root of the word eg loved becomes love 
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review=' '.join(review) #revert  back into string
    corpus.append(review)

#Creating the  Bag of words model i.e. taking out distinct words in bag
#number of time word appears 
#bag of words model is also called sparse mat 

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500) #1500  most frequent  words helpful in large  dataset
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)
  

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)  
    
    

