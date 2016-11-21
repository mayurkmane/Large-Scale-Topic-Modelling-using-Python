# Libraries used
import time
import glob
import pandas as pd 
import os
import numpy as np
from sklearn.linear_model import SGDClassifier
import nltk
from nltk.corpus import stopwords # Import the stop word list
import re
from bs4 import BeautifulSoup # Use bs4 to remove html tags or markup
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import accuracy_score
import pickle

print('All libarris are imported successfully!')

# Import data
print('\nData Importing.............')

os.chdir("C:\\Users\\Mayur\\Downloads\\TopicData")
path =r'C:\\Users\\Mayur\\Downloads\\TopicData'
filenames = glob.glob(path + "/*.csv")
dfs = []
for filename in filenames:
    dfs.append(pd.read_csv(filename, header=0, delimiter=",", encoding = "ISO-8859-1"))

Train = pd.concat(dfs, ignore_index=True)

print('\nShuffle dataset to make robust...')
Train.iloc[np.random.permutation(len(Train))]
Train.reset_index(drop=True)

print('\nTraining Data imported with dimensions:', Train.shape)

#Data Preprocessing

print("\nRemoving NA values...")
Train = Train.replace(np.nan,' ', regex=True)

print("\nDividing data into train and test set...")
msk = np.random.rand(len(Train)) < 0.9

train = Train[msk]
test = Train[~msk]

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)

# Data Cleaning

def tweet_to_words( raw_tweet ):
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_tweet).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. Make list of stopwords
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_tweet = [w for w in words if not w in stops]   
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_tweet ))

# Get the number of tweets based on the dataframe column size
num_tweets = train["Tweet"].size

# Initialize an empty list to hold the clean tweets
clean_train_tweets = []

print("\nCleaning and parsing the training set tweets...")
startTime = time.time()

for i in range( 0, num_tweets ):
    # If the index is evenly divisible by 100000, print a message
    if( (i+1)%1000000 == 0 ):
        print("Tweet %d of %d\n" % ( i+1, num_tweets ))                                                                   
    clean_train_tweets.append( tweet_to_words( train["Tweet"][i] ))
    
print('\nCleaning of all tweets have been done in ', time.time()-startTime, 'seconds.')

# Object Serialization
##pickle_in = open('clean_train_tweets.pickle','wb')
##pickle.dump(clean_train_tweets, pickle_in)
##pickle_in.close()
##pickle_in = open('clean_train_tweets.pickle','rb')
##clean_train_tweets = pickle.load(pickle_in)
print('\nYou can use pickling to get cleaned trained tweets (clean_train_tweets.pickle).')

# Initialize the "HashingVectorizer" object, which is scikit-learn's
# bag of words tool with max_features  
vectorizer = HashingVectorizer(analyzer = "word", \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             ngram_range = (1, 2),\
                             n_features = 2**18)

train_data_features = vectorizer.fit_transform(clean_train_tweets)

print('\nWe have created  document term matrix with dimensions ', train_data_features.shape, ' and we have target variable with dimensions ', train['Tweet'].shape)

#Batch Processing
print("\nDoing batch processing...")

numtrainingpoints = len(train)
def iter_minibatches(chunksize):
    # Provide chunks one by one
    chunkstartmarker = 0
    while chunkstartmarker < numtrainingpoints:
        chunkrows = chunkstartmarker+chunksize
        try:
            X_chunk, Y_chunk = train_data_features[chunkstartmarker:chunkrows],train["Subtopic"][chunkstartmarker:chunkrows]
            yield X_chunk, Y_chunk
            chunkstartmarker += chunksize
        except IndexError:
            print("\nAll batches are processed!!!")
            break

print('\n==========================================================================================')

# Using generator to create small batches from sparse matrix

print("\nDividing data into small chunks with size 100K...")
batcherator = iter_minibatches(chunksize=100000)

clf = SGDClassifier(loss='modified_huber', random_state=1, n_iter=1, n_jobs = -1)

classes = np.unique(train['Subtopic'])
BatchNumber = 1
for X_chunk, Y_chunk in batcherator:
    Y_chunk = Y_chunk.reset_index()
    clf.partial_fit(X_chunk, Y_chunk['Subtopic'], classes = classes)
    print('\n', BatchNumber ,'Batch Processed')
    BatchNumber += 1

# Processing Test data
# Create an empty list and append the clean reviews one by one
num_tweets_test = len(test["Tweet"])
clean_test_tweets = [] 

print("\nCleaning and parsing the test set tweets...")
for i in range(0,num_tweets_test):
    if( (i+1) % 100000 == 0 ):
        print("Tweet %d of %d\n" % (i+1, num_tweets_test))
    clean_tweet = tweet_to_words( test["Tweet"][i] )
    clean_test_tweets.append( clean_tweet )
print("\nTest data has been cleaned successfully! Hurray!")

test_data_features = vectorizer.transform(clean_test_tweets)

print("\nPredicting on test data and storing in file output...")
result = clf.predict(test_data_features)

# Copy the results to a pandas dataframe with an "id" column and
# a "sentiment" column
output = pd.DataFrame( data={"Tweet":test["Tweet"], "Subtopic":result} )

print('\nWe have received accuracy of ', accuracy_score(test['Subtopic'], output['Subtopic'])*100)

# Use pandas to write the comma-separated output file
# output.to_csv( "Predicted_topics.csv",sep=',', encoding='utf-8', index=False)

##print('\n Predicted output as follows:')
##output.head()

a = (output.groupby('Subtopic').size() / output.groupby('Subtopic').size().sum())*100
a = pd.DataFrame({'Subtopic':a.index, 'Percentage':a.values})
a.sort(['Percentage'], ascending=[False,True])
a = a[["Subtopic", "Percentage"]]

print('\nTopic distribution in percentages as follows:\n')
print(a.head(10))

print('Prediction of maximum likelihood topics is completed now!')
