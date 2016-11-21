Python 3.4.4 (v3.4.4:737efcadf5a6, Dec 20 2015, 20:20:57) [MSC v.1600 64 bit (AMD64)] on win32
Type "copyright", "credits" or "license()" for more information.
>>> 
================== RESTART: C:\Python34\SGDandPartialFit.py ==================
All libarris are imported successfully!

Data Importing.............

Training Data imported with dimensions: (2176739, 3)

Removing NA values...

Dividing data into train and test set...

Cleaning and parsing the training set tweets...

Warning (from warnings module):
  File "C:\Python34\lib\site-packages\bs4\__init__.py", line 181
    markup_type=markup_type))
UserWarning: No parser was explicitly specified, so I'm using the best available HTML parser for this system ("html.parser"). This usually isn't a problem, but if you run this code on another system, or in a different virtual environment, it may use a different parser and behave differently.

The code that caused this warning is on line 1 of the file <string>. To get rid of this warning, change code that looks like this:

 BeautifulSoup([your markup])

to this:

 BeautifulSoup([your markup], "html.parser")

Tweet 100000 of 1741752

Tweet 200000 of 1741752

Tweet 300000 of 1741752

Tweet 400000 of 1741752

Tweet 500000 of 1741752

Tweet 600000 of 1741752

Tweet 700000 of 1741752

Tweet 800000 of 1741752

Tweet 900000 of 1741752

Tweet 1000000 of 1741752

Tweet 1100000 of 1741752

Tweet 1200000 of 1741752

Tweet 1300000 of 1741752

Tweet 1400000 of 1741752

Tweet 1500000 of 1741752

Tweet 1600000 of 1741752

Tweet 1700000 of 1741752


Cleaning of all tweets have been done in  2596.2021930217743 seconds.

You can use pickling to get cleaned trained tweets (clean_train_tweets.pickle).

We have created  document term matrix with dimensions  (1741752, 262144)  and we have target variable with dimensions  (1741752,)

Doing batch processing...

==========================================================================================

Dividing data into small chunks with size 100K...

 1 Batch Processed

 2 Batch Processed

 3 Batch Processed

 4 Batch Processed

 5 Batch Processed

 6 Batch Processed

 7 Batch Processed

 8 Batch Processed

 9 Batch Processed

 10 Batch Processed

 11 Batch Processed

 12 Batch Processed

 13 Batch Processed

 14 Batch Processed

 15 Batch Processed

 16 Batch Processed

 17 Batch Processed

All batches are processed!!!

Cleaning and parsing the test set tweets...
Tweet 100000 of 434987

Tweet 200000 of 434987

Tweet 300000 of 434987

Tweet 400000 of 434987


Test data has been cleaned successfully! Hurray!

Predicting on test data and storing in file output...

We have received accuracy of  81.5371493861

Topic distribution in percentages as follows:

Subtopic
Art and Entertainment                                      3.307225
Art and Entertainment/Books and Literature                 0.019771
Art and Entertainment/Books and Literature/Children's      0.000460
Art and Entertainment/Books and Literature/Fiction         0.164143
Art and Entertainment/Books and Literature/Non-Fiction     0.008276
Art and Entertainment/Comedy                               3.105840
Art and Entertainment/Comics and Animation                 0.546453
Art and Entertainment/Entertainment News                   0.538867
Art and Entertainment/Movies and TV                        0.770828
Art and Entertainment/Movies and TV/Comedy                 0.007357
Art and Entertainment/Movies and TV/Movies                 0.363689
Art and Entertainment/Movies and TV/TV                     3.738962
Art and Entertainment/Music                               68.593314
Art and Entertainment/Music News                           2.229032
Art and Entertainment/Performing Arts/Dance                0.004828
Art and Entertainment/Performing Arts/Theatre              0.000460
Art and Entertainment/Podcasts                             0.010115
Art and Entertainment/Radio                                0.586454
Art and Entertainment/Shows and Events                     0.093106
Art and Entertainment/Social Media                         2.470189
Art and Entertainment/Social Media/Instagram               0.121843
Art and Entertainment/Social Media/Snapchat                0.000460
Art and Entertainment/Social Media/Vine                    0.209432
Art and Entertainment/Social Media/YouTube                 5.330734
Art and Entertainment/Visual Art and Design/Visual Art     0.041610
Sports/Basketball                                          0.084600
Sports/Boxing                                              0.025288
Sports/Cricket                                             0.036783
Sports/Football                                            0.003218
Sports/Formula 1                                           0.070347
Sports/Golf                                                0.184833
Sports/Ice Hockey                                          0.035863
Sports/Mixed Martial Arts                                  0.073795
Sports/Motorcycling                                        0.002299
Sports/Rugby                                               0.044369
Sports/Running and Jogging                                 0.001379
Sports/Soccer                                              6.173288
Sports/Sports News                                         0.523924
Sports/Tennis                                              0.260008
Sports/Wrestling                                           0.216558
dtype: float64
>>> 
