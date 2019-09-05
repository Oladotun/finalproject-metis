#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 14:42:49 2019

@author: opasina
"""

import ast
import pandas as pd
import datetime

def diff_dates(d1, d2):
    date1 = datetime.datetime.strptime(d1.replace(",","").strip(), '%Y-%m-%d')
    date2 = datetime.datetime.strptime(d2.replace(",","").strip(), '%Y-%m-%d')
    return abs(date2-date1).days
def timedatelinesplit(newKick, columnName):
    newKick[columnName+"_datetime"] = newKick[columnName].apply(lambda x: datetime.datetime.fromtimestamp(float(x)).strftime('%Y-%m-%d , %H:%M:%S, %A'))
    new = newKick[columnName+"_datetime"].str.split(",", n = 2, expand = True) 
    newKick[columnName+"_date"] = new[0]
    newKick[columnName+"_time"] = new[1]
    timeDayArray =  []
    for hours in new[1]:
        hours = hours.strip().split(":")

        timeOfDay = ""
        if int(hours[0]) in [12,13,14,15]:
            timeOfDay = "Afternoon"
            timeDayArray.append(timeOfDay)
        elif hours[0] in [16,17,18,19,20]:
            timeOfDay = "Evening"
            timeDayArray.append(timeOfDay)
        elif hours[0] in [20,21,22,23,24]:
            timeOfDay = "Night"
            timeDayArray.append(timeOfDay)
        else:
            timeOfDay = "Morning"
            timeDayArray.append(timeOfDay)
        
    newKick[columnName+"_moment"] = timeDayArray
    newKick[columnName+"_day"] = new[2]
    
    newKick = newKick.drop(columns = [columnName, columnName+"_datetime"])
    
    return newKick
def getDfValues(df,names):
    return df[names[0]], df[[names[1]]]
def stateReturn(state):
    if state == 'successful':
        return 1
    else:
        return 0

kick = pd.read_csv("Kickstarter001.csv")
## Drop non needed columns
newKick = kick.drop(columns = ["is_backing","is_starred","static_usd_rate","urls",
                     "source_url","currency_trailing_code","id","slug", "current_currency",
                     "currency","profile","photo","permissions","created_at",
                     "usd_pledged","usd_type","converted_pledged_amount",
                     "currency_symbol","disable_communication","friends",
                     "fx_rate","is_starrable"
                     ])

## Get only for USA
newKick = newKick[newKick["country"] == "US"]
newKick = newKick.apply(lambda row: row[newKick['state'].isin(['successful','failed'])])

namesArray = []
slugArray = []
categoryNameSlugArray = []
for row in newKick["category"]:
    rowDict = ast.literal_eval(row)
    namesArray.append(rowDict["name"])
    
    slugArray.append(rowDict["slug"].split("/")[0])
    categoryNameSlugArray.append(rowDict["name"] + " "+rowDict["slug"].split("/")[0])
    

newKick["category_name"] = namesArray
newKick["category_slug"] = slugArray
newKick["category_name_slug"] = categoryNameSlugArray
newKick = newKick.drop(columns = ["category"])
creatorArray = []
creatorNameArray = []   
for row in newKick["creator"]:
    rowDict = str(row)
    rowArray = rowDict.split(":")
    rowOneArray = rowArray[1].split(",")
    creatorArray.append(rowOneArray[0])
    rowTwoArray = rowArray[2].split(",")
    creatorNameArray.append(rowTwoArray[0])
newKick["creator_id"] = creatorArray
newKick["creator_name"] = creatorNameArray

newKick = newKick.drop(columns = ["creator"])

locationArray = []
locationNameArray = []  
for row in newKick["location"]:
    if str(row) != 'nan': 
        rowDict = str(row)
        rowArray = rowDict.split(",")
        rowOneArray = rowArray[1].split(":")
        rowTwoArray = rowArray[9].split(":")
        
        locationArray.append(rowOneArray[1].replace('\"',"").strip() + " "+ rowTwoArray[1].replace('\"',"").strip())
    else:
        locationArray.append("help")

newKick["city_state"] = locationArray
newKick = newKick.drop(columns = ["location", "country"])    
newKick = timedatelinesplit(newKick, "deadline")
newKick = timedatelinesplit(newKick, "launched_at")
newKick["goal_pledged_diff"] = newKick["pledged"] - newKick["goal"]
newKick["duration_for_days"] = newKick[["deadline_date","launched_at_date"]].apply(lambda x: diff_dates(x[0],x[1]), axis = 1)
newKick["staff_pick"] = newKick["staff_pick"]


### EDA GroupBys
#category = newKick[newKick["state"]=="successful"].groupby("category_slug").count()
category = newKick[newKick["state"]=="successful"].groupby("category_slug").count()

cleanedUpKick = newKick.drop(columns=["backers_count","blurb","name","pledged","state_changed_at",
                      "deadline_date","deadline_time","launched_at_date", "creator_name" , "category_name","category_slug","goal_pledged_diff","spotlight"])
cols = ['goal', 'duration_for_days']
subset_df = newKick[cols]

#Standard Scaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
scaled_df = ss.fit_transform(subset_df)
scaled_df = pd.DataFrame(scaled_df, columns=cols)


# Category Slug
cleanedUpKick['staff_pick'] = cleanedUpKick['staff_pick'].astype(int)
cleanedUpKick['state'] = cleanedUpKick['state'].apply(lambda x:stateReturn(x))
cleanedWithKickTime = pd.get_dummies(cleanedUpKick, columns=["category_name_slug","city_state","launched_at_day","launched_at_moment"])
cleanedWithKickTime[["goal","duration_for_days"]] = scaled_df
cleanedWithKickTime.dropna(axis='rows',inplace=True)
X = cleanedWithKickTime.drop(columns=["state","deadline_moment","deadline_day","launched_at_time","creator_id"])
y = cleanedWithKickTime["state"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)

from sklearn import metrics
prediction_test = model.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))

## To get the weights of all the variables
weights = pd.Series(model.coef_[0],index=X.columns.values)
weightSorted = weights.sort_values(ascending = False)


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import (Dense, Embedding, Reshape, Activation, 
                          SimpleRNN, LSTM, Convolution1D, 
                          MaxPooling1D, Dropout, Bidirectional,SpatialDropout1D)
#from keras.layers import SpatialDropout1D
#from keras.callbacks import EarlyStopping

## Neural Network Logistic Regression

model = Sequential()
model.add(Dense(input_dim=1053, init = 'uniform' ,output_dim=527, activation = 'relu'))
model.add(Dense(output_dim=527, activation = 'relu'))
model.add(Dense(output_dim = 1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()
epochs = 10
batch_size = 64


model.fit(X_train, y_train, batch_size=256, epochs=epochs, 
              validation_data=(X_test, y_test))
hidden_features = model.predict(X_test)

## Neural Network Text Fields
MAX_SEQUENCE_LENGTH = 250
EMBEDDING_DIM = 100
MAX_NB_WORDS = 30000

tokenizer = Tokenizer(num_words = MAX_NB_WORDS, filters="""!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',""")
tokenizer.fit_on_texts(newKick['blurb'].values)
word_index = tokenizer.word_index
X_blurb = tokenizer.texts_to_sequences(newKick['blurb'])
X_blurb = pad_sequences(X_blurb, maxlen=MAX_SEQUENCE_LENGTH)
X_train_blurb, X_test_blurb, y_train_blurb, y_test_blurb = train_test_split(X_blurb,y, test_size = 0.20)


model = Sequential()
model.add(Embedding(input_dim=MAX_NB_WORDS, output_dim=40, 
                    embeddings_initializer='glorot_uniform', input_length=MAX_SEQUENCE_LENGTH))
model.add(SpatialDropout1D(0.2))

model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(18, activation='softmax'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

model.summary()
epochs = 10
batch_size = 64


model.fit(X_train_blurb, y_train_blurb, batch_size=256, epochs=epochs, 
              validation_data=(X_test, y_test))
hidden_features_blurb = model.predict(X_test_blurb)


