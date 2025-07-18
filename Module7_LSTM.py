#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report

df = pd.read_csv('train_val.csv')

df = df.drop(columns=['DAY_OF_MONTH', 'OP_UNIQUE_CARRIER', 'AIRLINE_FLIGHTS_MONTH', 'TAIL_NUM',
                      'ORIGIN_AIRPORT_ID', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME',
                      'DEST', 'DEST_CITY_NAME', 'DEP_TIME', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW',
                      'CANCELLED', 'DATE', 'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST',
                      'ARR_AIRPORT_HIST', 'DAY_HISTORICAL', 'DEP_BLOCK_HIST', 'PREV_AIRPORT_HIST', 'TMAX_C'])
df = df.dropna(subset=['DEP_DEL15'])

def clean_labels_encoder(list_of_labels, df):
    for label in list_of_labels:
        df[label] = pd.factorize(df[label])[0]
    return df

list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
df = clean_labels_encoder(list_of_labels, df)

df['TMAX'] = df['TMAX'].fillna(df['TMAX'].mean())

# Replace other NA values with 0
df = df.fillna(0)

# Splitting the dataset into training and testing sets
X = df.drop(columns=['DEP_DEL15'])
y = df['DEP_DEL15']

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

# Scaling the features
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshaping the data for LSTM
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=5, batch_size=64, validation_data=(X_test_scaled, y_test))

# Evaluate the model
score = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predictions
predictions_train = (model.predict(X_train_scaled) > 0.5).astype("int32")
predictions_test = (model.predict(X_test_scaled) > 0.5).astype("int32")

#Confusion Matrix and Classification Report for Test data
cm_test = confusion_matrix(y_test, predictions_test)
print("Confusion Matrix for LSTM for test:\n", cm_test)

# Calculate and Print Accuracy
accuracy_train = accuracy_score(y_train, predictions_train)
accuracy_test = accuracy_score(y_test, predictions_test)
print("Train Accuracy for LSTM:", accuracy_train)
print("Test Accuracy for LSTM:", accuracy_test)

# Classification Report
print("\nClassification Report for Training Data:")
print(classification_report(y_train, predictions_train))

print("\nClassification Report for Test Data:")
print(classification_report(y_test, predictions_test))







# Now repeat with a balanced dataset

# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import classification_report
import gc  # Python's garbage collector

df = pd.read_csv('train_val.csv')

# Data preprocessing
df = df.drop(columns=['DAY_OF_MONTH', 'OP_UNIQUE_CARRIER', 'AIRLINE_FLIGHTS_MONTH', 'TAIL_NUM',
                      'ORIGIN_AIRPORT_ID', 'OP_CARRIER_FL_NUM', 'ORIGIN', 'ORIGIN_CITY_NAME',
                      'DEST', 'DEST_CITY_NAME', 'DEP_TIME', 'DEP_DELAY_NEW', 'ARR_DELAY_NEW',
                      'CANCELLED', 'DATE', 'CARRIER_HISTORICAL', 'DEP_AIRPORT_HIST',
                      'ARR_AIRPORT_HIST', 'DAY_HISTORICAL', 'DEP_BLOCK_HIST', 'PREV_AIRPORT_HIST', 'TMAX_C'])
df = df.dropna(subset=['DEP_DEL15'])

# Encode categorical variables
def clean_labels_encoder(list_of_labels, df):
    for label in list_of_labels:
        df[label] = pd.factorize(df[label])[0]
    return df

list_of_labels = ['CARRIER_NAME', 'DEPARTING_AIRPORT', 'PREVIOUS_AIRPORT', 'DEP_TIME_BLK']
df = clean_labels_encoder(list_of_labels, df)

# Replace NA values in TMAX with the mean
df['TMAX'] = df['TMAX'].fillna(df['TMAX'].mean())

# Replace other NA values with 0
df = df.fillna(0)

# Separate majority and minority classes
df_majority = df[df.DEP_DEL15==0]
df_minority = df[df.DEP_DEL15==1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
replace=True, # sample with replacement
n_samples=len(df_majority), # to match majority class
random_state=123) # reproducible results

# Combine datasets and clear individual ones from memory
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
del df_majority, df_minority, df_minority_upsampled  # Free up memory
gc.collect()  # Explicit garbage collection

# Display new class counts
print(df_upsampled.DEP_DEL15.value_counts())

# Now df_upsampled is balanced

#Separate the features and target variable from the balanced dataframe
X = df_upsampled.drop('DEP_DEL15', axis=1)
y = df_upsampled.DEP_DEL15

#Splitting the balanced dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

#Count the occurrences of each class in the training set
train_counts = y_train.value_counts()
print("Training set counts:")
print(train_counts)

#Count the occurrences of each class in the test set
test_counts = y_test.value_counts()
print("\nTest set counts:")
print(test_counts)

# Feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.transform(X_test.astype(np.float32))

# Clear unscaled data from memory
del X_train, X_test
gc.collect()

# Reshaping the data for LSTM
X_train_scaled = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Define and compile the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(1, X_train_scaled.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=16))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_scaled, y_train, epochs=5, batch_size=64, validation_data=(X_test_scaled, y_test))

# Evaluate the model
score = model.evaluate(X_test_scaled, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predictions
predictions_train = (model.predict(X_train_scaled) > 0.5).astype("int32")
predictions_test = (model.predict(X_test_scaled) > 0.5).astype("int32")

#Confusion Matrix and Classification Report for Test data
cm_test = confusion_matrix(y_test, predictions_test)
print("Confusion Matrix for LSTM for test:\n", cm_test)

# Calculate and Print Accuracy
accuracy_train = accuracy_score(y_train, predictions_train)
accuracy_test = accuracy_score(y_test, predictions_test)
print("Train Accuracy for LSTM:", accuracy_train)
print("Test Accuracy for LSTM:", accuracy_test)

# Classification Report
print("\nClassification Report for Training Data:")
print(classification_report(y_train, predictions_train))

print("\nClassification Report for Test Data:")
print(classification_report(y_test, predictions_test))


# 
