### a script used to split the orginal data into test and training sub-chunks
seed      = 2018
test_size = 0.2
#############################################################################

import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
print("Reading the data")
data = pd.read_csv("test.csv").fillna(0)

print("Drop key features")
X = data.drop(["flowStartMilliseconds","flowDurationMilliseconds_x","sourceIPAddress","destinationIPAddress","sourceTransportPort_x","destinationTransportPort_x","mode(destinationIPAddress)","mode(sourceTransportPort)","mode(destinationTransportPort)","mode(protocolIdentifier)","flowDurationMilliseconds_y","sourceTransportPort_y","destinationTransportPort_y","__NTAFlowID","__NTAPorts", "Label", "Attack"], axis=1)

y = data["Label"]

# split
print("Start spliting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# concatenate teh data and the labels
testing = pd.concat([X_test, y_test], axis=1)
training = pd.concat([X_train, y_train], axis=1)

# export
print("Exporting")
testing.to_csv('CAIA_testing.csv', index=False)
training.to_csv('CAIA_training.csv', index=False)
