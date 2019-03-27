### a script used to split the orginal data into test and training sub-chunks
seed      = 2018
test_size = 0.2
#############################################################################






import pandas as pd
from sklearn.model_selection import train_test_split

# read the data
print("Reading the data")
data = pd.read_csv("test.csv").fillna(0)

# convert the protocol field into dummy variables and concatenate it with the data
#print("Converting the Protocol feature into dummy")
#data = pd.concat([data, pd.get_dummies(data=data.protocolIdentifier, prefix='protocolIdentifier')], axis=1)

print("Drop key features")
X = data.drop(["flowStartSeconds", "sourceIPAddress", "destinationIPAddress",
               "sourceTransportPort", "destinationTransportPort", "Label", "Attack"], axis=1)
y = data["Label"]

# split
print("Start spliting")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

# concatenate teh data and the labels
testing = pd.concat([X_test, y_test], axis=1)
training = pd.concat([X_train, y_train], axis=1)

# export
print("Exporting")
testing.to_csv('Consensus_testing.csv', index=False)
training.to_csv('Consensus_training.csv', index=False)