import pickle
# scikit-learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# load the serialized data from data.pickle
data_dict = pickle.load(open('./data.pickle', 'rb'))
print(data_dict)

# convert the data and labels into numpy arrays
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# splits the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train) # uses fit for training model

y_predict = model.predict(x_test) # predict the test data

# calculates score
score = accuracy_score(y_predict, y_test) 
print('{}% of samples were classified correctly !'.format(score * 100))

# save the trained Random Forest model
f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()
