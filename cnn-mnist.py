from mnist import MNIST
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mndata = MNIST('./data')
mndata.gz = True
train_images, train_labels = mndata.load_training()
train_mod_labels = pd.get_dummies(train_labels).to_numpy()

test_images, test_labels = mndata.load_testing()
# test_labels = np.array(test_labels)
# test_mod_labels = pd.get_dummies(test_labels).to_numpy()


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten

unravelled_train_images = []
for img in train_images:
    unravelled_train_images.append(np.array(img).reshape(28,28,1))
    
unravelled_test_images = []
for img in test_images:
    unravelled_test_images.append(np.array(img).reshape(28,28,1))


model4 = Sequential()
model4.add(Conv2D(32, kernel_size=(3, 3), activation='relu',input_shape=(28,28,1)))
model4.add(Conv2D(64, (3, 3), activation='relu'))
model4.add(MaxPooling2D(pool_size=(2, 2)))
model4.add(Dropout(0.25))
model4.add(Flatten())
model4.add(Dense(128, activation='relu'))
model4.add(Dropout(0.5))
model4.add(Dense(10, activation='softmax'))


model4.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['accuracy'])
model4.fit(np.array(unravelled_train_images), train_mod_labels, epochs=8, batch_size=1000, verbose=0)

y_pred = model4.predict(np.array(unravelled_test_images))
pred4 = list()
for i in range(len(y_pred)):
    pred4.append(np.argmax(y_pred[i]))
test_labels = np.array(test_labels)

for p in pred4:
	print(p)