import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import catboost




train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv').values
sub = pd.read_csv('sample_submission.csv')


X_train = train.iloc[:,1:].values
y_train = train.iloc[:,0:1].values



arr = np.random.randint(42000, size = 10)
for i in arr:
    image = X_train[i].copy()
    image = image.reshape((28,28))
    image = np.array(image, dtype = 'float32')
    
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    plt.imshow(image.astype('uint8'))
    plt.title(y_train[i][0])
    plt.axis('off')
    plt.show()






catboost_model = catboost.CatBoostClassifier()
catboost_model.fit(X_train, y_train)
pred = catboost_model.predict(test)

sub['Label'] = pred
sub.to_csv("sub.csv", index = None)
