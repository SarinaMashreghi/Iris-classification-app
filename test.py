import pickle
import numpy as np

model = pickle.load(open('iris_model.pkl','rb'))
classes = ['Iris Setosa', 'Iris Versicolor', 'Iris Virginica']
values = np.array([[1,1,1,1]]).astype('float32')
print(classes[model.predict(values).argmax()])