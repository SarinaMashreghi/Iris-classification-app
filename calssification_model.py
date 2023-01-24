import pickle

import tensorflow as tf
import tensorflow_datasets as tfds

data = tfds.load("iris", split='train[:80%]', as_supervised=True)
data = data.batch(10).map(lambda x, y: (x, tf.one_hot(y, depth=3))).repeat()

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(data, steps_per_epoch=10, epochs=100)

pickle.dump(model, open('iris_model.pkl', 'wb'))