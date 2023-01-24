import tensorflow as tf
import tensorflow_datasets as tfds

data = tfds.load("iris", split='train[:80%]', as_supervised=True)
print(data.head(10))