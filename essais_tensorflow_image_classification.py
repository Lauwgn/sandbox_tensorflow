# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
print(train_labels[:5])
print(len(train_labels))

# Une image : np.array 28 x 28, chaque valeur entre 0 et 255 donnant une couleur

# plt.figure()
# plt.imshow(train_images[10])
# plt.colorbar()
# plt.grid(False)
# plt.show()

""" bricolages pour faire des tests """
# x = train_images[10].copy()
# x[0, :] = [0 for i in range(x.shape[0])]
# print(x)
# plt.figure()
# plt.imshow(x)
# plt.colorbar()
# plt.grid(False)
# plt.show()
# exit()






train_images = train_images / 255.0
test_images = test_images / 255.0

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, train_labels, batch_size=128, epochs=5)

# test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images)

print(predictions[0])
print(np.sum(predictions[0]))
print(np.argmax(predictions[0]))

# Grab an image from the test dataset.
img = test_images[1]
print(img.shape)










