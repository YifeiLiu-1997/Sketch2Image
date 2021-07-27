"""
    sketch2image:
    Asian face sketch -> real face image
"""

from my_utils import *
from model import *
import numpy as np
import os

# data preprocess
SIZE = 256

image_path = 'datasets/photos'
sketch_path = 'datasets/sketches'

image_file = sorted_path(os.listdir(image_path))
sketch_file = sorted_path(os.listdir(sketch_path))

img_array = expand_image_array(path=image_path, path_list=image_file, size=(SIZE, SIZE))
sketch_array = expand_image_array(path=sketch_path, path_list=sketch_file, size=(SIZE, SIZE))

print("Total number of sketch images:", len(sketch_array))
print("Total number of images:", len(img_array))

# show example
plot_images(img_array[0], sketch_array[0])

train_sketch_image = sketch_array[:1400]
train_image = img_array[:1400]
test_sketch_image = sketch_array[1400:]
test_image = img_array[1400:]

# reshaping
train_sketch_image = np.reshape(train_sketch_image, (len(train_sketch_image), SIZE, SIZE, 3))
train_image = np.reshape(train_image, (len(train_image), SIZE, SIZE, 3))
print('Train color image shape:', train_image.shape)
test_sketch_image = np.reshape(test_sketch_image, (len(test_sketch_image), SIZE, SIZE, 3))
test_image = np.reshape(test_image, (len(test_image), SIZE, SIZE, 3))
print('Test color image shape', test_image.shape)

# create model
model = Sketch2Image()

# optimizer and loss
model.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error,
              metrics=['accuracy'])

# training
history = model.fit(train_sketch_image, train_image, epochs=0, validation_data=(test_sketch_image, test_image),
                    validation_freq=1)
# model.summary()

# plot metric
plot_metrics(history, metric='accuracy', show=False)

# save model1
save_path = 'my_model/my_model'
save_all_model(model, save_path, save=False)

# load model
fresh_model = load_all_model(model, save_path, load=True)

# show predict
ls = [i for i in range(16, 40, 8)]
for i in ls:
    predicted = np.clip(fresh_model.predict(test_image[i].reshape(1, SIZE, SIZE, 3)), 0.0, 1.0).reshape((SIZE, SIZE, 3))
    print('pred', predicted.shape)
    show_images(test_image[i], test_sketch_image[i], predicted)

# test my own image
show_my_images('my_images/criminal_1.jpg', model=fresh_model, size=SIZE)
