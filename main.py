import numpy as np
import pandas as pd
import os
import imageio
import matplotlib.pyplot as plt
from tensorflow import keras
from skimage import transform
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

# funkcija za ocitavanje podataka
def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in os.listdir(label_directory)
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(imageio.imread(f))
            labels.append(int(d))
    return images, labels

# imena znakova/labela
signs_names = ["Warning for a bad road surface", "Warning for a speed bump", "Warning for a slippery road surface",
               "Warning for a curve to the left", "Warning for a curve to the right",
               "Warning for a double curve, first left then right", "Warning for a double curve, first right then left",
               "Warning for children", "Warning for cyclists", "Warning for cattle on the road",
               "Warning for roadworks", "Warning for a traffic light", "Warning for a railroad crossing with barriers",
               "Warning for a danger with no specific traffic sign", "Warning for a road narrowing",
               "Warning for a road narrowing on the left", "Warning for a road narrowing on the right",
               "Warning for a crossroad side roads on the left and right", "Warning for an uncontrolled crossroad",
               "Give way to all drivers", "Give way to oncoming drivers", "Stop", "No entry for vehicular traffic",
               "Cyclists prohibited", "Vehicles heavier than indicated prohibited", "Trucks prohibited",
               "Vehicles wider than indicated prohibited", "Vehicles higher than indicated prohibited",
               "Road closed to all vehicles in both directions", "Turning left prohibited", "Turning right prohibited",
               "Overtaking prohibited", "Maximum speed limit", "Shared use path", "Proceed straight", "Turn left",
               "Proceed straight or turn right", "Roundabout", "Mandatory cycle-way",
               "Track only for cycles and pedestrians", "No parking", "No parking or standing",
               "No parking allowed between 1st - 15th days of the month",
               "No parking allowed between 16st - 131th days of the month", "Priority over oncoming vehicles",
               "Parking", "Parking for invalids", "Parking for cars", "Parking exclusively for lorries",
               "Parking exclusively for buses", "Parking only allowed on the sidewalk", "Begin of a residential area",
               "End of the residential area", "One-way traffic", "No exit", "End of road works",
               "Crossing for pedestrians", "Crossing for cyclists", "Indicating parking", "Speed bump",
               "End of the priority road", "Begin of a priority road"]

# putanja ka podacima
ROOT_PATH = "./input/"
train_data_dir = os.path.join("BelgiumTSC_Training/Training")
test_data_dir = os.path.join("BelgiumTSC_Testing/Testing")

# cuvavanje podataka unutar trening i test lista
train_images, train_labels = load_data(train_data_dir)
test_images, test_labels = load_data(test_data_dir)

# predobrada slika u istu dimenziju
train_images = [transform.resize(image, (32, 32)) for image in train_images]
test_images = [transform.resize(image, (32, 32)) for image in test_images]

# konvertovanje lista u niz
test_images = np.array(test_images)
train_images = np.array(train_images)
test_labels = np.array(test_labels)
train_labels = np.array(train_labels)

# konstrukcija modela
inputs = keras.Input((32,32,3))
layer1 = keras.layers.Conv2D(3, 5, activation="relu", padding='same')(inputs)
layer2 = keras.layers.Conv2D(6, 5, activation="relu", padding='same')(layer1)
layer3 = keras.layers.MaxPooling2D()(layer2)
layer4 = keras.layers.Conv2D(12, 4, activation="relu", padding='same')(layer3)
layer5 = keras.layers.Conv2D(24, 2, activation="relu", padding='same')(layer4)
layer6 = keras.layers.Flatten()(layer5)
layer7 = keras.layers.Dense(200,activation='relu')(layer6)
outputs = keras.layers.Dense(62, activation='softmax')(layer7)
test=0
for j in range(5):
    model = keras.Model(inputs=[inputs], outputs=[outputs])
    # definisanje parametara potrebni za obuku modela
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    # obuka modela
    model.fit(train_images, train_labels, epochs=15,verbose=2)

    # ispis preciznosti u odnosu na test podake
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    test = test + test_acc
print("Accuracy:", test / 5)

# vizualni prikaz funkcionalnosti modela
prediction = model.predict(test_images)
i = 0
plt.axis('off')
plt.text(30, 20, "Truth:    " + signs_names[test_labels[i]]
         + "\nPrediction: " + signs_names[np.argmax(prediction[i])], fontsize=12)
plt.imshow(test_images[i])
plt.show()
