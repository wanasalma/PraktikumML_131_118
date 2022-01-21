# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from app import APP_ROOT


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(100, 100))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


def predict_img(path):
    # load model
    print(os.listdir(APP_ROOT))
    target = os.path.join(APP_ROOT, 'static/')
    model = load_model(target+"CNN1_KelompokML.h5")
    labels = ["Apple Braeburn", "Apple Golden 1", "Apple Golden 2",
              "Apple Golden 3", "Apple Granny Smith", "Apple Red 1",
              "Apple Red 2", "Apple Red 3", "Apple Red Delicious",
              "Apple Red Yellow 1", "Apple Red Yellow 2", "Cherry 1",
              "Cherry 2", "Cherry Rainier", "Cherry Wax Black",
              "Cherry Wax Red", "Cherry Wax Yellow", "Grape Blue",
              "Grape Pink", "Grape White", "Grape White 2", "Grape White 3",
              "Grape White 4", "Grapefruit Pink", "Grapefruit White"]
    # image path
    target = os.path.join(APP_ROOT, 'temp/')
    img_path = (target+path)    # dog
    #img_path = '/media/data/dogscats/test1/19.jpg'      # cat
    # load a single image
    new_image = load_image(img_path)

    # check prediction
    pred = model.predict(new_image)
    label = np.argmax(pred)
    #os.remove(img_path)
    return labels[label]