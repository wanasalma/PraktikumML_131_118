# load_model_sample.py
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
from app import APP_ROOT


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(128, 128))
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
    model = load_model(target+"CNN100%_KelompokML.h5")
    labels = ["apple_6", "apple_braeburn_1", "apple_crimson_snow_1",
              "apple_golden_1", "apple_golden_2", "apple_golden_3",
              "apple_granny_smith_1", "apple_hit_1", "apple_pink_lady_1",
              "apple_red_1", "apple_red_2", "apple_red_3",
              "apple_red_delicios_1", "apple_red_yellow_1", "apple_rotten_1",
              "cabbage_white_1", "carrot_1", "cucumber_1",
              "cucumber_3", "eggplant_violet_1", "pear_1",
              "pear_3", "zucchini_1", "zucchini_dark_1"]
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