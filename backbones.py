import DataReaders
import ModelGetter
import random
import cv2
import numpy as np
import train_test_spliter
import pandas as pd
import model
from Learner import Learner
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Module

#############################################
##1.image transformers:
#############################################
def transform(img):

    # AUGMENTATION VARIABLES
    RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change
    RANDOM_CONTRAST = 5  # range (0-100), 0=no change

    # We flip it to rgb for visualization purposes
    img = cv2.resize(img, (331, 331))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = img / 255.

    # Random flip
    flip_hor = bool(random.getrandbits(1))
    flip_ver = bool(random.getrandbits(1))
    if (flip_hor):
        img = img[:, ::-1]
    if (flip_ver):
        img = img[::-1, :]

    # Random brightness
    br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
    img = img + br

    # Random contrast
    cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
    img = img * cr

    # clip values to 0-1 range
    img = np.clip(img, 0, 1.0)

    #unsqueeze:
    img = img.unsqueeze(0)

    return img
def test_trans(img):
    img = cv2.resize(img, (331, 331))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = img / 255.

    #unsqueeze:
    img = img.unsqueeze(0)

    return img

#############################################
##1. create the dataloader using DataReaders:
##optional: split the csv to train and test
#############################################
train_test_spliter.split_train_test("/kaggle/input/train_labels.csv", 0.1, 123, "/kaggle/input/train.csv", "/kaggle/input/test.csv")
train_df = pd.read_csv("/kaggle/input/train.csv")
test_df = pd.read_csv("/kaggle/input/test.csv")
train_loader = DataReaders.getDataLoaderFromCSV(train_df, "/kaggle/input/train", ".tif", transform, 8, True, 8)
test_loader = DataReaders.getDataLoaderFromCSV(test_df, "kaggle/input/train", ".tif", test_trans, 8, True, 8)


#############################################
##2. create the model and loss and optimizer:
#############################################
model = model.PNASNet5Large()
model = ModelGetter.getCUDAModel(model) #type:Module

loss = CrossEntropyLoss()

optimizer = Adam(model.parameters(), lr=0.0001)


###########################################
##epoch after function:
###########################################
def epoch_after_fn(learner:Learner, epoch):
    print("evaluating epoch {}".format(epoch), end=" ")
    learner.evl_model(True)

#####################
#learner
#####################
learner = Learner(train_loader = train_loader, model=model, epochs=6, loss_fn=loss, optimizer=optimizer, prt_loss_ite=5, test_loader=test_loader, ite_after_fn=None, epoch_after_fn=epoch_after_fn)
learner.learn()