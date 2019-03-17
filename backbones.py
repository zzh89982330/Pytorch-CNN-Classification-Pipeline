import random
import cv2
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.nn import Module
from torchvision.transforms import Normalize, Compose, ToTensor

from ModelGetter import getCUDAModel
from lr_finder import LRFinder
from model import pnasnet5large
from train_test_spliter import split_train_test
from Learner import Learner

#############################################
##1.image transformers:
#############################################
from DataReaders import getDataLoaderFromCSV


def transform(img):

    # AUGMENTATION VARIABLES
    RANDOM_BRIGHTNESS = 7  # range (0-100), 0=no change
    RANDOM_CONTRAST = 5  # range (0-100), 0=no change

    # We flip it to rgb for visualization purposes
    img = cv2.resize(img, (331, 331))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(img)
    img = img.numpy()
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
    return img

def test_trans(img):
    img = cv2.resize(img, (331, 331))
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    img = Compose([ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])(img)
    img = img.numpy()
    img = img / 255.

    #unsqueeze:
    #img = np.expand_dims(img, 0)

    return img


#############################################
##1. create the dataloader using DataReaders:
##optional: split the csv to train and test
#############################################
train_df, test_df = split_train_test("/kaggle/input/train_labels.csv", 0.1, 123, None, None)
train_loader = getDataLoaderFromCSV(train_df, "/kaggle/input/train", ".tif", transform, 64, True, 0)
test_loader = getDataLoaderFromCSV(test_df, "kaggle/input/train", ".tif", test_trans, 64, True, 0)


#############################################
##2. create the model and loss and optimizer:
#############################################
model = pnasnet5large()
model = getCUDAModel(model) #type:Module
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=5e-4)

#####################################
##2.1 lr_finder (fast.ai resembling)
## comment these lines if not used
#####################################
optimizer = Adam(model.parameters(), lr=1e-7, weight_decay=1e-2)
lr_finder = LRFinder(model, optimizer, loss, device="cuda")
lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
lr_finder.plot()

###########################################
##epoch after function:
###########################################
def epoch_after_fn(learner:Learner, epoch):
    print("evaluating epoch {}".format(epoch), end=" ")
    learner.evl_model(True)

#####################
#learner
#####################
#learner = Learner(train_loader = train_loader, model=model, epochs=6, loss_fn=loss, optimizer=optimizer, prt_loss_ite=5, test_loader=test_loader, ite_after_fn=None, epoch_after_fn=[epoch_after_fn])
#learner.learn()
