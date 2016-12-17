from helper import *
import os



# Change this to where the Data from this project CamVid is:
# https://github.com/alexgkendall/SegNet-Tutorial
path = '../SegNet-Tutorial/CamVid/'


def load_data(mode):
    train_data = []
    train_label = []
    with open(path + mode +'.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]
    for i in range(len(txt)):
        train_data.append(np.rollaxis(normalized(cv2.imread(os.getcwd() + txt[i][0][7:])),2))
        train_label.append(binarylab(cv2.imread(os.getcwd() + txt[i][1][7:][:-1])[:,:,0]))
        print('.',end='')
    return np.array(train_data), np.array(train_label)


train_data, train_label = prep_data()

train_label = np.reshape(train_label,(367,data_shape,12))

np.save("train_data", x)

np.save("train_label", x)

# FYI they are:
# Sky = [128,128,128]
# Building = [128,0,0]
# Pole = [192,192,128]
# Road_marking = [255,69,0]
# Road = [128,64,128]
# Pavement = [60,40,222]
# Tree = [128,128,0]
# SignSymbol = [192,128,128]
# Fence = [64,64,128]
# Car = [64,0,128]
# Pedestrian = [64,64,0]
# Bicyclist = [0,128,192]
# Unlabelled = [0,0,0]