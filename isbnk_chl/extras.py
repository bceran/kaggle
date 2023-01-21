import sklearn.linear_model as skl_lm
from sklearn.utils import class_weight

import numpy as np


def find_class_weigths(y_train):
    class_weights = class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)
    print(class_weights)
    # class_weights_dict ={0:0.52119527 , 1:10.29508197}