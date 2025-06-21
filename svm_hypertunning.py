import sklearn 
import math
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# get data
train = pd.read_csv("train_data.csv.gz")
X_train, y_train = train.drop("Activity", axis=1), train["Activity"]

test = pd.read_csv("test_data.csv.gz")
X_test, y_test = test.drop("Activity", axis=1), test["Activity"]

val =  pd.read_csv("val_data.csv.gz")
X_val, y_val = val.drop("Activity", axis=1), val["Activity"]

#############################################################################
# hyper_parameter tunning

#module for optimization
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.model_selection import cross_val_score
# module for logging data 
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
# module for retriving datat 
from bayes_opt.util import load_logs# bounded parameter regions 

# set bounds for optimization (must be ranges of bounds => (lower bound, upper bound))
pbounds = {
    "C" : (0.01, 100),          # C parameter bounds 
    "gamma" : (0.001, 0.5),     #  gamma bounds 
    "k" : (0, 4)                   # kernel bounds 
}

# define wrapped funciton
def svm_object(C, gamma, k):
    # some categorical parameter
    kernel = ['linear', 'poly', 'rbf', 'sigmoid'][math.floor(k)]
    # model 
    svm = SVC(C=C, gamma=gamma, kernel=kernel, random_state=33)
    scores = cross_val_score(svm, X_val, y_val, cv=5, scoring="f1_weighted" )
    return scores.mean()

# create instance of optimizer 
optimizer1 = BayesianOptimization(
    f = svm_object,
    pbounds = pbounds,
    random_state = 1
)

# create UtilityFunction object for aqu. function
utility = UtilityFunction(kind = "ei", xi= 0.02)

# set gaussian process parameter
optimizer1.set_gp_params(alpha = 1e-6)

# create logger 
logger = JSONLogger(path = "./tunning1.log")
optimizer1.subscribe(Events.OPTIMIZATION_STEP, logger)

# initial search 
optimizer1.maximize(
    init_points = 5, # number of random explorations before bayes_opt
    n_iter = 15, # number of bayes_opt iterations
)

# print out the data from the initial run to check if bounds need update 
for i, param in enumerate(optimizer1.res):
    print(f"Iteration {i}: \n\t {param}")

# get best parameter
print("Best Parameters found: ")
print(optimizer1.max)

################################################################

# train SVM one-to-one
prep = "smt_here"
pipe = PipeLine([("preproc", prep), ("clf", SVC(decision_function_shape="ovo"))])
pipe.fit(X_train, y_train)

# predict and evaluate 
y_pred = pipe.predict(X_test)
print("One-vs-one Accuracy:", accuracy_score(y_test, y_pred))

