import pickle
import os
path = "supervised_learning.pickle"
if os.path.exists(path):
    with open(path, 'rb') as f:
        t, d = pickle.load(f)
print("hello")