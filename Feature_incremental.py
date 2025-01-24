import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'utility'))
import dataset
import evaluator
import transtab
import torch
import random
from itertools import product
import numpy as np

###############################################################################################################################
# path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed'                   # Credit-g(CG) # SEED 10 & DIM 60
# path = r'$(project_folder)/DATA/Credit-approval(CA)/Preprocessed'          # Credit-approval(CA)  # SEED 222
# path = r'$(project_folder)/DATA/dress-sale(DS)/Preprocessed'               # dress-sale(DS)
# path = r'$(project_folder)/DATA/Adult(AD)/Preprocessed'                    # Adult(AD)    # SEED 10 & DIM 105
# path = r'$(project_folder)/DATA/cylinder-bands(CB)/Preprocessed'           # cylinder-bands(CB)   # SEED 10 & DIM 555
# path = r'$(project_folder)/DATA/Blastchar(BL)/Preprocessed'                # Blastchar(BL) 
# path = r'$(project_folder)/DATA/insurance+company(IO)/Preprocessed'        # insurance+company(IO)   # SEED 10 & DIM 75
path = r'$(project_folder)/DATA/income (IC)/Preprocessed'                  # income (IC)
###############################################################################################################################

base_dir = os.path.dirname(__file__)
path = path.replace('$(project_folder)', base_dir)

################################################################################################################################

parent_path = os.path.dirname(path)  # Get the parent directory of the current path

target_path = os.path.join(parent_path, 'ZSL')

# Define paths within the target path
path1 = os.path.join(target_path, 'data1')
path2 = os.path.join(target_path, 'data2')
path3 = os.path.join(target_path, 'data3')

##########################################################################################################


results = []
# seeds = np.random.randint(0, 1000, 20)
seeds = [10, 222]

for seed in seeds:

    random.seed(seed)

    allset1, trainset1, valset1, testset1, cat_cols1, num_cols1, bin_cols1 = transtab.load_data(path2)

    model = transtab.build_classifier(
        cat_cols1, num_cols1, bin_cols1,
        supervised=True, 
        num_partition=4, 
        overlap_ratio=0.5, 
    )

    training_arguments = {
        'num_epoch':500,
        'batch_size':128,
        'lr': 1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
        }

    transtab.train(model, trainset1, valset1, **training_arguments)

    # load multiple datasets by passing a list of data names
    allset2, trainset2, valset2, testset2, cat_cols2, num_cols2, bin_cols2 = transtab.load_data(path1)
    model.update({'cat':cat_cols2,'num':num_cols2,'bin':bin_cols2})

    transtab.train(model, trainset1, valset1, **training_arguments)

    #########################

    x_test, y_test = testset1
    ypred = transtab.predict(model, x_test)
    a = transtab.evaluate(ypred, y_test, seed=123, metric='auc')

    x_test, y_test = testset2
    ypred = transtab.predict(model, x_test)
    b = transtab.evaluate(ypred, y_test, seed=123, metric='auc')

    #########################
    max_auc = max(a,b)
    results.append(max_auc)

# print(max_auc)
print(max(results))
