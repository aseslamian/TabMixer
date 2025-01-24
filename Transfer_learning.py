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
path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed'                   # Credit-g(CG) # SEED 10 & DIM 60
# path = r'$(project_folder)/DATA/Credit-approval(CA)/Preprocessed'          # Credit-approval(CA)  # SEED 222
# path = r'$(project_folder)/DATA/dress-sale(DS)/Preprocessed'               # dress-sale(DS)
# path = r'$(project_folder)/DATA/Adult(AD)/Preprocessed'                    # Adult(AD)    # SEED 10 & DIM 105
# path = r'$(project_folder)/DATA/cylinder-bands(CB)/Preprocessed'           # cylinder-bands(CB)   # SEED 10 & DIM 555
# path = r'$(project_folder)/DATA/Blastchar(BL)/Preprocessed'                # Blastchar(BL) 
# path = r'$(project_folder)/DATA/insurance+company(IO)/Preprocessed'        # insurance+company(IO)   # SEED 10 & DIM 75
# path = r'$(project_folder)/DATA/income (IC)/Preprocessed'                  # income (IC)
###############################################################################################################################

base_dir = os.path.dirname(__file__)
path = path.replace('$(project_folder)', base_dir)

################################################################################################################################

parent_path = os.path.dirname(path)  
target_path = os.path.join(parent_path, 'TransferLearning')

path1 = os.path.join(target_path, 'data1')
path2 = os.path.join(target_path, 'data2')

##########################################################################################################################

results = []
# seeds = random.sample(range(1000), 20)
seeds = [10]

for seed in seeds:

    random.seed(seed)
    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(path1) 

    # build transtab classifier model
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

    # specify training arguments, take validation loss for early stopping
    training_arguments = {
        'num_epoch':500,
        'batch_size':128,
        'lr': 1e-4,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint'
        }
    transtab.train(model, trainset, valset, **training_arguments)

    ##########################################################################################################################

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(path2) 

    # load the pretrained model
    model.load('./checkpoint')
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

    transtab.train(model, trainset, valset, **training_arguments)

    #########################################################################################################################
    x_test, y_test = testset

    ypred = transtab.predict(model, x_test)
    result = transtab.evaluate(ypred, y_test, seed=123, metric='auc')
    results.append(result)

print(results)