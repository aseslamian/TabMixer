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
path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed-test'                   # Credit-g(CG) # SEED 10 & DIM 60
# path = r'$(project_folder)/DATA/Credit-approval(CA)/Preprocessed-test'          # Credit-approval(CA)  # SEED 222
# path = r'$(project_folder)/DATA/dress-sale(DS)/Preprocessed-test'               # dress-sale(DS)
# path = r'$(project_folder)/DATA/Adult(AD)/Preprocessed-test'                    # Adult(AD)    # SEED 10 & DIM 105
# path = r'$(project_folder)/DATA/cylinder-bands(CB)/Preprocessed-test'           # cylinder-bands(CB)   # SEED 10 & DIM 555
# path = r'$(project_folder)/DATA/Blastchar(BL)/Preprocessed-test'                # Blastchar(BL) 
# path = r'$(project_folder)/DATA/insurance+company(IO)/Preprocessed'        # insurance+company(IO)   # SEED 10 & DIM 75
# path = r'$(project_folder)/DATA/income (IC)/Preprocessed-test'                  # income (IC)

###############################################################################################################################
# path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed-2/'                   # Credit-g(CG) # SEED 10 & DIM 60
# path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed-3/'                   # Credit-g(CG) # SEED 10 & DIM 60  # Try this
# path = r'$(project_folder)/DATA/Credit-g(CG)/Preprocessed-4/'                   # Credit-g(CG) # SEED 10 & DIM 60  # Try this WM-VAE

# path = r'$(project_folder)/DATA/dress-sale(DS)/Preprocessed-2/'               # dress-sale(DS)
# path = r'$(project_folder)/DATA/dress-sale(DS)/Preprocessed-4/'               # dress-sale(DS)

# path = r'$(project_folder)/DATA/insurance+company(IO)/Preprocessed-2'        # insurance+company(IO)   # SEED 10 & DIM 75

###############################################################################################################################
# path = r'$(project_folder)/DATA/newDATA/1_Bank/preprocessed/'                   # SEED 222
# path = r'$(project_folder)/DATA/newDATA/2_HTRU2/preprocessed/'          
# path = r'$(project_folder)/DATA/newDATA/3_shoppers/preprocessed/'
# path = r'$(project_folder)/DATA/newDATA/4_arrhythmia/preprocessed/'
# path = r'$(project_folder)/DATA/newDATA/5_credit/preprocessed/'
# path = r'$(project_folder)/DATA/newDATA/6_spambase/preprocessed/'
# path = r'$(project_folder)/DATA/newDATA/7_QSAR Bio/preprocessed/'

# path = r'$(project_folder)/DATA/Forest/Preprocessed'                               # Forest

###############################################################################################################################

base_dir = os.path.dirname(__file__)
path = path.replace('$(project_folder)', base_dir)

################################################################################################################################


results = []
# seeds = random.sample(range(1000), 10)
seeds = [10]

random.seed(10)

# for seed in seeds:
# random.seed(seed)

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(path)

# Build model
model = transtab.build_classifier(
    cat_cols, num_cols, bin_cols,
    supervised=True, 
    num_partition=4, 
    overlap_ratio=0.5, 
)

training_arguments = {
    'num_epoch': 1000,
    'batch_size': 128,
    'lr': 1e-4,
    'eval_metric': 'val_loss',
    'eval_less_is_better': True,
    'output_dir': './checkpoint'
}

transtab.train(model, trainset, valset, **training_arguments)

# Test and evaluate model
x_test, y_test = testset
ypred = transtab.predict(model, x_test)
result = transtab.evaluate(ypred, y_test, seed=123, metric='auc')
print(result)
# results.append(result)

# print(results)

# print(max(results))