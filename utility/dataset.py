import os
import pdb

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import openml
from loguru import logger
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder

# TODO
# organize teh dataset_config for the load_data API.
# dataset_config = {
# 'dataname': { 'cat':[],'bin':[], 'num':[], 
# 'cols':[]}
# }


OPENML_DATACONFIG = {
    'credit-g': {'bin': ['own_telephone', 'foreign_worker']},
}


def load_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):
    if dataset_config is None: dataset_config = OPENML_DATACONFIG
    if isinstance(dataname, str):
        # load a single tabular data
        return load_single_data(dataname=dataname, dataset_config=dataset_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
    
    if isinstance(dataname, list):
        # load a list of datasets, combine together and outputs
        num_col_list, cat_col_list, bin_col_list = [], [], []
        all_list = []
        train_list, val_list, test_list = [], [], []
        for dataname_ in dataname:
            allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = \
                load_single_data(dataname_, dataset_config=dataset_config, encode_cat=encode_cat, data_cut=data_cut, seed=seed)
            num_col_list.extend(num_cols)
            cat_col_list.extend(cat_cols)
            bin_col_list.extend(bin_cols)
            all_list.append(allset)
            train_list.append(trainset)
            val_list.append(valset)
            test_list.append(testset)
        return all_list, train_list, val_list, test_list, cat_col_list, num_col_list, bin_col_list


def load_single_data(dataname, dataset_config=None, encode_cat=False, data_cut=None, seed=123):

    if dataset_config is None: dataset_config = OPENML_DATACONFIG
    print('####'*10)
    if os.path.exists(dataname):
        print(f'load from local data dir {dataname}')
        filename = os.path.join(dataname, 'data_processed.csv')
        df = pd.read_csv(filename, index_col=False)
        y = df['target_label']
        X = df.drop(['target_label'], axis=1)
        all_cols = [col.lower() for col in X.columns.tolist()]

        X.columns = all_cols
        attribute_names = all_cols
        ftfile = os.path.join(dataname, 'numerical_feature.txt')
        if os.path.exists(ftfile):
            with open(ftfile,'r') as f: num_cols = [x.strip().lower() for x in f.readlines()]
        else:
            num_cols = []
        bnfile = os.path.join(dataname, 'binary_feature.txt')
        if os.path.exists(bnfile):
            with open(bnfile,'r') as f: bin_cols = [x.strip().lower() for x in f.readlines()]
        else:
            bin_cols = []
        cat_cols = [col for col in all_cols if col not in num_cols and col not in bin_cols]

        # update cols by loading dataset_config
        if dataname in dataset_config:
            data_config = dataset_config[dataname]
            if 'columns' in data_config:
                new_cols = dataset_config[dataname]['columns']
                X.columns = new_cols

            if 'bin' in data_config:
                bin_cols = data_config['bin']
            
            if 'cat' in data_config:
                cat_cols = data_config['cat']

            if 'num' in data_config:
                num_cols = data_config['num']
        
    else:
        dataset = openml.datasets.get_dataset(dataname)
        X,y,categorical_indicator, attribute_names = dataset.get_data(dataset_format='dataframe', target=dataset.default_target_attribute)
        
        if isinstance(dataname, int):
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            dataname = openml_list.loc[openml_list.did == dataname].name.values[0]
        else:
            openml_list = openml.datasets.list_datasets(output_format="dataframe")  # returns a dict
            print(f'openml data index: {openml_list.loc[openml_list.name == dataname].index[0]}')
        
        print(f'load data from {dataname}')

        # drop cols which only have one unique value
        drop_cols = [col for col in attribute_names if X[col].nunique()<=1]

        all_cols = np.array(attribute_names)
        categorical_indicator = np.array(categorical_indicator)
        cat_cols = [col for col in all_cols[categorical_indicator] if col not in drop_cols]
        num_cols = [col for col in all_cols[~categorical_indicator] if col not in drop_cols]
        all_cols = [col for col in all_cols if col not in drop_cols]
        
        if dataname in dataset_config:
            if 'bin' in dataset_config[dataname]: bin_cols = [c for c in cat_cols if c in dataset_config[dataname]['bin']]
        else: bin_cols = []
        cat_cols = [c for c in cat_cols if c not in bin_cols]

    # encode target label
    y = LabelEncoder().fit_transform(y.values)
    y = pd.Series(y, index=X.index)

    # start processing features
    # process num
    if len(num_cols) > 0:
        for col in num_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        X[num_cols] = MinMaxScaler().fit_transform(X[num_cols])

    if len(cat_cols) > 0:
        for col in cat_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        # process cate
        if encode_cat:
            X[cat_cols] = OrdinalEncoder().fit_transform(X[cat_cols])
        else:
            X[cat_cols] = X[cat_cols].astype(str)

    if len(bin_cols) > 0:
        for col in bin_cols: X[col].fillna(X[col].mode()[0], inplace=True)
        if dataname in dataset_config:
            if 'binary_indicator' in dataset_config[dataname]:
                X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in dataset_config[dataname]['binary_indicator'] else 0).values
        else:
            X[bin_cols] = X[bin_cols].astype(str).applymap(lambda x: 1 if x.lower() in ['yes','true','1','t'] else 0).values
        # if no dataset_config given, keep its original format
    
    X = X[bin_cols + num_cols + cat_cols]

    # rename column names if is given
    if dataname in dataset_config:
        data_config = dataset_config[dataname]
        if 'columns' in data_config:
            new_cols = data_config['columns']
            X.columns = new_cols
            attribute_names = new_cols

        if 'bin' in data_config:
            bin_cols = data_config['bin']
        
        if 'cat' in data_config:
            cat_cols = data_config['cat']

        if 'num' in data_config:
            num_cols = data_config['num']

    # split train/val/test
    ##########################################################################
            
    train_dataset, test_dataset, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y, shuffle=True)

    ######## Splite with the same ratio (pos/neg) in Test and Train ##################

    # X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split( X[y==1], y[y==1], test_size=0.2, random_state=seed, stratify=y[y==1])
    # X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split( X[y==0], y[y==0], test_size=0.2, random_state=seed, stratify=y[y==0])

    # train_dataset = pd.concat([X_train_pos, X_train_neg])
    # y_train = pd.concat([y_train_pos, y_train_neg])
    # test_dataset = pd.concat([X_test_pos, X_test_neg])
    # y_test = pd.concat([y_test_pos, y_test_neg])

    ################ SMOTE ##################
            
    # X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split( X[y==1], y[y==1], test_size=0.2, random_state=seed, stratify=y[y==1])
    # X_train_neg, X_test_neg, y_train_neg, y_test_neg = train_test_split( X[y==0], y[y==0], test_size=0.2, random_state=seed, stratify=y[y==0])
    # train_dataset = pd.concat([X_train_pos, X_train_neg])
    # y_train = pd.concat([y_train_pos, y_train_neg])
    # test_dataset = pd.concat([X_test_pos, X_test_neg])
    # y_test = pd.concat([y_test_pos, y_test_neg])
    # numerical_columns = train_dataset.select_dtypes(include=['int64', 'float64']).columns
    # categorical_columns = train_dataset.select_dtypes(include=['object']).columns
    # encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    # train_categorical_encoded = encoder.fit_transform(train_dataset[categorical_columns])
    # train_categorical_encoded_df = pd.DataFrame(train_categorical_encoded, columns=encoder.get_feature_names_out(categorical_columns), index=train_dataset.index)
    # train_X = pd.concat([train_dataset[numerical_columns], train_categorical_encoded_df], axis=1)
    # smote = SMOTE(sampling_strategy='auto', random_state=seed) 
    # train_dataset, y_train = smote.fit_resample(train_X, y_train)
    
    ###########################################################################

    val_size = int(len(y)*0.1)
    val_dataset = train_dataset.iloc[-val_size:]
    y_val = y_train[-val_size:]
    train_dataset = train_dataset.iloc[:-val_size]
    y_train = y_train[:-val_size]

    if data_cut is not None:
        np.random.shuffle(all_cols)
        sp_size=int(len(all_cols)/data_cut)
        col_splits = np.split(all_cols, range(0,len(all_cols),sp_size))[1:]
        new_col_splits = []
        for split in col_splits:
            candidate_cols = np.random.choice(np.setdiff1d(all_cols, split), int(sp_size/2), replace=False)
            new_col_splits.append(split.tolist() + candidate_cols.tolist())
        if len(col_splits) > data_cut:
            for i in range(len(col_splits[-1])):
                new_col_splits[i] += [col_splits[-1][i]]
                new_col_splits[i] = np.unique(new_col_splits[i]).tolist()
            new_col_splits = new_col_splits[:-1]

        # cut subset
        trainset_splits = np.array_split(train_dataset, data_cut)
        train_subset_list = []
        for i in range(data_cut):
            train_subset_list.append(
                (trainset_splits[i][new_col_splits[i]], y_train.loc[trainset_splits[i].index])
            )
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X, y), train_subset_list, (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols

    else:
        print('# data: {}, # feat: {}, # cate: {},  # bin: {}, # numerical: {}, pos rate: {:.2f}'.format(len(X), len(attribute_names), len(cat_cols), len(bin_cols), len(num_cols), (y==1).sum()/len(y)))
        return (X,y), (train_dataset,y_train), (val_dataset,y_val), (test_dataset, y_test), cat_cols, num_cols, bin_cols