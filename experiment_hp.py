from __future__ import print_function

from collections import Counter
from datetime import datetime, timedelta

import numpy as np
import pickle
import torch
import math
import time
import os

from datasets import PepBDB_dataset, subset_to_pepbdb_dataset
import models

from torch.utils.data import DataLoader, Subset
from torch.autograd import Variable
from torch import nn, optim

from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

from hyperopt import hp, fmin, tpe, STATUS_OK, SparkTrials, Trials
import pyspark

def config():
    print('\n\tInitializing experiment configurations...')
    exp_settings = {}
    exp_settings['batch_size'] = 128
    exp_settings['num_epochs'] = 1000
    exp_settings['patience'] = 10
    exp_settings['window_size'] = 7
    #exp_settings['root_dir'] = '/home/mhilali/projects/def-bingalls/mhilali/img_output/'  # Root directory for train, val, test
    exp_settings['root_dir'] = './peppi_data_imgs'  # Root directory for train, val, test
    exp_settings['folder'] = './experiment_output'
    if not os.path.exists(exp_settings['folder']):
        os.mkdir(exp_settings['folder'])
    exp_settings['datafile'] = os.path.join(exp_settings['folder'], 'data_summary.txt')
    exp_settings['run_file'] = os.path.join(exp_settings['folder'], 'eval_summary.txt')
    exp_settings['run_counter'] = 0
    exp_settings['glob_loss'] = 10.0
    exp_settings['space'] = {
        'lr': hp.uniform('lr', 0.00001, 0.001),
        'num_kernels': hp.choice('num_kernels', [256, 512, 1024])  # Change range and step size as needed
    }
    return exp_settings

def split_and_load_dataset(dataset, exp_settings, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert train_ratio + val_ratio + test_ratio == 1, "Ratios must sum to 1"
    
    indices = list(range(len(dataset)))
    y = dataset.label_list  # Labels
    
    # Split into train and remaining (validation + test)
    train_indices, remaining_indices = train_test_split(indices, test_size=1-train_ratio, stratify=y, random_state=4)
    remaining_y = [y[i] for i in remaining_indices]
    
    # Split remaining into validation and test
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_indices, test_indices = train_test_split(remaining_indices, test_size=1-val_ratio_adjusted, stratify=remaining_y, random_state=4)

    # Create Subsets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    # Calculate class weights based on the training set
    train_labels = [dataset.label_list[i] for i in train_indices]
    class_counts = Counter(train_labels)
    num_classes = len(class_counts)
    total_samples = len(train_labels)
    class_weights = [total_samples / (num_classes * count) for count in class_counts.values()]
    class_weights = torch.FloatTensor(class_weights)
    
    print('\n\tLoading datasets...')
    sets = {}
    batch_size = exp_settings['batch_size']
    sets['training'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    sets['testing'] = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    sets['validation'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    def save_details():
        for mode, loader in sets.items():
            print('Summarizing ', mode, ' set...')
            with open(exp_settings['datafile'], 'a') as f:
                f.write('\n' + mode + ' set contains ' + str(len(loader.dataset)) + ' samples')
    
    save_details()
    
    return sets, class_weights

def create_subset(dataset, subset_ratio=0.1):
    indices = list(range(len(dataset)))
    y = dataset.label_list
    
    # create a stratified subset for HP tuning
    subset_indices, _ = train_test_split(indices, test_size=1-subset_ratio, stratify=y, random_state=4)
    subset_dataset = Subset(dataset, subset_indices)
    
    return subset_dataset

def train(model, data, lr, class_weights):
    global exp_settings, device

    criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    max_loss = 10
    patience_counter = exp_settings['patience']
    best_val = {}
    best_val['epoch_counter'] = 0
    best_val['val_loss_list'] = []
    best_val['train_loss_list'] = []
    
    for epoch in range(exp_settings['num_epochs']):
        epoch_train_loss, epoch_val_loss = [], []
        
        # Training phase
        model.train()
        for images, labels in data['training']:
            inputs = Variable(images).to(device)
            labels = Variable(labels.long()).to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            epoch_train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        best_val['train_loss_list'].append(sum(epoch_train_loss) / len(epoch_train_loss))
        
        # Validation phase
        val_outputs, val_labels, val_prob = [], [], []
        model.eval()
        for images, labels in data['validation']:
            inputs = Variable(images).to(device)
            labels = Variable(labels.long()).to(device)
            outputs = model(inputs)
            val_loss = criterion(outputs, labels)
            epoch_val_loss.append(val_loss.item())
            out_max = outputs.detach()
            val_prob.append(out_max)
            out_max = torch.argmax(out_max, dim=1)
            val_outputs.append(out_max)
            val_labels.append(labels)
            
        val_loss = sum(epoch_val_loss) / len(epoch_val_loss)
        best_val['val_loss_list'].append(val_loss)
        
        # Early stopping logic
        if val_loss < max_loss:
            max_loss = val_loss
            patience_counter = 0
            best_val['val_outputs'] = torch.cat(val_outputs).cpu().numpy()
            best_val['val_labels'] = torch.cat(val_labels).cpu().numpy()
            best_val['output_prob'] = torch.cat(val_prob).cpu().numpy()[:, 1:]
            best_val['model_state'] = model.state_dict()
            best_val['epoch_counter'] = epoch
        else:
            patience_counter += 1
        if patience_counter == exp_settings['patience']:
            break
        print(f"\tEpoch: {epoch}/{exp_settings['num_epochs']}. Loss: {val_loss}.")    
    # Save the best validation results
    with open(exp_settings['folder'] + '/run' + str(exp_settings['run_counter']) + '.pickle', 'wb') as f:
        pickle.dump(best_val, f)
    
    # Evaluate on the independent test set
    test_outputs, test_labels, test_prob = [], [], []
    model.load_state_dict(best_val['model_state'])
    model.eval()
    for images, labels in data['testing']:
        inputs = Variable(images).to(device)
        labels = Variable(labels.long()).to(device)
        outputs = model(inputs)
        out_max = outputs.detach()
        test_prob.append(out_max)
        out_max = torch.argmax(out_max, dim=1)
        test_outputs.append(out_max)
        test_labels.append(labels)
    
    best_val['test_outputs'] = torch.cat(test_outputs).cpu().numpy()
    best_val['test_labels'] = torch.cat(test_labels).cpu().numpy()
    best_val['test_output_prob'] = torch.cat(test_prob).cpu().numpy()[:, 1:]
    
    return best_val

def calc_metrics(iter_dict):
    scores = {}
    temp = confusion_matrix(iter_dict['val_labels'], iter_dict['val_outputs'])
    TN, FP, FN, TP = temp.ravel()
    scores['sensitivity'] = TP / (FN + TP)
    scores['specificity'] = TN / (TN + FP)
    precision = TP / (TP + FP)
    scores['f_score'] = (2 * scores['sensitivity'] * precision) / (scores['sensitivity'] + precision)
    
    mcc_numerator = (TP * TN) - (FP * FN)
    mcc_denominator = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    if mcc_denominator == 0: mcc_denominator = 1
    scores['mcc'] = mcc_numerator / math.sqrt(mcc_denominator)
    scores['auc'] = roc_auc_score(iter_dict['val_labels'], iter_dict['output_prob'])
    scores['accuracy'] = (TP + TN) / (TN + FP + FN + TP)
    scores['conf_matrix'] = temp
    
    return scores

def pad(x):
    max_len = 11
    x = str(x)
    missing_len = max_len - len(x)
    x = x + (' ' * missing_len)
    return x

def add_header(file, score_dict):
    with open(file, 'a+') as f:
        f.write('\n|' + ('=' * (18 * 11)) + '|')
        f.write('\n|' + pad('Iter') + '|'+ pad('Status')   + '|' + pad('Loss')  + '|'+ pad('So far') + '|'+ pad('Runtime') + 
                '|' + pad('LR') + '|'+ pad('Nodes') + '|'+ pad('Epochs') + '|')
        for key in score_dict.keys():
            if key == 'conf_matrix': f.write(pad('TN FP FN TP') + (' ' * 19) + '|')
            else: f.write(pad(key) + '|')
        f.write('\n|' + ('=' * (18 * 11)) + '|')
        
def save_to_file(file, count, status, loss, sofar, runtime, epochs, lr, num_kernels, score_dict):
    with open(file, 'a+') as f:
        f.write(('\n|' + pad(count)  + '|'+ pad(status)   + '|' + pad(round(loss, 6))+ '|'+ pad(round(sofar, 6)) + '|'+ 
                 pad(runtime)+ '|'+  pad(round(lr, 6)) + '|' + pad(num_kernels) + '|' + pad(epochs + 1)   + '|'))
        for key, val in score_dict.items():
            if key == 'conf_matrix': f.write(str(val.ravel()))
            else: f.write(pad(round(val, 6)) + '|')
        f.write('\n|' + ('-' * (18 * 11)) + '|')

def obj_fn(space):
    global device, exp_settings
    start_time = time.time()
    
    lr = space['lr']
    num_kernels = int(space['num_kernels'])
    model = models.dynamic_model(exp_settings['window_size'], 41, num_kernels).to(device)
    
    best_val = train(model, data, lr, class_weights)
    scores = calc_metrics(best_val)
    
    loss = -scores['auc']  # Minimize negative AUC to maximize AUC
    
    status = STATUS_OK
    if loss < exp_settings['glob_loss']:
        exp_settings['glob_loss'] = loss
    
    runtime = str(timedelta(seconds=round(time.time() - start_time)))
    print(f'Loss: {loss}, Runtime: {runtime}, LR: {lr}, Num Kernels: {num_kernels}')
    
    return {'loss': loss, 'status': status}

def run_trials():
    global exp_settings, data
    
    max_trials = 100  # Number of trials to run
    file_name = exp_settings['folder'] + '/trials_file.hyperopt'
    
    # Use SparkTrials to parallelize the search
    #spark_trials = SparkTrials(parallelism=5)  # Adjust parallelism as needed
    
    try:
        trials = pickle.load(open(file_name, 'rb'))
        print("\n\tFound saved Trials! Loading...")
    except:
        print("\n\tCreating new Trials!...")
        trials = Trials()
    
    best = fmin(fn=obj_fn, space=exp_settings['space'], algo=tpe.suggest, trials=trials, max_evals=max_trials)
    
    print('Best hyperparameters:', best)
    with open(file_name, 'wb') as f:
        pickle.dump(trials, f)

def main():
    global exp_settings, device, data, class_weights
    
    print('\nTraining CNN for predicting protein-peptide binding sites.' + 
          '\n---------------------------------------------------------')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Device:', device)
    
    exp_settings = config()
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Loading dataset...')
    
    pepbdb_dataset = PepBDB_dataset(exp_settings['root_dir'])
    
    # Create a smaller subset for hyperparameter tuning
    subset_dataset = create_subset(pepbdb_dataset, subset_ratio=0.1)
    subset_dataset = subset_to_pepbdb_dataset(subset_dataset, pepbdb_dataset)
    
    # Split the subset dataset
    subset_data, class_weights = split_and_load_dataset(dataset=subset_dataset, exp_settings=exp_settings)
    data = subset_data  # Use the subset data for hyperparameter tuning
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Subset dataset loaded and split.')
    
    run_trials()
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Hyperparameter tuning completed.')
    
    best_hp = pickle.load(open(exp_settings['folder'] + '/trials_file.hyperopt', 'rb')).best_trial['result']['space']
    print('Best hyperparameters:', best_hp)
    
    # Train the final model with the best hyperparameters on the full dataset
    best_lr = best_hp['lr']
    best_num_kernels = int(best_hp['num_kernels'])
    
    # Split the full dataset
    full_data, class_weights = split_and_load_dataset(dataset=pepbdb_dataset, exp_settings=exp_settings)
    data = full_data  # Use the full data for final training
    
    model = models.dynamic_model(exp_settings['window_size'], 41, best_num_kernels).to(device)
    
    best_val = train(model, data, best_lr, class_weights)
    scores = calc_metrics(best_val)
    
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Best validation scores:', scores)

if __name__ == '__main__':
    main()

