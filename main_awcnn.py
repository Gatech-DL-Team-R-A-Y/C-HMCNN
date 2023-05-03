import os
import importlib
os.environ["DATA_FOLDER"] = "./"

import argparse

import time

import torch
import torch.utils.data
import torch.nn as nn

import random

# from utils.parser import *
from utils.parser_aw_cnn import *
from utils import datasets

from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, hamming_loss, f1_score, jaccard_score

import numpy

from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve, roc_auc_score, auc
from models import AW_CNN, AW_ConstrainedFFNNModel
from visualization.viz import draw_loss_acc


def get_constr_out(x, R):
    """ Given the output of the neural network x returns the output of MCM given the hierarchy constraint expressed in the matrix R """
    c_out = x.double()
    c_out = c_out.unsqueeze(1)
    # print('AW debug len(x), R.shape[1]', len(x), R.shape[1])
    c_out = c_out.expand(len(x),R.shape[1], R.shape[1])
    R_batch = R.expand(len(x),R.shape[1], R.shape[1])
    final_out, _ = torch.max(R_batch*c_out.double(), dim = 2)
    return final_out


def main():

    parser = argparse.ArgumentParser(description='Train neural network on train and validation set')

    # Required  parameter
    parser.add_argument('--dataset', type=str, default=None, required=True,
                        help='dataset name, must end with: "_GO", "_FUN", or "_others"' )
    # Other parameters
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed (default: 0)')
    parser.add_argument('--device', type=str, default='0',
                        help='GPU (default:0)')
    parser.add_argument('--model', type=str, default='fc',
                        help='model (default:fc)')   
    args = parser.parse_args()

    assert('_' in args.dataset)
    assert('FUN' in args.dataset or 'GO' in args.dataset or 'others' in args.dataset)

    # Load train, val and test set
    dataset_name = args.dataset
    data = dataset_name.split('_')[0]
    ontology = dataset_name.split('_')[1]
    # print('AW Debug:', data)
    # print('AW Debug ontology:', ontology)

    # Dictionaries with number of features and number of labels for each dataset
    input_dims = {'diatomsCNNDebug':196608, 'diatomsCNN':196608, 'diatoms':371, 'enron':1001,'imclef07a': 80, 'imclef07d': 80,'cellcycle':77, 'derisi':63, 'eisen':79, 'expr':561, 'gasch1':173, 'gasch2':52, 'seq':529, 'spo':86}
    output_dims_others = {'diatomsCNNDebug':398, 'diatomsCNN':398, 'diatoms':398,'enron':56, 'imclef07a': 96, 'imclef07d': 46, 'reuters':102}
    output_dims = {'others':output_dims_others}

    #Dictionaries with the hyperparameters associated to each dataset
    hidden_dims_others = {'diatoms':2000, 'enron':1000,'imclef07a':1000, 'imclef07d':1000, 'diatomsCNNDebug':2048, 'diatomsCNN':2048}
    hidden_dims = {'others':hidden_dims_others}
    lrs_FUN = {'cellcycle':1e-4, 'derisi':1e-4, 'eisen':1e-4, 'expr':1e-4, 'gasch1':1e-4, 'gasch2':1e-4, 'seq':1e-4, 'spo':1e-4}
    lrs_GO = {'cellcycle':1e-4, 'derisi':1e-4, 'eisen':1e-4, 'expr':1e-4, 'gasch1':1e-4, 'gasch2':1e-4, 'seq':1e-4, 'spo':1e-4}
    lrs_others = {'diatoms':1e-5, 'enron':1e-5,'imclef07a':1e-5, 'imclef07d':1e-5, 'diatomsCNNDebug':1e-4, 'diatomsCNN':1e-6}
    lrs = {'FUN':lrs_FUN, 'GO':lrs_GO, 'others':lrs_others}
    epochss_FUN = {'cellcycle':106, 'derisi':67, 'eisen':110, 'expr':20, 'gasch1':42, 'gasch2':123, 'seq':13, 'spo':115}
    epochss_GO = {'cellcycle':62, 'derisi':91, 'eisen':123, 'expr':70, 'gasch1':122, 'gasch2':177, 'seq':45, 'spo':103}
    epochss_others = {'diatoms':474, 'enron':133,'imclef07a':592, 'imclef07d':588, 'diatomsCNNDebug':10, 'diatomsCNN':40}
    epochss = {'FUN':epochss_FUN, 'GO':epochss_GO, 'others':epochss_others}

    # Set the hyperparameters 
    batch_size = 4
    num_layers = 3
    dropout = 0.7
    non_lin = 'relu'
    hidden_dim = hidden_dims[ontology][data]
    lr = lrs[ontology][data]
    weight_decay = 1e-5
    num_epochs = epochss[ontology][data]
    hyperparams = {'batch_size':batch_size, 'num_layers':num_layers, 'dropout':dropout, 'non_lin':non_lin, 'hidden_dim':hidden_dim, 'lr':lr, 'weight_decay':weight_decay}


    # Set seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:" + str(args.device) if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available:
        pin_memory = True


    # Load the datasets
    if ('others' in args.dataset):
        train, test = initialize_other_dataset(dataset_name, datasets)
        print('AW debug ------------------------------------------------------')
        print('AW debug len(train.X), len(test.X)', len(train.X), len(test.X))
        print('AW debug ------------------------------------------------------')
        train.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8),  torch.tensor(test.to_eval, dtype=torch.uint8)
    else:
        train, val, test = initialize_dataset(dataset_name, datasets)
        train.to_eval, val.to_eval, test.to_eval = torch.tensor(train.to_eval, dtype=torch.uint8), torch.tensor(val.to_eval, dtype=torch.uint8), torch.tensor(test.to_eval, dtype=torch.uint8)
    
    different_from_0 = torch.tensor(np.array((test.Y.sum(0)!=0), dtype = np.uint8), dtype=torch.uint8)

    # Compute matrix of ancestors R
    # Given n classes, R is an (n x n) matrix where R_ij = 1 if class i is descendant of class j
    R = np.zeros(train.A.shape)
    np.fill_diagonal(R, 1)
    g = nx.DiGraph(train.A) # train.A is the matrix where the direct connections are stored 
    for i in range(len(train.A)):
        ancestors = list(nx.descendants(g, i)) #here we need to use the function nx.descendants() because in the directed graph the edges have source from the descendant and point towards the ancestor 
        if ancestors:
            R[i, ancestors] = 1
    R = torch.tensor(R)
    #Transpose to get the descendants for each node 
    R = R.transpose(1, 0)
    R = R.unsqueeze(0).to(device)


    # Rescale data and impute missing data
    if ('others' in args.dataset):
        pass
        # scaler = preprocessing.StandardScaler().fit((train.X.astype(float)))
        # print('AW Debug: This is other dataset!')
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit((train.X.astype(float)))
    else:
        scaler = preprocessing.StandardScaler().fit(np.concatenate((train.X, val.X)))
        imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit(np.concatenate((train.X, val.X)))
        val.X, val.Y = torch.tensor(scaler.transform(imp_mean.transform(val.X))).to(device), torch.tensor(val.Y).to(device)
    # train.X, train.Y = torch.tensor(scaler.transform(imp_mean.transform(train.X))).to(device), torch.tensor(train.Y).to(device)        
    train.X, train.Y = torch.tensor(train.X).to(device), torch.tensor(train.Y).to(device)        
    # test.X, test.Y = torch.tensor(scaler.transform(imp_mean.transform(test.X))).to(device), torch.tensor(test.Y).to(device)
    test.X, test.Y = torch.tensor(test.X).to(device), torch.tensor(test.Y).to(device)

    #Create loaders 
    train_dataset = [(x, y) for (x, y) in zip(train.X, train.Y)]
    if ('others' not in args.dataset):
        val_dataset = [(x, y) for (x, y) in zip(val.X, val.Y)]
        for (x, y) in zip(val.X, val.Y):
            train_dataset.append((x,y))
    test_dataset = [(x, y) for (x, y) in zip(test.X, test.Y)]

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=False)

    # We do not evaluate the performance of the model on the 'roots' node (https://dtai.cs.kuleuven.be/clus/hmcdatasets/)
    if 'GO' in dataset_name: 
        num_to_skip = 4
    else:
        num_to_skip = 1 

    # Create the model
    if args.model == 'FC':
        model = AW_ConstrainedFFNNModel(input_dims[data], hidden_dim, output_dims[ontology][data]+num_to_skip, hyperparams, R)
    elif args.model == 'CNN':
        model = AW_CNN(R)
    else:
        print('AW debug: Wrong model name!')
        return
    
    model.to(device)
    print("Model on gpu", next(model.parameters()).is_cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    ############################## Visualization ##################################
    train_loss_list = []
    val_loss_list = []

    # train_acc_list = []
    # val_acc_list = []

    train_score_list = []
    val_score_list = []
    ################################################################

    for epoch in range(num_epochs):

        model.train()

        total_train = 0.0
        correct_train = 0.0

        train_score = 0
        total_train_loss = 0.

        for i, (x, labels) in enumerate(train_loader):

            print(f'AW debug Epoch {epoch} Training Batch {i}')
            
            if args.model == 'FC':
                x = x.view(x.size(0), -1)

            x = x.to(device)
            labels = labels.to(device)
        
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            # print('Main AW Debug:', x.shape)
            output = model(x.float())

            #MCLoss
            # print('AW_DEBUG R shape', R.shape)
            constr_output = get_constr_out(output, R)
            train_output = labels*output.double()
            train_output = get_constr_out(train_output, R)
            train_output = (1-labels)*constr_output.double() + labels*train_output

            # print('AW debug ------------------------------------------------------')
            # mask = torch.logical_or(constr_output < 0, constr_output >= 1)
            # print('AW debug constr_output', constr_output[mask])
            # print('AW debug constr_output', constr_output)

            # mask = torch.logical_or(train_output < 0, train_output >= 1)
            # print('AW debug train_output', train_output[mask])
            # print('AW debug ------------------------------------------------------')
            # print('AW debug train_output, labels', train_output, labels)
            # print('AW debug train_output.shape, labels.shape', train_output.shape, labels.shape)
            # print('AW debug ------------------------------------------------------')
            # print('AW debug train_output[:,train.to_eval], labels[:,train.to_eval]', train_output[:,train.to_eval], labels[:,train.to_eval])
            # print('AW debug train_output.shape[:,train.to_eval], labels.shape[:,train.to_eval]', train_output[:,train.to_eval].shape, labels[:,train.to_eval].shape)
            # print('AW debug train.to_eval', train.to_eval)
            # mask = torch.logical_or(train_output < 0, train_output >= 1)
            # print('AW debug', train_output[mask])
            # print('AW debug ------------------------------------------------------')
            loss = criterion(train_output[:,train.to_eval], labels[:,train.to_eval]) 

            ########################### Loss Viz #########################################
            print('AW debug Training loss: ', loss.item())
            total_train_loss += loss.item()

            # predicted = constr_output.data > 0.5

            # Total number of labels
            # total_train = labels.size(0) * labels.size(1)
            # Total correct predictions
            # correct_train = (predicted == labels.byte()).sum()

            # print('AW debug backward()', i)
            loss.backward()
            # print('AW debug optimizer.step()', i)
            optimizer.step()

        # model.eval()
        constr_output = constr_output.to('cpu')

        labels = labels.to('cpu')
        ##########  Viz ##############
        train_score = average_precision_score(labels[:, train.to_eval], constr_output.data[:, train.to_eval], average='micro')
        avg_train_loss = total_train_loss / len(train_loader)
        ##################################

        total_val_loss = 0.
        for i, (x,y) in enumerate(test_loader):

            print(f'AW debug Test Batch number {i}')
            if args.model == 'FC':
                x = x.view(x.size(0), -1)

            model.eval()
                    
            x = x.to(device)
            y = y.to(device)

            with torch.no_grad():
                # constrained_output = model(x.float())
                output = model(x.float())
                constr_output = get_constr_out(output, R)
                val_output = y*output.double()
                val_output = get_constr_out(val_output, R)
                val_output = (1-y)*constr_output.double() + y*val_output
                loss = criterion(val_output[:, train.to_eval], y[:, train.to_eval].double())
                total_val_loss += loss.item()
            # predicted = constrained_output.data > 0.5
            # Total number of labels
            # total = y.size(0) * y.size(1)
            # Total correct predictions
            # correct = (predicted == y.byte()).sum()

            #Move output and label back to cpu to be processed by sklearn
            # predicted = predicted.to('cpu')
            cpu_constrained_output = constr_output.to('cpu')
            y = y.to('cpu')

            if i == 0:
                # predicted_test = predicted
                constr_test = cpu_constrained_output
                y_test = y
            else:
                # predicted_test = torch.cat((predicted_test, predicted), dim=0)
                constr_test = torch.cat((constr_test, cpu_constrained_output), dim=0)
                y_test = torch.cat((y_test, y), dim =0)


        score = average_precision_score(y_test[:,test.to_eval], constr_test.data[:,test.to_eval], average='micro')
        #################### for val loss viz ##########################
        avg_val_loss = total_val_loss / len(test_loader)
        
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)

        train_score_list.append(train_score)
        val_score_list.append(score)
        ################################################################

    draw_loss_acc(args.model, data, train_loss_list, val_loss_list, 'Loss')
    draw_loss_acc(args.model, data, train_score_list, val_score_list, 'Score')

    f = open('results/'+dataset_name+'.csv', 'a')
    f.write(str(seed)+ ',' +str(epoch) + ',' + str(score) + '\n')
    f.close()

if __name__ == "__main__":
    main()