import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import math

plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == "cosine":
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def cal_sensitivity_specificity(y_pred, y_true, average='macro'):
    """
    Calculate both sensitivity (recall) and specificity for classification
    Handles both binary and multi-class cases automatically.
    
    Args:
        y_pred: predicted labels (1D array of class indices)
        y_true: true labels (1D array of class indices)
        average: 'macro' for unweighted mean, 'weighted' for weighted by support
    Returns:
        tuple: (sensitivity score, specificity score)
    """
    # Get unique classes and check if binary or multi-class
    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)
    is_binary = n_classes == 2
    
    if is_binary:
        # For binary case, calculate metrics directly
        positive_class = classes[1]  # conventionally the larger number is positive class
        
        # Calculate confusion matrix elements
        tp = np.sum((y_true == positive_class) & (y_pred == positive_class))
        tn = np.sum((y_true != positive_class) & (y_pred != positive_class))
        fp = np.sum((y_true != positive_class) & (y_pred == positive_class))
        fn = np.sum((y_true == positive_class) & (y_pred != positive_class))
        
        # Calculate metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return sensitivity, specificity
    
    else:
        # For multi-class case, calculate per-class metrics
        sensitivities = []
        specificities = []
        sensitivity_supports = []
        specificity_supports = []
        
        for c in classes:
            # True positives: predicted class c when true class is c
            tp = np.sum((y_true == c) & (y_pred == c))
            # False negatives: predicted not c when true class is c
            fn = np.sum((y_true == c) & (y_pred != c))
            # True negatives: predicted not c when true class is not c
            tn = np.sum((y_true != c) & (y_pred != c))
            # False positives: predicted c when true class is not c
            fp = np.sum((y_true != c) & (y_pred == c))
            
            # Sensitivity = TP / (TP + FN)
            if tp + fn > 0:
                sensitivity = tp / (tp + fn)
                sensitivities.append(sensitivity)
                sensitivity_supports.append(tp + fn)
            else:
                sensitivities.append(0)
                sensitivity_supports.append(0)
            
            # Specificity = TN / (TN + FP)
            if tn + fp > 0:
                specificity = tn / (tn + fp)
                specificities.append(specificity)
                specificity_supports.append(tn + fp)
            else:
                specificities.append(0)
                specificity_supports.append(0)
        
        # Calculate final scores based on averaging method
        if average == 'macro':
            final_sensitivity = np.mean(sensitivities)
            final_specificity = np.mean(specificities)
        elif average == 'weighted':
            final_sensitivity = np.average(sensitivities, weights=sensitivity_supports)
            final_specificity = np.average(specificities, weights=specificity_supports)
        else:
            final_sensitivity = np.array(sensitivities)
            final_specificity = np.array(specificities)
        
        return final_sensitivity, final_specificity