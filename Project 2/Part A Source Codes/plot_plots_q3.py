import pylab as plt
import numpy as np
from process_training_logs import convert_logs_to_test_accuracy_dict, convert_logs_to_train_loss_dict

if __name__ == '__main__':

    log_files = 'different_optimizers_and_dropout_comparison/*.txt'
    test_accuracies_per_combination = convert_logs_to_test_accuracy_dict(log_files)
    train_losses_per_combination = convert_logs_to_train_loss_dict(log_files)

    colors = [  'b',\
                'g',\
                'r',\
                'c',\
                'm',\
                'y',\
                'k',\
                'w']
    plt.figure(1)
    max_loss = -1
    for i, combination in enumerate(train_losses_per_combination.keys()):
        _, _, optimizer, dropout = combination.split('_')
        optimizer = int(optimizer)
        dropout = float(dropout)
        legend = "default"
        if (dropout < 1):
            legend = "Dropout with Adam Optimizer"
        if (optimizer == 1):
            legend = "Momentum"
        if (optimizer == 2):
            legend = "RMSProp"
        if (optimizer == 3):
            legend = "Adam Optimizer"

        if (max(train_losses_per_combination[combination]) > max_loss):
            max_loss = max(train_losses_per_combination[combination])
        plt.plot(range(2000),\
                 train_losses_per_combination[combination],\
                 colors[i % len(train_losses_per_combination)],\
                 label=legend)
    plt.xlabel('Epochs')
    plt.ylabel('Training Costs')
    plt.legend(loc='upper right')
    plt.title('Training Costs vs. Iterations')

    plt.figure(2)
    for i, combination in enumerate(test_accuracies_per_combination.keys()):
        _, _, optimizer, dropout = combination.split('_')
        optimizer = int(optimizer)
        dropout = float(dropout)
        legend = "Gradient Descent"
        if (dropout < 1):
            legend = "Dropout with Adam Optimizer"
        if (optimizer == 1):
            legend = "Momentum"
        if (optimizer == 2):
            legend = "RMSProp"
        if (optimizer == 3):
            legend = "Adam Optimizer"

        plt.plot(range(2000),\
                 test_accuracies_per_combination[combination],\
                 colors[i % len(test_accuracies_per_combination)],\
                 label=legend)
    plt.xlabel('Epochs')
    plt.ylabel('Testing Accuracy')
    plt.legend(loc='lower right')
    plt.title('Testing Accuracy vs. Iterations')

    plt.show()
