import re

from glob import glob

def convert_logs_to_train_loss_dict(path):
    log_files = glob(path)
    train_losses_per_combination = {}

    for filename in log_files:
        matches = re.findall("\(\d+\)", filename)
        matches = map(lambda match: match.strip('(').strip(')'), matches)
        matches = list(matches)
        filter1 = matches[0]
        filter2 = matches[1]
        optimizer = matches[2]
        dropout = matches[3]

        train_losses = []
        with open(filename) as f:
           results = f.readlines()

           for line in results:
               train_entropy = re.findall("train_entropy \d\.\d+", line)
               train_entropy = re.findall("\d\.\d+", train_entropy[0])
               train_losses.append(float(train_entropy[0]))
        train_losses_per_combination["{}_{}_{}_{}".format(filter1, filter2, optimizer, dropout)] = train_losses

    return train_losses_per_combination

def convert_logs_to_test_accuracy_dict(path):
    log_files = glob(path)
    test_accuracies_per_combination = {}

    for filename in log_files:
        matches = re.findall("\(\d+\)", filename)
        matches = map(lambda match: match.strip('(').strip(')'), matches)
        matches = list(matches)
        filter1 = matches[0]
        filter2 = matches[1]
        optimizer = matches[2]
        dropout = matches[3]

        test_accuracies = []
        with open(filename) as f:
           results = f.readlines()

           for line in results:
               test_accuracy = re.findall("test_accuracy 0.\d+", line)
               test_accuracy = re.findall("0.\d+", test_accuracy[0])
               test_accuracies.append(float(test_accuracy[0]))
        test_accuracies_per_combination["{}_{}_{}_{}".format(filter1, filter2, optimizer, dropout)] = test_accuracies

    return test_accuracies_per_combination

if __name__ == '__main__':
    # log_files = 'epoch_200_initial_grid_search_train_logs/train_logs/*.txt'
    # log_files = 'epoch_2000_grid_search_biggest_step/*.txt'
    # log_files = 'epoch_2000_grid_search_second_step/*.txt'
    log_files = 'epoch_2000_grid_search_expanding_the_search/*.txt'
    train_losses_per_combination = convert_logs_to_test_accuracy_dict(log_files)

    # first get max test accuracy from each combination
    max_test_accuracies_per_combination = {}
    for combination, test_accuracies in train_losses_per_combination.items():
        max_test_accuracies_per_combination[combination] = max(test_accuracies)

    # then sort each dictionary by max test accuracy
    max_test_accuracies_per_combination_list = list(max_test_accuracies_per_combination.items())
    max_test_accuracies_per_combination_list.sort(key=lambda item: item[1], reverse=True)

    print(max_test_accuracies_per_combination_list)

    # TRUNCATED_EPOCH = 1500

    # max_test_accuracies_per_combination_truncated = list(max_test_accuracies_per_combination_truncated.items())
    # max_test_accuracies_per_combination_truncated.sort(key=lambda item: item[1], reverse=True)

    # # then get max value if truncated
    # max_test_accuracies_per_combination_truncated = {}
    # for combination, test_accuracies in test_accuracies_per_combination.items():
    #     max_test_accuracies_per_combination_truncated[combination] = max(test_accuracies[:TRUNCATED_EPOCH])

    # print(max_test_accuracies_per_combination_truncated[:5])
