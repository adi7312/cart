"""
UMA Semestr Ziomowy 2024/2025
Adrian Zalewski
Juliusz Kuzyka
exp.py - Eksperymenty
"""

import numpy as np
from cart import CART
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import load_data, extract_features, evaluate_model
from prettytable import PrettyTable
import matplotlib.pyplot as plt


def train_and_evaluate(data_file, experiment_type, depth_limit, criterion, own_impl=True):
    sequences, labels = load_data(data_file)
    features = extract_features(sequences)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=None)
    if own_impl:
        classifier = CART(depth_limit=depth_limit, criterion=criterion)
    else:
        classifier = DecisionTreeClassifier(random_state=None)
    classifier.fit(X_train, y_train)
    return evaluate_model(classifier, X_test, y_test)

def process_experiment_results(results: list, display=True):
    avg_accuracy = np.mean([result['accuracy'] for result in results])
    std_accuracy = np.std([result['accuracy'] for result in results])
    avg_recall = np.mean([result['recall'] for result in results])
    std_recall = np.std([result['recall'] for result in results])
    avg_precision = np.mean([result['precision'] for result in results])
    std_precision = np.std([result['precision'] for result in results])
    avg_f1 = np.mean([result['f1'] for result in results])
    std_f1 = np.std([result['f1'] for result in results])
    if display:
        t = PrettyTable(['Metric', 'Average', 'Standard Deviation'])
        t.add_row(['Accuracy', f'{avg_accuracy:.2f}', f'{std_accuracy:.2f}'])
        t.add_row(['Precision', f'{avg_precision:.2f}', f'{std_precision:.2f}'])
        t.add_row(['Recall', f'{avg_recall:.2f}', f'{std_recall:.2f}'])
        t.add_row(['F1', f'{avg_f1:.2f}', f'{std_f1:.2f}'])
        print(t)
    return {'avg_accuracy': avg_accuracy, 'std_accuracy': std_accuracy, 'avg_recall': avg_recall, 'std_recall': std_recall, 'avg_precision': avg_precision, 'std_precision': std_precision, 'avg_f1': avg_f1, 'std_f1': std_f1}


def experiment(data_file, exp, depth_limit, criterion, own_impl=True):
    print(f"{exp}: (depth_limit={depth_limit}, criterion={criterion})")
    results = []
    for _ in range(25):
        results.append(train_and_evaluate(data_file, exp, depth_limit=depth_limit, criterion=criterion, own_impl=own_impl).copy())
    return results
    

def max_depth_experiment(data_file, exp, own_impl=True):
    test_suite = [2,3,4,5,6,7,8]
    exp_results = {}
    for depth in test_suite:
        results = experiment(data_file, exp, depth, 'gini', own_impl=own_impl)
        r = process_experiment_results(results)
        exp_results[depth] = r
    return exp_results

def plot_max_depth_experiment(exp_results, exp_results2, metrics):
    depths1 = list(exp_results.keys())
    depths2 = list(exp_results2.keys())

    for metric in metrics:
        metrics1 = [exp_results[depth][metric] for depth in depths1]
        metrics2 = [exp_results2[depth][metric] for depth in depths2]

        plt.figure(figsize=(10, 6))
        plt.scatter(depths1, metrics1, marker='o', label='CART - Own implementation')
        plt.scatter(depths2, metrics2, marker='x', label='CART - sklearn implementation')
        plt.xlabel('Depth')
        plt.ylabel(metric.capitalize())
        plt.title(f'Max Depth Experiment Results - {metric.capitalize()}')
        plt.legend()
        plt.grid(True)
        plt.show()


def criterion_experiment(data_file, exp, depth, own_impl=True):
    test_suite = ['gini', 'entropy']
    for criterion in test_suite:
        results = experiment(data_file, exp, depth, criterion, own_impl=own_impl)
        process_experiment_results(results)
        




if __name__ == '__main__':

    #max_depth_exp_acceptor_own = max_depth_experiment('spliceATrainKIS.dat', 'Max Depth - Acceptors - Own Implementation', own_impl=True)
    #max_depth_exp_acceptor_sklearn = max_depth_experiment('spliceATrainKIS.dat', 'Max Depth - Acceptors - scikit implementation', own_impl=False)
    #plot_max_depth_experiment(max_depth_exp_acceptor_own,max_depth_exp_acceptor_sklearn, ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1'])

    max_depth_exp_donor_own = max_depth_experiment('spliceDTrainKIS.dat', 'Max Depth - Donors - Own Implementation', own_impl=True)
    max_depth_exp_donor_sklearn = max_depth_experiment('spliceDTrainKIS.dat', 'Max Depth - Donors - scikit implementation', own_impl=False)
    plot_max_depth_experiment(max_depth_exp_donor_own,max_depth_exp_donor_sklearn, ['avg_accuracy', 'avg_precision', 'avg_recall', 'avg_f1'])


    criterion_exp_donor = criterion_experiment('spliceDTrainKIS.dat', 'Criterion - Donors', 6)
    criterion_exp_acceptor = criterion_experiment('spliceATrainKIS.dat', 'Criterion - Acceptors', 6)
