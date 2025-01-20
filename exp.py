import numpy as np
from cart import CART
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import load_data, extract_features, evaluate_model
from prettytable import PrettyTable

def train_and_evaluate(data_file, experiment_type, depth_limit, criterion, own_impl=True):
    sequences, labels = load_data(data_file)
    features = extract_features(sequences)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=None)
    if own_impl:
        classifier = CART(depth_limit=depth_limit, criterion=criterion)
        #print(classifier.depth_limit)
    else:
        classifier = DecisionTreeClassifier(random_state=None)

    classifier.fit(X_train, y_train)

    #print(f"\n{experiment_type} Data - Own Model Evaluation:")
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


def experiment(data_file, exp, depth_limit, criterion):
    print(f"{exp}: (depth_limit={depth_limit}, criterion={criterion})")
    results = []
    for _ in range(25):
        results.append(train_and_evaluate(data_file, exp, depth_limit=depth_limit, criterion=criterion).copy())
    return results
    

def max_depth_experiment(data_file, exp):
    test_suite = [3,5,10,15]
    for depth in test_suite:
        results = experiment(data_file, exp, depth, 'gini')
        process_experiment_results(results)

def criterion_experiment(data_file, exp):
    test_suite = ['gini', 'entropy']
    for criterion in test_suite:
        results = experiment(data_file, exp, 5, criterion)
        process_experiment_results(results)




if __name__ == '__main__':
    max_depth_experiment('spliceDTrainKIS.dat', 'Max Depth - Donors')
    max_depth_experiment('spliceATrainKIS.dat', 'Max Depth - Acceptors')
    criterion_experiment('spliceDTrainKIS.dat', 'Criterion - Donors')
    criterion_experiment('spliceATrainKIS.dat', 'Criterion - Acceptors')
