import numpy as np
from cart import CART
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from utils import load_data, extract_features, evaluate_model

def train_and_evaluate(data_file, experiment_type):
    sequences, labels = load_data(data_file)
    features = extract_features(sequences)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=None)

    cart = CART(depth_limit=5, min_crit=0.1, criterion='gini')
    cart.fit(X_train, y_train)

    print(f"\n{experiment_type} Data - Own Model Evaluation:")
    evaluate_model(cart, X_test, y_test)

def train_and_evaluate_model(data_file, experiment_type):
    sequences, labels = load_data(data_file)
    features = extract_features(sequences)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=None)

    clf = DecisionTreeClassifier(random_state=None)
    clf.fit(X_train, y_train)

    print(f'\n{experiment_type} Data - Imported Model Evaluation:')
    evaluate_model(clf, X_test, y_test)

if __name__ == '__main__':
    train_and_evaluate('spliceDTrainKIS.dat', 'Donor')
    train_and_evaluate('spliceATrainKIS.dat', 'Acceptor')
    train_and_evaluate_model('spliceATrainKIS.dat', 'Donor')
    train_and_evaluate_model('spliceDTrainKIS.dat', 'Acceptor')
