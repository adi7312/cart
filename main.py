import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from utils import load_data, extract_features, evaluate_model

def donor_experiment_model():
    donor_sequences, donor_labels = load_data('spliceATrainKIS.dat')
    donor_features = extract_features(donor_sequences)
    X_train_donor, X_test_donor, y_train_donor, y_test_donor = train_test_split(donor_features, donor_labels, test_size=0.3, random_state=None)

    donor_clf = DecisionTreeClassifier(random_state=None)
    donor_clf.fit(X_train_donor, y_train_donor)

    print(f'\nDonor Data - Model Evaluation:')
    evaluate_model(donor_clf, X_test_donor, y_test_donor)

def acceptor_experiment_model():
    acceptor_sequences, acceptor_labels = load_data('spliceDTrainKIS.dat')
    acceptor_features = extract_features(acceptor_sequences)
    X_train_acceptor, X_test_acceptor, y_train_acceptor, y_test_acceptor = train_test_split(acceptor_features, acceptor_labels, test_size=0.3, random_state=None)

    acceptor_clf = DecisionTreeClassifier(random_state=None)
    acceptor_clf.fit(X_train_acceptor, y_train_acceptor)

    print(f'\nAcceptor Data - Model Evaluation:')
    evaluate_model(acceptor_clf, X_test_acceptor, y_test_acceptor)

if __name__ == '__main__':
    donor_experiment_model()
    acceptor_experiment_model()