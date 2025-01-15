import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from utils import load_data, extract_features, evaluate_model

# Wczytanie danych
donor_sequences, donor_labels = load_data('spliceATrainKIS.dat')
acceptor_sequences, acceptor_labels = load_data('spliceDTrainKIS.dat')

# Ekstrakcja cech z sekwencji
donor_features = extract_features(donor_sequences)
acceptor_features = extract_features(acceptor_sequences)

# Łączenie danych
X = np.concatenate((donor_features, acceptor_features), axis=0)
y = np.concatenate((donor_labels, acceptor_labels), axis=0)

# Dzielimy dane na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tworzymy klasyfikator Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# Trening na danych z donorów
X_train_donor, X_test_donor, y_train_donor, y_test_donor = train_test_split(donor_features, donor_labels, test_size=0.3, random_state=42)
clf.fit(X_train_donor, y_train_donor)

# Ewaluacja modelu dla donorów
print(f'\nDonor Data - Model Evaluation:')
evaluate_model(clf, X_test_donor, y_test_donor)

# Trening na danych z akceptorów
X_train_acceptor, X_test_acceptor, y_train_acceptor, y_test_acceptor = train_test_split(acceptor_features, acceptor_labels, test_size=0.3, random_state=42)
clf.fit(X_train_acceptor, y_train_acceptor)

# Ewaluacja modelu dla akceptorów
print(f'\nAcceptor Data - Model Evaluation:')
evaluate_model(clf, X_test_acceptor, y_test_acceptor)
