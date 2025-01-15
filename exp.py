import numpy as np
from sklearn.model_selection import train_test_split
from cart import CART  # Importowanie klasy CART
from utils import load_data, extract_features, evaluate_model

# Wczytanie danych
donor_sequences, donor_labels = load_data('spliceATrainKIS.dat')
acceptor_sequences, acceptor_labels = load_data('spliceDTrainKIS.dat')

# Ekstrakcja cech z sekwencji
donor_features = extract_features(donor_sequences)
acceptor_features = extract_features(acceptor_sequences)

# Rozdzielamy dane na treningowe i testowe
X_donor_train, X_donor_test, y_donor_train, y_donor_test = train_test_split(donor_features, donor_labels, test_size=0.3, random_state=42)
X_acceptor_train, X_acceptor_test, y_acceptor_train, y_acceptor_test = train_test_split(acceptor_features, acceptor_labels, test_size=0.3, random_state=42)

# Tworzymy klasyfikator CART
cart_donor = CART(depth_limit=5, min_crit=0.1, criterion='gini')
cart_acceptor = CART(depth_limit=5, min_crit=0.1, criterion='gini')

# Trening klasyfikatora na danych donorów
cart_donor.fit(X_donor_train, y_donor_train)
print("\nDonor Data - Model Evaluation:")
evaluate_model(cart_donor, X_donor_test, y_donor_test)

# Trening klasyfikatora na danych akceptorów
cart_acceptor.fit(X_acceptor_train, y_acceptor_train)
print("\nAcceptor Data - Model Evaluation:")
evaluate_model(cart_acceptor, X_acceptor_test, y_acceptor_test)
