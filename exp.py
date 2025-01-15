import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
from cart import CART

# Zakładając, że klasa CART jest zaimplementowana, jak podałeś:

# Funkcja do wczytania danych z plików
def load_data(file_path):
    labels = []
    sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):  # Zaczynamy od linii 2 (pomijamy pierwszą, która to liczba)
            label = int(lines[i].strip())  # Etykieta (0 lub 1)
            sequence = lines[i + 1].strip()  # Sekwencja DNA
            labels.append(label)
            sequences.append(sequence)
    return np.array(sequences), np.array(labels)

# Funkcja do ekstrakcji cech z sekwencji DNA
def extract_features(sequences, seq_length=90): 
    features = []
    for seq in sequences:
        feature = []
        for nucleotide in seq:
            if nucleotide == 'A':
                feature.extend([1, 0, 0, 0])  # Kodowanie A jako [1, 0, 0, 0]
            elif nucleotide == 'T':
                feature.extend([0, 1, 0, 0])  # Kodowanie T jako [0, 1, 0, 0]
            elif nucleotide == 'G':
                feature.extend([0, 0, 1, 0])  # Kodowanie G jako [0, 0, 1, 0]
            elif nucleotide == 'C':
                feature.extend([0, 0, 0, 1])  # Kodowanie C jako [0, 0, 0, 1]
        
        # Jeśli sekwencja jest krótsza niż zakładana długość, uzupełniamy ją zerami
        if len(feature) < seq_length * 4:
            feature.extend([0] * (seq_length * 4 - len(feature)))  # Wypełnienie zerami
        
        # Jeśli sekwencja jest dłuższa, obcinamy do wymaganej długości
        features.append(feature[:seq_length * 4])
    
    return np.array(features)

# Funkcja do oceny klasyfikatora
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\nPrecision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

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
