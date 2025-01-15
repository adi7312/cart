import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report

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

# Predykcja na zbiorze testowym dla donorów
y_pred_donor = clf.predict(X_test_donor)

# Ocena dokładności dla donorów
accuracy_donor = accuracy_score(y_test_donor, y_pred_donor)
print(f'Donor Data - Model Evaluation:')
print(f'Accuracy: {accuracy_donor * 100:.2f}%')

# Obliczenie macierzy pomyłek dla donorów
cm_donor = confusion_matrix(y_test_donor, y_pred_donor)
print("\nConfusion Matrix:")
print(cm_donor)

# Obliczenie precyzji, czułości i F1-score dla donorów
precision_donor = precision_score(y_test_donor, y_pred_donor)
recall_donor = recall_score(y_test_donor, y_pred_donor)
f1_donor = f1_score(y_test_donor, y_pred_donor)

print(f"\nPrecision: {precision_donor:.2f}")
print(f"Recall: {recall_donor:.2f}")
print(f"F1-score: {f1_donor:.2f}")

# Można także wyświetlić pełne podsumowanie za pomocą classification_report
print("\nClassification Report:")
print(classification_report(y_test_donor, y_pred_donor))


# Trening na danych z akceptorów
X_train_acceptor, X_test_acceptor, y_train_acceptor, y_test_acceptor = train_test_split(acceptor_features, acceptor_labels, test_size=0.3, random_state=42)
clf.fit(X_train_acceptor, y_train_acceptor)

# Predykcja na zbiorze testowym dla akceptorów
y_pred_acceptor = clf.predict(X_test_acceptor)

# Ocena dokładności dla akceptorów
accuracy_acceptor = accuracy_score(y_test_acceptor, y_pred_acceptor)
print(f'\nAcceptor Data - Model Evaluation:')
print(f'Accuracy: {accuracy_acceptor * 100:.2f}%')

# Obliczenie macierzy pomyłek dla akceptorów
cm_acceptor = confusion_matrix(y_test_acceptor, y_pred_acceptor)
print("\nConfusion Matrix:")
print(cm_acceptor)

# Obliczenie precyzji, czułości i F1-score dla akceptorów
precision_acceptor = precision_score(y_test_acceptor, y_pred_acceptor)
recall_acceptor = recall_score(y_test_acceptor, y_pred_acceptor)
f1_acceptor = f1_score(y_test_acceptor, y_pred_acceptor)

print(f"\nPrecision: {precision_acceptor:.2f}")
print(f"Recall: {recall_acceptor:.2f}")
print(f"F1-score: {f1_acceptor:.2f}")

# Można także wyświetlić pełne podsumowanie za pomocą classification_report
print("\nClassification Report:")
print(classification_report(y_test_acceptor, y_pred_acceptor))