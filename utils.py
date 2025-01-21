"""
UMA Semestr Ziomowy 2024/2025
Adrian Zalewski
Juliusz Kuzyka
utils.py - Funkcje pomocnicze
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report


def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1} 
    encoded = np.zeros((len(sequence), 4)) 
    for i, nucleotide in enumerate(sequence):
        index = mapping.get(nucleotide, -1)
        if index != -1:
            encoded[i, index] = 1
    return encoded.flatten()


def load_data(file_path):
    labels = []
    sequences = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for i in range(1, len(lines), 2):
            label = int(lines[i].strip())
            sequence = lines[i + 1].strip()
            labels.append(label)
            sequences.append(sequence)
    return np.array(sequences), np.array(labels)

def extract_features(sequences, seq_length=91):
    features = []
    for seq in sequences:
        feature = one_hot_encode(seq)
        if len(feature) < seq_length * 4:
            feature = np.pad(feature, (0, seq_length * 4 - len(feature)), mode='constant')
        features.append(feature[:seq_length * 4])
    return np.array(features)


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
