import numpy as np

def one_hot_encode(sequence):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': -1}  # Dodanie 'N'
    encoded = np.zeros((len(sequence), 4))  # Macierz o wymiarach [długość_sekwencji x 4]
    for i, nucleotide in enumerate(sequence):
        index = mapping.get(nucleotide, -1)  # Obsługa nukleotydów spoza zestawu
        if index != -1:
            encoded[i, index] = 1
    return encoded.flatten()