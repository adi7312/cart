import numpy as np
from sklearn.model_selection import train_test_split
from cart import CART
from sklearn.preprocessing import OneHotEncoder
from utils import one_hot_encode




def donor_training():
   with open("spliceATrainKIS.dat", "r") as file:
    lines = file.readlines()
    raw_data = lines[1:] 

    labels = [int(raw_data[i].strip()) for i in range(0, len(raw_data), 2)]
    sequences = [raw_data[i + 1].strip() for i in range(0, len(raw_data), 2)]
    

    X = np.array([one_hot_encode(seq) for seq in sequences])
    y = np.array(labels)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    cls = CART()
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    print(f"Accuracy: {accuracy}")
    #cls.print_tree()

def acceptor_training():
    with open("spliceATrainKIS.dat", "r") as file:
        lines = file.readlines()
        raw_data = lines[1:] 

        labels = [int(raw_data[i].strip()) for i in range(0, len(raw_data), 2)]
        sequences = [raw_data[i + 1].strip() for i in range(0, len(raw_data), 2)]
        

        X = np.array([one_hot_encode(seq) for seq in sequences])
        y = np.array(labels)


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        cls = CART()
        cls.fit(X_train, y_train)
        y_pred = cls.predict(X_test)
        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy}")
        #cls.print_tree()


donor_training()
acceptor_training()