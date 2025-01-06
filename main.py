from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import tree as sktree
from cart import CART

def classification_example():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)
    cls = CART()
    print(X_train)
    print(y_train)
    cls.fit(X_train, y_train)
    cls.print_tree()
    
    

classification_example()