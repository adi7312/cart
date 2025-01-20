from sklearn.model_selection import train_test_split
from cart import CART
from utils import load_data, extract_features, evaluate_model



def donor_experiment():
    donor_sequences, donor_labels = load_data('spliceDTrainKIS.dat')
    donor_features = extract_features(donor_sequences)

    X_donor_train, X_donor_test, y_donor_train, y_donor_test = train_test_split(donor_features, donor_labels, test_size=0.3, random_state=None)

    cart_donor = CART(depth_limit=5, min_crit=0.1, criterion='gini')
    cart_donor.fit(X_donor_train, y_donor_train)
    print("\nDonor Data - Model Evaluation:")
    evaluate_model(cart_donor, X_donor_test, y_donor_test)


def acceptor_experiment():
    acceptor_sequences, acceptor_labels = load_data('spliceATrainKIS.dat')
    acceptor_features = extract_features(acceptor_sequences)

    X_acceptor_train, X_acceptor_test, y_acceptor_train, y_acceptor_test = train_test_split(acceptor_features, acceptor_labels, test_size=0.3, random_state=None)

    cart_acceptor = CART(depth_limit=5, min_crit=0.1, criterion='gini')
    cart_acceptor.fit(X_acceptor_train, y_acceptor_train)
    print("\nAcceptor Data - Model Evaluation:")
    evaluate_model(cart_acceptor, X_acceptor_test, y_acceptor_test)

if __name__ == '__main__':
    donor_experiment()
    acceptor_experiment()