from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
def read_MNIST():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000,784)
    x_test = x_test.reshape(10000,784)
    sc_X = StandardScaler()
    x_train = sc_X.fit_transform(x_train)
    x_test = sc_X.transform(x_test)
    return ((x_train, y_train), (x_test, y_test))
if __name__ == "__main__":
    read_MNIST()



