



from keras.datasets import mnist


from data import Data

dataset = Data()

(x_t,y_t),(x,y) = dataset.getKerasDataset()
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train[0])
print("-------")
print(y_train[0])
print("-------")
print(X_test[0])
print("-------")
print(y_test[0])
print("-------")


print(x_t[0])
print("-------")
print(y_t[0])
print("-------")
print(x[0])
print("-------")
print(y[0])
print("-------")
