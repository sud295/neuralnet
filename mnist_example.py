import tensorflow as tf
from network import *
import random

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

assigment_map = {0:[1,0,0,0,0,0,0,0,0,0],1:[0,1,0,0,0,0,0,0,0,0],
                 2:[0,0,1,0,0,0,0,0,0,0],3:[0,0,0,1,0,0,0,0,0,0],
                 4:[0,0,0,0,1,0,0,0,0,0],5:[0,0,0,0,0,1,0,0,0,0],
                 6:[0,0,0,0,0,0,1,0,0,0],7:[0,0,0,0,0,0,0,1,0,0],
                 8:[0,0,0,0,0,0,0,0,1,0],9:[0,0,0,0,0,0,0,0,0,1]}

net = Network()

l1 = Layer()
l2 = Layer(activation_fcn="relu")
l3 = Layer(activation_fcn="softmax")

l1.fill_layer(784,"input")
l2.fill_layer(16,"cong")
l3.fill_layer(10,"cong")

net.layers.append(l1)
net.layers.append(l2)
net.layers.append(l3)

net.connect_layers()
net.add_celoss_vertex()

arr = list(range(len(x_train)))

for epoch in range(5):
    random.shuffle(arr)
    for i in range(len(x_train)):
        x = x_train[arr[i]].flatten()
        y = y_train[arr[i]]
        
        l1.set_input(x)
        net.set_true_out(assigment_map.get(y))
        
        net.forward_pass()
        net.backward_pass()
        net.reset_gradients()

        # Batch size of 100
        if i % 1000 == 0:
            net.update_weights()
            print(f"Epoch: {epoch + 1}, Step: {i}, Loss: {net.err}, \
                  Output: {net.get_output()}, True: {y}")

net.forward_pass()
print("Final Output:",net.get_output(),"Loss:",net.err)
