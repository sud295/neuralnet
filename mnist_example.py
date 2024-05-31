import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data
x_train, x_test = x_train / 255.0, x_test / 255.0

assigment_map = {0:[1,0,0,0,0,0,0,0,0,0], 1:[0,1,0,0,0,0,0,0,0,0],2:[0,0,1,0,0,0,0,0,0,0],3:[0,0,0,1,0,0,0,0,0,0],4:[0,0,0,0,1,0,0,0,0,0],5:[0,0,0,0,0,1,0,0,0,0],6:[0,0,0,0,0,0,1,0,0,0],7:[0,0,0,0,0,0,0,1,0,0],8:[0,0,0,0,0,0,0,0,1,0],9:[0,0,0,0,0,0,0,0,0,1]}

from network import *

net = Network()

l1 = Layer()
l2 = Layer(activation_fcn="sigmoid")
l3 = Layer(activation_fcn="sigmoid")
l4 = Layer(activation_fcn="sigmoid")
l1.fill_layer(784,"input")
l2.fill_layer(128,"cong")
l3.fill_layer(64,"cong")
l4.fill_layer(10,"cong")

l1.set_input(list(x_train[0].flatten()))

net.layers.append(l1)
net.layers.append(l2)
net.layers.append(l3)
net.layers.append(l4)

net.connect_layers()
net.add_mserror_vertex()
net.set_true_out(assigment_map.get(5))

net.forward_pass()

i=0
for j in range(1000):
    net.forward_pass()
    print("Prediction:",net.get_output(),"Loss:",net.err,"Iter:",j+1)
    net.backward_pass()
    net.update_weights()
    net.reset_gradients()
    i+=1
    ind = np.random.randint(1, 60000)
    ind-=1
    x = x_train[ind].flatten()
    y = y_train[ind]
    l1.set_input(x)
    net.set_true_out(assigment_map.get(y))


print()
net.forward_pass()
print("Final Output:",net.get_output(),"Loss:",net.err,"Iterations:",i)
