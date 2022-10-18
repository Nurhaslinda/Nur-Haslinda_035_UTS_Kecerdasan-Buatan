#Nama   : Nur Haslinda
#NIM    : 21091397035
#Kelas  : 2021 A
#Multi neuron

#inisialisasi numpy
import numpy as np

#inisialisasi variabel dengan jumlah input 10
inputs = [3.0, 8.0, 2.0, 9.0, 4.0, 1.0, 7.0, 5.0, 6.0, 10.0]

#bobot per neuron
#panjang weights sesuai dengan panjang input, dan jumlah weights sesuai dengan jumlah neuron
weights = [[0.2, 0.4, 0.6, 0.8, 0.9, 0.2, 0.1, 0.3, 0.5, -0.4],
           [0.22, 0.24, 0.29, 0.2, 0.8, 0.3, 0.4, 0.99, 0.89, 0.49],
           [0.34, 0.6, 0.7, 0.23, 0.24, -0.29, -0.46, 0.78, 0.99, -0.1],
           [1.0, 0.29, 0.24, 0.3, 0.2, 4.0, 8.0, 0.75, 0.35, 0.22],
           [9.0, 0.3, 0.4, 0.2, -0.33, -0.44, 7.0, -0.1, 0.34, -0.56]]

#bias per neuron
#jumlah bias sesusai dengan jumlah neuron
biases = [9.0, 4.0, 2.0, 8.0, 3.0]

#output dengan menggunakan metode numpy
layer_outputs = np.dot(weights, inputs) + biases

#print output
print(layer_outputs)