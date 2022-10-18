#Nama   : Nur Haslinda
#NIM    : 21091397035
#Kelas  : 2021 A
#single neuron

#inisialisasi numpy
import numpy as np

#inisialisasi variabel dengan jumlah input 10
inputs = [3, 8, 2, 9, 4, 1, 7, 5, 6, 10]

#bobot per neuron
#panjang weights sesuai dengan panjang input, dan jumlah weights sesuai dengan jumlah neuron
weights = [0.2, 0.4, 0.6, 0.8, 0.9, 0.2, 0.1, 0.3, 0.5, -0.4]

#bias per neuron
#jumlah bias sesusai dengan jumlah neuron
bias = 9

#output dengan menggunakan metode numpy
output = np.dot(weights, inputs) + bias

#print output
print(output)