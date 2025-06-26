import numpy as np
from PIL import Image
from natsort import natsorted
import os
import matplotlib.pyplot as plt  # ✅ added matplotlib

def process(path):
    im = Image.open(path).convert('L')
    re = im.resize((28, 28))
    a = np.array(re)
    k = a.flatten() / 255.0
    return k.reshape(1, 784)

def load(folder):
    images = []
    files = natsorted(os.listdir(folder))
    for i in files:
        if os.path.isfile(os.path.join(folder, i)) and i.lower().endswith(('.png', '.jpg')):
            pat = os.path.join(folder, i)
            l = process(pat)
            images.append(l)
    return np.array(images)

def sigmoid(num):
    return 1 / (1 + np.exp(-num))

def sigmoid_derivative(k):
    return k * (1 - k)

def softmax(z):
    z_exp = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return z_exp / np.sum(z_exp, axis=-1, keepdims=True)

def cross_entropy_loss(predictions, targets):
    # Small value added to prevent log(0)
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    return -np.sum(targets * np.log(predictions))

# Load or initialize weights
if os.path.exists("W1.npy"):
    W1 = np.load("W1.npy")
    b1 = np.load("b1.npy")
    W2 = np.load("W2.npy")  
    b2 = np.load("b2.npy")
else:
    W1 = np.random.randn(784, 64) * np.sqrt(1 / 784)
    b1 = np.zeros((1, 64))
    W2 = np.random.randn(64, 10) * np.sqrt(1 / 64)
    b2 = np.zeros((1, 10))
    print("New weights and biases initialized")

# One-hot encoding for labels
y = []
for i in range(10): 
    j = np.zeros((10,), dtype=int)
    j[i] = 1
    y.append(j)

# Load dataset
p = "/Users/yaswanthreddy/Desktop/idk/untitled folder/folder/digits updated/2"
k = load(p).reshape(-1, 784)
learning_rate = 0.01
epochs = 100
losses = []

# Uncomment this block to train:
# for i in range(epochs):
#     w = 0 
#     n = 0  
#     total_loss = 0
#     for j in range(len(k)):
#         if n >= 100:
#             n = 0
#             w += 1
#             if w >= 10:
#                 break
#         l = k[j].reshape(1, 784)
#         h1 = np.dot(l, W1) + b1
#         h2 = sigmoid(h1)
#         h3 = np.dot(h2, W2) + b2
#         h4 = softmax(h3)
#         loss = cross_entropy_loss(h4, y[w])
#         total_loss += loss
#         h5 = h4 - y[w]
#         h6 = np.dot(h5, W2.T)
#         h7 = h6 * sigmoid_derivative(h2)
#         W2 -= np.dot(h2.T, h5) * learning_rate
#         W1 -= np.dot(l.T, h7) * learning_rate
#         b2 -= np.sum(h5, axis=0, keepdims=True) * learning_rate
#         b1 -= np.sum(h7, axis=0, keepdims=True) * learning_rate
#         n += 1
#     avg_loss = total_loss / len(k)
#     losses.append(avg_loss)
#     if i % 10 == 0:
#         print(f"Epoch {i} - Loss: {avg_loss:.4f}")

# # Save model weights
# np.save("W1.npy", W1)
# np.save("W2.npy", W2)
# np.save("b1.npy", b1)
# np.save("b2.npy", b2)

# # 📊 Plot the loss graph
# plt.plot(range(epochs), losses)
# plt.title("Training Loss Over Epochs")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.savefig("training_loss.png")  # Save as image
# plt.show()
path=""
while(path!="q"):
    print("Path:")
    def predict(pat):
        im = Image.open(pat).convert('L')
        re = im.resize((28, 28))
        re.show()
        a = np.array(re)
        k = a.flatten() / 255.0
        k1=k.reshape(1, 784)
        h11 = np.dot(k1, W1) + b1
        h22 = sigmoid(h11)
        h33 = np.dot(h22, W2) + b2
        h44 = softmax(h33)
        label = np.argmax(h44)
        return h44, label
    path = str(input())
    if(path!="q"):
        prob, label = predict(path)
        
        print(f" probab: {prob}")
        print(f"label: {label}")
