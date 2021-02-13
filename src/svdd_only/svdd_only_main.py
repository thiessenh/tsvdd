#  from tsvdd.libsvm import svmutil
import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
# from tsvdd.ga import tga_dissimilarity, train_kernel_matrix, test_kernel_matrix
#from tsvdd.utils import svmlib_kernel_format
#from sklearn.svm import SVDD
from tsvdd import newsvmutil
import subprocess


def numpy_to_precomputed(X, y, fil_name):
    with open(fil_name, 'w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            x = X[i]
            line = f'{y[i]} 0:{i+1}'
            for j in range(X.shape[1]):
                line += f' {j+1}:{x[j]}'
            line += '\n'
            f.write(line)


def numpy_to_file(X, y, fil_name):
    with open(fil_name, 'w', encoding='utf-8') as f:
        for i in range(X.shape[0]):
            x = X[i]
            line = f'{y[i]}'
            for j in range(X.shape[1]):
                line += f' {j+1}:{x[j]}'
            line += '\n'
            f.write(line)


def predict_to_numpy(fil_name):
    with open(fil_name, 'r', encoding='utf-8') as f:
        lines = f.read().splitlines()
    return np.array(lines, dtype=np.float64)




n_samples = 2000
outlier_ratio = 0.5
X, y = make_circles(n_samples=n_samples, factor=.3, noise=0.05)
y[y == 0] = -1


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
outlier_ratio_train = y_train[y_train == -1].shape[0] / (
            y_train[y_train == -1].shape[0] + y_train[y_train == 1].shape[0])
outlier_ratio_test = y_test[y_test == -1].shape[0] / (y_test[y_test == -1].shape[0] + y_test[y_test == 1].shape[0])
n_instances = X_train.shape[0]

fig, axs = plt.subplots(1, 2, figsize=(10, 10))
plt.title('Train split')
reds = y_test == -1
blues = y_test == 1

axs[0].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
axs[0].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
axs[0].set_title('Test split')
axs[0].set_aspect(aspect='equal')
reds = y_train == -1
blues = y_train == 1

axs[1].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
axs[1].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
axs[1].set_title('Train split')
axs[1].set_aspect(aspect='equal')
plt.show()


X_train_gram = np.dot(X_train, X_train.T)
X_train_gram_diag = np.diagonal(X_train_gram)
X_test_gram = np.dot(X_test, X_train.T)
X_test_gram_diag = np.diagonal(np.dot(X_test, X_test.T))
numpy_to_precomputed(X_train_gram, y_train, 'X_train.txt')
numpy_to_precomputed(X_test_gram, y_test, 'X_test.txt')
np.savetxt('X_train_diag.txt',X_train_gram_diag)
np.savetxt('X_test_diag.txt',X_test_gram_diag)

train_output = subprocess.run(["./svm-train", "-s", "5", "-t", "4", "-c", f"{1/(outlier_ratio_train*n_instances)}", "-e", f'{10e-7}', "X_train.txt"], capture_output=True)
out_train = train_output.stdout.decode("utf-8")
print(train_output.stderr.decode("utf-8"))
print(out_train)
lines = out_train.splitlines()
r_square = lines[2].split("=")[1]
r_square = float(r_square.strip())
subprocess.run(["./svm-predict", "X_train.txt", 'X_train.txt.model', "X_train.txt.output", "X_train_diag.txt"])
subprocess.run(["./svm-predict", "X_test.txt", 'X_train.txt.model', "X_test.txt.output", "X_test_diag.txt"])

y_pred_train = predict_to_numpy("X_train.txt.output")
y_pred_test = predict_to_numpy("X_test.txt.output")


fig, axs = plt.subplots(1, 2, figsize=(10, 10))
reds = y_pred_train == -1
blues = y_pred_train == 1


axs[1].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
axs[1].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
axs[1].set_title('Train split SVDD')
axs[1].set_aspect(aspect='equal')
circle = plt.Circle((0, 0), np.sqrt(r_square), color='b', fill=False)
axs[1].add_patch(circle)

reds = y_pred_test == -1
blues = y_pred_test == 1

axs[0].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
axs[0].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
axs[0].set_title('Test split SVDD')
axs[0].set_aspect(aspect='equal')
circle = plt.Circle((0, 0), np.sqrt(r_square), color='b', fill=False)
axs[0].add_patch(circle)

plt.show()


