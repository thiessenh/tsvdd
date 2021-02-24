import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
from tsvdd.SVDD import SVDD
from sklearn.metrics import balanced_accuracy_score

N_SAMPLES = 1000
outlier_ratio = 0.5
X1, y1 = make_circles(n_samples=N_SAMPLES, factor=.1, noise=0.1)
y1[y1 == 0] = 1

# X2, y2 = make_circles(n_samples=n_samples, factor=.6, noise=0.25)
# y2[y2 == 0] = -1

# X = np.concatenate((X1, X2), axis=0)
# y = np.concatenate((y1, y2))
X = X1; y = y1;
# del X1, X2, y1, y2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
outlier_ratio_train = y_train[y_train == -1].shape[0] / (
            y_train[y_train == -1].shape[0] + y_train[y_train == 1].shape[0])
outlier_ratio_test = y_test[y_test == -1].shape[0] / (y_test[y_test == -1].shape[0] + y_test[y_test == 1].shape[0])
n_instances = X_train.shape[0]

y_train[np.argmax(X_train, axis=0)] = -1
y_test[np.argmax(X_test, axis=0)] = -1

fig, axs = plt.subplots(1, 2, figsize=(10, 6))
fig.suptitle('Active learning Data', fontsize=12)
reds = y_train == -1
blues = y_train == 1

axs[0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
axs[0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
axs[0].set_title('Train split')
axs[0].set_aspect(aspect='equal')

reds = y_test == -1
blues = y_test == 1

# circle = plt.Circle((0, 0), 0.5, color='b', fill=False)
# axs[1].add_patch(circle)
axs[1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
axs[1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
axs[1].set_title('Test split')
axs[1].set_aspect(aspect='equal')

plt.show()



gram_train = linear_kernel(X_train, X_train)
gram_test = linear_kernel(X_test, X_train)
gram_diagonal_test = np.diagonal(linear_kernel(X_test, X_test))
gram_diagonal_train = np.diagonal(gram_train)

def meshgrid(X, resolution=201):
    x_max = np.max(X[:, 0])
    x_min = np.min(X[:, 0])
    y_max = np.max(X[:, 1])
    y_min = np.min(X[:, 1])


    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    XY = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, XY

xx, yy, XY = meshgrid(X)



gram_Z = np.dot(XY, X_train.T)
gram_z_diag = np.apply_along_axis(lambda xi : np.dot(xi, xi.T), 1, XY)


# start with slighly too high initial outlier estimation
# SVDD should learn decision boundary that is too tight, thus, excluding inliers

C = 1/(0.5*n_instances)
svdd = SVDD(kernel='precomputed', C=C,verbose=True, tol=10e-6, shrinking=False)

# initally sample weights are all 1
W = np.ones(X_train.shape[0], dtype=np.float64)

# conture plot for visualization
svdd.fit(gram_train, W=W)
y_pred_train = svdd.predict(gram_train, gram_diagonal_train)
y_pred_test = svdd.predict(gram_test, gram_diagonal_test)

y_pred_train_val = svdd.decision_function(gram_train, gram_diagonal_train)
y_pred_test_val = svdd.decision_function(gram_test, gram_diagonal_test)

plot_train_test_pred(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, np.sqrt(svdd.r_square))

Z = svdd.decision_function(gram_Z, gram_z_diag)
Z = Z.reshape(xx.shape)
decision_boundary(X_train, y_train,xx, yy, Z, 'Decision boundary Train y_train')
decision_boundary(X_train, y_pred_train, xx, yy, Z, 'Decision boundary Train y_pred')
decision_boundary(X_test, y_test, xx, yy, Z, 'Decision boundary Test y_test')
decision_boundary(X_test, y_pred_test, xx, yy, Z, 'Decision boundary Test y_pred')

U = list(range(X_train.shape[0]))
L_in = list()
L_out = list()

v_in = 10.0
v_out = 0.01

bal_accs = list()
radii = list()
# Active Learning Loop
for i in range(50):
    print(f'Iteration: {i}')
    svdd = SVDD(kernel='precomputed', C=C, verbose=True, tol=10e-6, shrinking=False)
    # train / retrain
    svdd.fit(gram_train, W=W)
    print(f'R = {np.sqrt(svdd.r_square)}')
    radii.append(np.sqrt(svdd.r_square))
    y_pred = svdd.predict(gram_train, gram_diagonal_train)
    y_vals = svdd.decision_function(gram_train, gram_diagonal_train)
    bal_acc = balanced_accuracy_score(y_train, y_pred)
    bal_accs.append(bal_acc)
    # get most informative unknown sample outside decision boundary
    idx_in_U = np.abs(np.where(y_vals[U] <= 0, np.inf, y_vals[U])).argmin()
    idx = U[idx_in_U]
    print(f'Most informative sample is {idx} with decision value {y_vals[idx]}')
    # ask oracle for annotation
    annotation = y_train[idx]
    print(f'Annotation is {annotation} for most informative sample {idx}')
    # remove sample from U
    U.remove(idx)
    # adjust sample weight
    if annotation == 1:
        W[idx] = v_in
        L_in.append(idx)
    else:
        W[idx] = v_out
        L_out.append(idx)
    print(f'New sample weight W[{idx}]: {W[idx]}')
    print('')

plt.plot(list(range(50)), bal_accs)
plt.plot(list(range(50)), radii)


Z = svdd.decision_function(gram_Z, gram_z_diag)
Z = Z.reshape(xx.shape)

y_pred_train = svdd.predict(gram_train, gram_diagonal_train)
y_pred_train = np.array(y_pred_train)

decision_boundary(X_train, y_train, xx, yy, Z, 'Active learning decision boundary Train y_train', (U, L_in, L_out))
decision_boundary(X_train, y_pred_train, xx, yy, Z, 'Active learning decision boundary Train y_pred', (U, L_in, L_out))

print("Annotated inliers:")
print(L_in)
print("Annotated outliers:")
print(L_out)


1

