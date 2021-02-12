import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
import numpy as np
from tsvdd.utils import svmlib_kernel_format
#from tsvdd import SVDD
from sklearn.svm import SVDD
from sklearn.metrics import balanced_accuracy_score


def plot_train_test_pred(X_train, X_test, y_train, y_test, y_pred_train=None, y_pred_test=None):
    if y_pred_test is not None and y_pred_train is not None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    else:
        fig, axs = plt.subplots(1, 2, figsize=(12, 12), dpi=600)
    fig.suptitle('Active learning Data with random cirlce', fontsize=12)
    reds = y_train == -1
    blues = y_train == 1

    axs[0,0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
    axs[0,0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0,0].set_title('Train split')
    axs[0,0].set_aspect(aspect='equal')

    reds = y_test == -1
    blues = y_test == 1

    axs[0,1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
    axs[0,1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0,1].set_title('Test split')
    axs[0,1].set_aspect(aspect='equal')

    if y_pred_test is not None and y_pred_train is not None:
        reds = y_pred_train == -1
        blues = y_pred_train == 1

        axs[1,0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
        axs[1,0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
        axs[1,0].set_title('Train split alSVDD')
        axs[1,0].set_aspect(aspect='equal')

        reds = y_pred_test == -1
        blues = y_pred_test == 1

        # if r_square:
        #     circle = plt.Circle((0, 0), r_square, color='b', fill=False)
        #     axs[1,1].add_patch(circle)
        axs[1,1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
        axs[1,1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
        axs[1,1].set_title('Test split alSVDD')
        axs[1,1].set_aspect(aspect='equal')

    plt.show()

def decision_boundary_train_test(X_train, X_test, y_pred_train, y_pred_test, xx, yy, Z):
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle('Active learning Data with random cirlce', fontsize=12)

    axs[0].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs[0].contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs[0].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    axs[1].contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs[1].contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs[1].contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    reds = y_pred_train == -1
    blues = y_pred_train == 1

    axs[0].scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
    axs[0].scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
    axs[0].set_title('Train split')
    axs[0].set_aspect(aspect='equal')

    reds = y_pred_test == -1
    blues = y_pred_test == 1

    axs[1].scatter(X_test[reds, 0], X_test[reds, 1], c="red", s=20, edgecolor='k')
    axs[1].scatter(X_test[blues, 0], X_test[blues, 1], c="blue", s=20, edgecolor='k')
    axs[1].set_title('Test split')
    axs[1].set_aspect(aspect='equal')

    plt.show()

def decision_boundary(X_train, y_pred_train, xx, yy, Z, title=None):
    fig, axs = plt.subplots(1, 1, figsize=(10, 6))
    if title:
        fig.suptitle(f'{title}', fontsize=12)
    else:
        fig.suptitle('Active learning Data decision boundary', fontsize=12)

    axs.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7),
                    cmap=plt.cm.PuBu, zorder=-99)
    axs.contourf(xx, yy, Z, levels=[0, Z.max()], colors='palevioletred',
                    zorder=-98)
    axs.contour(xx, yy, Z, levels=[0], linewidths=2, colors='darkred',
                   zorder=-97)

    reds = y_pred_train == -1
    blues = y_pred_train == 1

    axs.scatter(X_train[reds, 0], X_train[reds, 1], c="red", s=20, edgecolor='k')
    axs.scatter(X_train[blues, 0], X_train[blues, 1], c="blue", s=20, edgecolor='k')
    axs.set_title('Train split')
    axs.set_aspect(aspect='equal')

    plt.show()


n_samples = 1000
outlier_ratio = 0.5
X1, y1 = make_circles(n_samples=n_samples, factor=.6, noise=0.25)
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



gram_Z = linear_kernel(XY, X_train)
gram_z_diag = np.apply_along_axis(lambda xi : linear_kernel(xi.reshape(1, -1), xi.reshape(1, -1)), 1, XY)


# start with slighly too high initial outlier estimation
# SVDD should learn decision boundary that is too tight, thus, excluding inliers
nu = 0.5
svdd = SVDD(kernel='linear', nu=nu, tol=10e-5, verbose=True)

# initally sample weights are all 1
W = np.ones(X_train.shape[0], dtype=np.float64)

# conture plot for visualization
svdd.fit(X_train, sample_weight=W)
y_pred = svdd.predict(X_train)
Z = svdd._decision_function(XY)
Z = Z.reshape(xx.shape)
decision_boundary(X_train, y_train, xx, yy, Z, 'Active learning decision boundary Train y_train')
decision_boundary(X_train, y_pred, xx, yy, Z, 'Active learning decision boundary Train y_pred')


U = list(range(X_train.shape[0]))
L_in = list()
L_out = list()

v_in = 1000.0
v_out = 0.0001

bal_accs = list()
radii = list()
# Active Learning Loop
for i in range(50):
    print(f'Iteration: {i}')
    #print(f'R^2 = {svdd.r_square}')
    y_pred = svdd.predict(X_train)
    y_vals = svdd._decision_function(X_train)
    bal_acc = balanced_accuracy_score(y_train, y_pred)
    bal_accs.append(bal_acc)
    # get most informative unknown sample outside decision boundary
    idx_in_U = np.abs(np.where(y_vals[U] <= 0, y_vals[U], np.inf)).argmin()
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
    # retrain svdd
    svdd.fit(X_train, sample_weight=W)
    # radii.append(svdd.r_square)
    print('')

plt.plot(list(range(50)), bal_accs)

#plt.plot(list(range(50)), radii)

gram_Z = linear_kernel(XY, X_train)
gram_z_diag = np.apply_along_axis(lambda xi : linear_kernel(xi.reshape(1, -1), xi.reshape(1, -1)), 1, XY)

Z = svdd._decision_function(XY)
Z = Z.reshape(xx.shape)


y_pred_train =svdd.predict(X_train)
#y_pred_test = svdd.predict(gram_test, gram_diagonal_test)
#y_pred_test = np.array(y_pred_test)
y_pred_train = np.array(y_pred_train)

decision_boundary(X_train, y_train, xx, yy, Z, 'Active learning decision boundary Train y_train')
decision_boundary(X_train, y_pred_train, xx, yy, Z, 'Active learning decision boundary Train y_pred')

print("Annotated inliers:")
print(L_in)
print("Annotated outliers:")
print(L_out)


#plot_train_test_pred(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test)

1


# n_train = 30
# length_train = 10
# dim = 1
#
# rs = np.random.RandomState(1234)
# c_train_matrix = rs.rand(n_train, length_train, dim).astype(dtype=np.float64, order='c')
# res = np.zeros(shape=(n_train, n_train))
# sigma = 2
# triangular = 0
# for i in range(n_train):
#     seq_1 = c_train_matrix[i]
#     for j in range(n_train):
#         seq_2 = c_train_matrix[j]
#         res[i, j] = tga_dissimilarity(seq_1, seq_2, sigma, triangular)
# res = np.exp(-res)
# res_cython = train_kernel_matrix(c_train_matrix, sigma, triangular, 'exp')
# np.testing.assert_array_equal(res, res_cython)
#
# print(model.get_r2())
# print(model.get_rho())
