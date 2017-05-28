
# coding: utf-8

# In[25]:

import numpy as np
import pandas as pd
import pylab as plt
get_ipython().magic('matplotlib inline')
from sklearn.preprocessing import StandardScaler as scaler
from sklearn.model_selection import train_test_split


# In[26]:

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import log_loss
data = load_breast_cancer()
target = data.target
data = pd.DataFrame(data.data)
target = np.array(np.where(target != 0, target, -1))
data = data.fillna(data.median(axis=0), axis=0)
scale = scaler()
data = pd.DataFrame(scale.fit_transform(data), columns=data.columns)
X_train, X_test, y_train, y_test = train_test_split(data, target, stratify=target, test_size=0.1, random_state=42)


# In[27]:

def gradient_descent(n, w, h, C, inner_data, inner_target):
    w_cur = w
    iterations = 0
    log_losses = []
    while iterations < 10000:
        c = 1 - 1/(1 + np.exp(-np.multiply(np.sum(np.multiply(inner_data, w_cur), axis=1), inner_target)))
        e = np.multiply(c, inner_target)
        inner_gd_answer = np.vstack((1/(1 + np.exp(-np.sum(np.multiply(inner_data, w_cur), axis=1)*(-1))), 1/(1 + np.exp(-np.sum(np.multiply(inner_data, w_cur), axis=1))))).T
        w_new = np.sum(np.transpose(np.multiply(np.transpose(inner_data), e)), axis = 0)*h/n + w_cur*(1 - h*C)
        if np.abs(np.sqrt(np.sum(np.power(w_new - w_cur, 2)))) < 0.000001:
            break
        else:
            w_cur = w_new
            iterations += 1
            log_losses.append(log_loss(inner_target, inner_gd_answer))
            
    return (w_cur, log_losses, iterations)


# In[28]:

X_quarter, X_three_quaters, y_quarter, y_three_quaters = train_test_split(X_train, y_train, stratify=y_train, test_size=0.25, random_state=42)
X_half, X_dummy, y_half, y_dummy = train_test_split(X_train, y_train, stratify=y_train, test_size=0.5, random_state=42)


# In[29]:

from copy import deepcopy
log_losses_best = 100000000
h_best = 0
C_best = 0
w_best = np.zeros(data.shape[1])
iterations_dict = dict()
log_losses_dict = dict()
log_test_dict = dict()
log_losses_best_set = [0, 0, 0, 0]
w_best_set = [0, 0, 0, 0]
log_test_best_set = [0, 0, 0, 0]
log_iterations_best_set = [0, 0, 0, 0]
index = 1
for X, y in zip([X_quarter, X_half, X_three_quaters, X_train], [y_quarter, y_half, y_three_quaters, y_train]):
    iterations_set = []
    log_losses_set = []
    log_losses_test = []
    for C in [0, 1, 0.1, 0.01]:
        for h in [1, 0.1, 0.01, 0.001, 0.0001]:
            w, log_losses, iter_count = gradient_descent(len(y), np.zeros(data.shape[1]), h, C, np.array(X), np.array(y))
            test_answers = np.vstack((1/(1 + np.exp(-np.sum(np.multiply(X_test, w), axis=1)*(-1))), 1/(1 + np.exp(-np.sum(np.multiply(X_test, w), axis=1))))).T
            log_losses_test.append(log_loss(y_test, test_answers))
            log_losses_set.append(log_losses)
            iterations_set.append(iter_count)
            if log_losses[-1] < log_losses_best:
                log_losses_best = log_losses[-1]
                log_losses_best_set[index] = log_losses
                h_best = h
                C_best = C
                w_best = w
                w_best_set[index] = w
                log_test_best_set[index] = log_loss(y_test, test_answers)
                log_iterations_best_set[index] = iter_count
    iterations_dict[index*0.25] = deepcopy(iterations_set)
    log_losses_dict[index*0.25] = deepcopy(log_losses_set)
    log_test_dict[index*0.25] = deepcopy(log_losses_test)
    index += 1


# Построим графики качества классификации в зависимости от разного набора данных

# In[30]:

log_losses_best = 100000000
h_best = 0
C_best = 0
w_best = np.zeros(data.shape[1])
log_losses_best_set = [0, 0, 0, 0]
w_best_set = [0, 0, 0, 0]
log_test_best_set = [0, 0, 0, 0]
index = 0
for X, y in zip([X_quarter, X_half, X_three_quaters, X_train], [y_quarter, y_half, y_three_quaters, y_train]):
    for C in [0, 1, 0.1, 0.01]:
        for h in [1, 0.1, 0.01, 0.001, 0.0001]:
            w, log_losses, iter_count = gradient_descent(len(y), np.zeros(data.shape[1]), h, C, np.array(X), np.array(y))
            test_answers = np.vstack((1/(1 + np.exp(-np.sum(np.multiply(X_test, w), axis=1)*(-1))), 1/(1 + np.exp(-np.sum(np.multiply(X_test, w), axis=1))))).T
            if log_losses[-1] < log_losses_best:
                log_losses_best = log_losses[-1]
                log_losses_best_set[index] = log_losses_best
                h_best = h
                C_best = C
                w_best = w
                w_best_set[index] = w
                log_test_best_set[index] = log_loss(y_test, test_answers)
    index += 1


# In[31]:

x = [1, 2, 3, 4]
plt.plot(x, log_losses_best_set)
plt.xlabel('Data ratio')
plt.ylabel('Training data log loss')
plt.xticks(x, ['quarter', 'half', 'three_quarter', 'total'], rotation='vertical')
plt.show()


# In[32]:

x = [1, 2, 3, 4]
plt.plot(x, log_test_best_set)
plt.xlabel('Data ratio')
plt.ylabel('Test data log loss')
plt.xticks(x, ['quarter', 'half', 'three_quarter', 'total'], rotation='vertical')
plt.show()


# Построим графики показывающие результаты классификации при различном количестве признаков 

# In[33]:

log_losses_best = 100000000
h_best = 0
C_best = 0
w_best = np.zeros(data.shape[1])
log_losses_best_set2 = [0, 0, 0, 0]
w_best_set2 = [0, 0, 0, 0]
log_test_best_set2 = [0, 0, 0, 0]
index = 0
for features_num in [10, 20, 30]:
    for C in [0, 1, 0.1, 0.01]:
        for h in [1, 0.1, 0.01, 0.001, 0.0001]:
            w, log_losses, iter_count = gradient_descent(len(y_train), np.zeros(X_train.loc[:, 0:features_num].shape[1]), h, C, np.array(X_train.loc[:, 0:features_num]), np.array(y_train))
            test_answers = np.vstack((1/(1 + np.exp(-np.sum(np.multiply(X_test.loc[:, 0:features_num], w), axis=1)*(-1))), 1/(1 + np.exp(-np.sum(np.multiply(X_test.loc[:, 0:features_num], w), axis=1))))).T
            if log_losses[-1] < log_losses_best:
                log_losses_best = log_losses[-1]
                log_losses_best_set2[index] = log_losses_best
                h_best = h
                C_best = C
                w_best = w
                w_best_set2[index] = w
                log_test_best_set2[index] = log_loss(y_test, test_answers)
    index += 1


# In[34]:

x = [1, 2, 3]
plt.plot(x, log_losses_best_set2[0:3])
plt.xlabel('Features number')
plt.ylabel('Training data log loss')
plt.xticks(x, ['10', '20', '30'], rotation='vertical')
plt.show()


# In[35]:

x = [1, 2, 3]
plt.plot(x, log_test_best_set2[0:3])
plt.xlabel('Features number')
plt.ylabel('Test data log loss')
plt.xticks(x, ['10', '20', '30'], rotation='vertical')
plt.show()


# Loss function of iterations count by different h and C for random initial w for {} split

# In[36]:

def plot_all_graphs(iterations_set, log_losses_set, log_losses_test, key):
    figure, axes = plt.subplots(nrows=4, ncols=5, figsize=(30, 30))
    figure.suptitle('Loss function of iterations count by different h and C for random initial w for {} split'.format(key), fontsize=14, fontweight='bold')
    index = 0
    Cs = [0, 1, 0.1, 0.01]
    hs = [1, 0.1, 0.01, 0.001, 0.0001]
    for i in range(4):
        for j in range(5):
            axes[i][j].plot(np.arange(iterations_set[index]), log_losses_set[index])
            axes[i][j].legend(["h = {}, C = {}, test_error = {}".format(hs[j], Cs[i], round(log_losses_test[index], 3))])
            axes[i][j].set_xlabel("Iterations")
            axes[i][j].set_ylabel("Loss function")
            index += 1
    plt.show()


# In[37]:

for key in [0.25, 0.5, 0.75, 1]:
    plot_all_graphs(iterations_dict[key], log_losses_dict[key], log_test_dict[key], key)


# In[38]:

print(np.array(log_test_dict[0.25]) - np.array(log_test_dict[0.75]))


# In[39]:

print(np.min(np.array(log_test_dict[0.5])) - np.min(np.array(log_test_dict[1])))
print(np.min(np.array(log_test_dict[0.75])) - np.min(np.array(log_test_dict[1])))
print(np.min(np.array(log_test_dict[0.25])) - np.min(np.array(log_test_dict[0.75])))


# По результатам вычислений оказалось, что алгоритм, обученный на половине данных, имеет наименьшую ошибку на тестовых данных. Алгоритм, обученный на 3/4 данных оказался хуже, чем алгоритм, обученный на всей выборке. Это, однако, не означает, что необходимо обучаться только на половине данных, так как мы могли сделать такое разбиение, при котором именно половина тренировочных данных была наиболее похожа на тестовую выборку, а алгоритм, обученный на всей тренировочной выборке, имел лучшую обобщающую способность, но при этом чуть хуже показал себя на тестовых данных.

# In[40]:

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
print(np.argmax(np.abs(pca.components_)))
fig = plt.figure()
ax = fig.add_subplot(111)
x = np.random.randn(50)
ax.scatter(X_train[9], X_train[19], color = ['b' if i == 1 else 'g' for i in y_train])
ax.plot(x, -w_best_set2[-2][9]/w_best_set2[-2][19] * x + C_best, color='red')


# In[ ]:



