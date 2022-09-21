from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from matplotlib import colors
import time
%matplotlib notebook

#three activation function
def relu(x):
    return np.where(x <= 0,0,x)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

# Generate samples
X,y= datasets.make_circles(n_samples = 2000, factor=0.3, noise=.1)
# Dividing the training set and test set
X, X_test, y, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Constructing feature spaces
c,r = np.mgrid[[slice(X.min() - .5,X.max() + .5,50j)]*2]
# Iterators
p = np.c_[c.flat,r.flat]
# p is the coordinate space
# Normalization
%matplotlib inline
ss = StandardScaler().fit(X)
X = ss.transform(X)
p = ss.transform(p)
X_test = ss.transform(X_test)

#数据可视化
fig = plt.figure(figsize = (9,3))
cm_bright = ListedColormap(['#f32b1a','#0c6dea'])
plt.subplot(121)
m1 = plt.scatter(*X.T,c = y,cmap = cm_bright,edgecolors='white')
plt.title('train samples')
plt.axis('equal')
plt.subplot(122)
m2 = plt.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white');
plt.title('test samples')
plt.axis('equal')
ax = fig.get_axes()
plt.colorbar(ax = ax);
plt.show();

#分类
loss,train,test = [], [], []
#以Relu为例
MLP = MLPClassifier(hidden_layer_sizes=(4,2),max_iter = 20, activation='relu',
                    warm_start = True,learning_rate_init=0.008,)
start = time.time()
#训练了500次，epoch
for i in range(500):
    MLP.fit(X,y)
    loss.append(MLP.loss_)
    train.append(MLP.score(X,y))
    test.append(MLP.score(X_test,y_test))
#记录训练时间
    end = time.time()
print (end-start)
...;
#记录权重 和偏置量
W,B = MLP.coefs_ , MLP.intercepts_
W
B
z = MLP.predict(p)
...;
#数据训练情况
%matplotlib inline
fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(6, 6))
ax1.scatter(*p.T,c = z,cmap = cm_bright)
ax1.scatter(*X.T,c = y,cmap = cm_bright,edgecolors='white')
ax1.set_title('train score:%.5f'%train[-1])

ax2.scatter(*p.T,c = z,cmap = cm_bright)
ax2.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')
ax2.set_title('test score:%.5f'%test[-1])

ax3.plot(loss)
ax3.set_title('Loss')

ax4.plot(train,c = 'r',label = 'train')
ax4.plot(test,c = 'b',label = 'test')
ax4.set_title('train-test score');
ax4.legend();
plt.show();
