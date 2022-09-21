# In Jupyter/python3.0 
# Results in README.md
%matplotlib inline
#第一隐层
layer_one0 = (W[0].T) @ (p.T)
b = np.array([B[0]]).T
layer_one0 = layer_one0 + b

#非线性化操作
layer_one = relu(layer_one0)

vmin = layer_one0.min()
vmax = layer_one0.max()
norm = colors.Normalize(vmin=vmin, vmax=vmax)

cmap_name = 'bwr'
fig, ((ax1,ax2,ax3,ax4),(ax5,ax6,ax7,ax8)) = plt.subplots(2,4, figsize=(20, 8))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号

#作图 8张图
c1 = ax1.scatter(*p.T,c = layer_one0[0],cmap = cmap_name, norm = norm)
ax1.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax1.set_title('第二层的第一个节点')

c2 =ax2.scatter(*p.T,c = layer_one0[1],cmap = cmap_name, norm = norm)
ax2.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax2.set_title('第二层的第二个节点')

ax3.scatter(*p.T,c = layer_one0[2],cmap = cmap_name, norm = norm)
ax3.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax3.set_title('第二层的第三个节点')

ax4.scatter(*p.T,c = layer_one0[3],cmap = cmap_name, norm = norm)
ax4.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax4.set_title('第二层的第四个节点')

ax5.scatter(*p.T,c = layer_one[0],cmap = cmap_name, norm = norm)
ax5.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax5.set_title('非线性：第二层的第一个节点')

c6 = ax6.scatter(*p.T,c = layer_one[1],cmap = cmap_name, norm = norm)
ax6.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax6.set_title('非线性：第二层的第二个节点')

ax7.scatter(*p.T,c = layer_one[2],cmap = cmap_name, norm = norm)
ax7.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax7.set_title('非线性：第二层的第三个节点')

ax8.scatter(*p.T,c = layer_one[3],cmap = cmap_name, norm = norm)
ax8.scatter(*X_test.T,c = y_test,cmap = cmap_name,edgecolors='white')
ax8.set_title('非线性：第二层的第四个节点')

plt.colorbar(c2,ax=[ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8])
...;

#第一隐层的胞腔划分 二维空间
%matplotlib inline
plt.figure(figsize=(6,6))
layer_one_split = np.where(layer_one<=0,0,1)
layer_one_split_result=layer_one_split[0]*1+layer_one_split[1]*2+layer_one_split[2]*4+layer_one_split[3]*8
c2 =plt.scatter(*p.T,c = layer_one_split_result,cmap = 'tab20')
# plt.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')

% matplotlib notebook
#前三个cell动态图（画后三个cell的时候*x_.T[:3]改成*x_.T[1:]，
#layer_one[0],layer_one[1],layer_one[2]改成layer_one[1],layer_one[2],layer_one[3]
layer_one_split = np.where(layer_one<=0,0,1)
layer_one_split_result=layer_one_split[0]*1+layer_one_split[1]*2+layer_one_split[2]*4

x_ = relu(X @ W[0] + B[0])

fig,(ax1,ax2) = plt.subplots(1,2,figsize = (9,5),subplot_kw = {'projection':'3d'})
# ax1.scatter(layer_one[0],layer_one[1],layer_one[2],c = layer_one_split_result,cmap = 'tab10')
ax1.scatter(*x_.T[:3],c = y,cmap = cm_bright,edgecolors='white')
ax1.elev,ax1.azim = 15, -60

ax2.scatter(layer_one[0],layer_one[1],layer_one[2],c = layer_one_split_result,cmap = 'tab10')
ax2.scatter(*x_.T[:3],c = y,cmap = cm_bright,edgecolors='white')
def update(i):
    ax2.azim = i * 10
    ax2.elev = i * 10
    return ax2
ani = FuncAnimation(fig, update, 36,  interval=500);

#下载安装imagemagick，添加环境变量
ani.save('d3.gif', writer='imagemagick')
plt.show();
...;

% matplotlib inline
#第二隐层
#作图 共4张
layer_two = (W[1].T) @ layer_one
b = np.array([B[1]]).T
layer_two = layer_two + b


fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2, figsize=(9, 6))
plt.rcParams['font.sans-serif'] = ['SimHei']
vmin = -5
vmax = 5
norm = colors.Normalize(vmin=vmin, vmax=vmax)

ax1.scatter(*p.T,c = layer_two[0],cmap = cmap_name, norm = norm)
ax1.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')
ax1.set_title('第三层的第一个节点')

ax2.scatter(*p.T,c = layer_two[1],cmap = cmap_name, norm = norm)
ax2.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')
ax2.set_title('第三层的第二个节点')
#非线性
layer_two = relu(layer_two)

ax3.scatter(*p.T,c = layer_two[0],cmap = cmap_name, norm = norm)
ax3.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')
ax3.set_title('非线性：第三层的第一个节点')

c2 = ax4.scatter(*p.T,c = layer_two[1],cmap = cmap_name, norm = norm)
ax4.scatter(*X_test.T,c = y_test,cmap = cm_bright,edgecolors='white')
ax4.set_title('非线性：第二层的第二个节点')

plt.colorbar(c2,ax=[ax1,ax2,ax3,ax4])
...;

%matplotlib inline
#第二隐层（第三层）的二维平面图
x1_ = relu(X @ W[0] + B[0])
x2_ = relu(x1_ @ W[1] + B[1])
p1_ = relu(p @ W[0] + B[0])
p2_ = relu(p1_ @ W[1] + B[1])
fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(12, 6))
layer_two_split = np.where(layer_two<=0,0,1)
layer_two_split_result=layer_two_split[0]*1+layer_two_split[1]*2
c2 =ax1.scatter(*p2_.T,c = layer_two_split_result,cmap = cmap_name)
ax2.scatter(*x2_.T,c = y, cmap = cmap_name, edgecolors='white')
ax3.scatter(*p2_.T,c = layer_two_split_result,cmap = cmap_name)
ax3.scatter(*x2_.T,c = y, cmap = cmap_name, edgecolors='white')
...;

#输出层
#作图 分别为非线性变换前和非线性变换后
layer_two = relu(layer_two)
layer_three = (W[2].T) @ (layer_two)
layer_three = layer_three + B[2].T

x1_ = relu(X @ W[0] + B[0])
x2_ = relu(x1_ @ W[1] + B[1])
x3_ = x2_ @ W[2] + B[2]

p1_ = relu(p @ W[0] + B[0])
p2_ = relu(p1_ @ W[1] + B[1])
p3_ = p2_ @ W[2] + B[2]

plt.rcParams['font.sans-serif'] = ['SimHei']
fig, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(15, 9))


vmin = p3_.min()
vmax = p3_.max()
vmin = -vmax if vmax > -vmin else vmin
vmax = -vmin if vmax < -vmin else vmax
norm = colors.Normalize(vmin=vmin, vmax=vmax)

c2 = ax1.scatter(*p3_.T,*p3_.T,c = layer_three[0],cmap = cmap_name, norm = norm)
ax2.scatter(*x3_.T,*x3_.T,c = y,cmap = cmap_name,edgecolors='white')
ax3.scatter(*p3_.T,*p3_.T,c = layer_three[0],cmap = cmap_name, norm = norm)
ax3.scatter(*x3_.T,*x3_.T,c = y,cmap = cmap_name,edgecolors='white')
ax1.set_title('输出层特征空间')
ax2.set_title('输出层数据点')
ax3.set_title('合并')

layer_three = relu(layer_three)
x3_ = relu(x2_ @ W[2] + B[2])
p3_ = relu(p2_ @ W[2] + B[2])


ax4.scatter(*p3_.T,*p3_.T,c = layer_three[0],cmap = cmap_name, norm = norm)
ax5.scatter(*x3_.T,*x3_.T,c = y,cmap =cmap_name,edgecolors='white')
ax6.scatter(*p3_.T,*p3_.T,c = layer_three[0],cmap = cmap_name, norm = norm)
ax6.scatter(*x3_.T,*x3_.T,c = y,cmap =cmap_name,edgecolors='white')
ax4.set_title('非线性：输出层特征空间')
ax5.set_title('非线性：输出层数据点')
ax6.set_title('非线性：合并')

plt.colorbar(c2,ax=[ax1,ax2,ax3,ax4,ax5,ax6])
...;
