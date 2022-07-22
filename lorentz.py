import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
import torch.utils.data as Data
import torch.nn.functional as F


def Lorenz(x, t, dummy, sigma, b, r):
    x1, x2, x3 = x

    rhs = [sigma * (x2 - x1),
           r * x1 - x2 - x1 * x3,
           x1 * x2 - b * x3]
    return rhs


seed = 123
dt = 0.001
final_t = 10
n = int(final_t / dt) + 1
t = np.linspace(0, final_t, n)

beta = 8/3
sigma = 10
rho = [10, 28, 35]
x0 = [5, 5, 5]  # initial conditions

X_train, y_train = np.empty((0,4)), np.empty((0,3))
X_val, y_val = np.empty((0,4)), np.empty((0,3))
X_test, y_test = np.empty((0,4)), np.empty((0,3))
rho_training = rho
n_trials = 10  # number of iterations for generation of training data
EXTRAPOLATE = True

if n_trials > len(rho):  # Add random rho parameters
    for idx in range(0, n_trials-len(rho)):
        rho_training.append(np.random.randint(1, 50))

for idx in range(0, n_trials):
    if idx == 0:          
        x0_r = x0
    else:
        x_0r = np.random.randint(0, 11)
        y_0r = np.random.randint(0, 11)
        z_0r = np.random.randint(0, 11)
        x0_r = [x_0r, y_0r, z_0r]
    for _rho in rho_training:
        lor = odeint(Lorenz, x0_r, t, args=([], sigma, beta, _rho), mxstep=10**8)  # integrate PDE
        x_ode = np.reshape(lor[:, 0], (-1, 1))
        y_ode = np.reshape(lor[:, 1], (-1, 1))
        z_ode = np.reshape(lor[:, 2], (-1, 1))
        #x_ode_norm = (x_ode - min(x_ode))/(max(x_ode)-min(x_ode))
        #y_ode_norm = (y_ode - min(y_ode)) / (max(y_ode) - min(y_ode))
        #z_ode_norm = (z_ode - min(z_ode)) / (max(z_ode) - min(z_ode))
        param = np.ones_like(x_ode) * _rho
        data = np.hstack((x_ode, y_ode, z_ode, param))
        X = data[:-1, :]
        y = data[1:, :-1]
        _X_train, _X_tmp, _y_train, _y_tmp = train_test_split(X, y, test_size=0.3, random_state=seed)
        _X_val, _X_test, _y_val, _y_test = train_test_split(_X_tmp, _y_tmp, test_size=0.5, random_state=seed)
        X_train = np.append(X_train, _X_train, axis=0)
        y_train = np.append(y_train, _y_train, axis=0)
        X_val = np.append(X_val, _X_val, axis=0)
        y_val = np.append(y_val, _y_val, axis=0)
        X_test = np.append(X_test, _X_test, axis=0)
        y_test = np.append(y_test, _y_test, axis=0)


class MLPNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLPNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        y_pred = self.linear5(x)
        return y_pred


batch = 64  # int(X_train.shape[0] / len(rho) / 10)

train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = Data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
#test_dataset = Data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=X_val.shape[0], shuffle=False)
#test_loader = Data.DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=False)


D_in, H, D_out = 4, 100, 3
epochs = 500

model = MLPNN(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

best_so_far = np.inf
counter = 0
for epoch in range(epochs):
    epoch_val_loss = 0
    train_loss = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        train_loss.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model.eval()
    epoch_train_loss = np.sum(train_loss)/len(train_loader)
    for step, (val_x, val_y) in enumerate(val_loader):
        val_pred = model(val_x)
        loss_val = criterion(val_pred, val_y)
        epoch_val_loss = (epoch_val_loss + loss_val)/len(val_loader)
    if epoch_val_loss < best_so_far:
        best_so_far = epoch_val_loss
        counter = 0
    else:
        counter += 1
    if counter > 20:
        break
    model.train()
    print("Iteration: ", epoch, " Loss: ", epoch_train_loss, " Validation loss: ", epoch_val_loss)


x_plt = []
y_plt = []
z_plt = []
ynn_plt = []
output1 = np.zeros((len(t), 3))
for _rho in rho:
    lor = odeint(Lorenz, x0, t, args=([], sigma, beta, _rho), mxstep=10**8)  # integrate PDE
    x = np.reshape(lor[:, 0], (-1, 1))
    y = np.reshape(lor[:, 1], (-1, 1))
    z = np.reshape(lor[:, 2], (-1, 1))
    x_plt.append(x)
    y_plt.append(y)
    z_plt.append(z)

    ynn = np.zeros((len(t), 4))
    ynn[0, :-1] = x0
    ynn[:, -1] = _rho
    with torch.no_grad():
        for i in range(len(t)-1):
            ynn[i+1, :-1] = model(torch.from_numpy(ynn[i, :]).squeeze(0).float())
    #x_nn_plt = (ynn[:, 0] * (max(ynn[:, 0]) - min(ynn[:, 0])) + min(ynn[:, 0]))
    #y_nn_plt = (ynn[:, 1] * (max(ynn[:, 1]) - min(ynn[:, 1])) + min(ynn[:, 1]))
    #z_nn_plt = (ynn[:, 2] * (max(ynn[:, 2]) - min(ynn[:, 2])) + min(ynn[:, 2]))
    #output1[:, 0] = x_nn_plt[:]
    #output1[:, 1] = y_nn_plt[:]
    #output1[:, 2] = z_nn_plt[:]
    ynn_plt.append(ynn)


if EXTRAPOLATE:
    # future state prediction
    _rho = 17
    lor_f1 = odeint(Lorenz, x0, t, args=([], sigma, beta, _rho), mxstep=10**8) # integrate PDE
    x_f1 = np.reshape(lor_f1[:, 0], (-1, 1))
    y_f1 = np.reshape(lor_f1[:, 1], (-1, 1))
    z_f1 = np.reshape(lor_f1[:, 2], (-1, 1))
    ynn_f1 = np.zeros((len(t), 4))
    ynn_f1[0, :-1] = x0
    ynn_f1[:, -1] = _rho
    with torch.no_grad():
        for i in range(len(t) - 1):
            ynn_f1[i + 1, :-1] = model(torch.from_numpy(ynn_f1[i, :]).squeeze(0).float())
    #x_nn_plt_f1 = (ynn_f1[:, 0] * (max(ynn_f1[:, 0]) - min(ynn_f1[:, 0])) + min(ynn_f1[:, 0]))
    #y_nn_plt_f1 = (ynn_f1[:, 1] * (max(ynn_f1[:, 1]) - min(ynn_f1[:, 1])) + min(ynn_f1[:, 1]))
    #z_nn_plt_f1 = (ynn_f1[:, 2] * (max(ynn_f1[:, 2]) - min(ynn_f1[:, 2])) + min(ynn_f1[:, 2]))


    _rho = 40
    lor_f2 = odeint(Lorenz, x0, t, args=([], sigma, beta, _rho), mxstep=10**8) # integrate PDE
    x_f2 = np.reshape(lor_f2[:, 0], (-1, 1))
    y_f2 = np.reshape(lor_f2[:, 1], (-1, 1))
    z_f2 = np.reshape(lor_f2[:, 2], (-1, 1))
    ynn_f2 = np.zeros((len(t), 4))
    ynn_f2[0, :-1] = x0
    ynn_f2[:, -1] = _rho
    with torch.no_grad():
        for i in range(len(t) - 1):
            ynn_f2[i + 1, :-1] = model(torch.from_numpy(ynn_f2[i, :]).squeeze(0).float())
    #x_nn_plt_f2 = (ynn_f2[:, 0] * (max(ynn_f2[:, 0]) - min(ynn_f2[:, 0])) + min(ynn_f2[:, 0]))
    #y_nn_plt_f2 = (ynn_f2[:, 1] * (max(ynn_f2[:, 1]) - min(ynn_f2[:, 1])) + min(ynn_f2[:, 1]))
    #z_nn_plt_f2 = (ynn_f2[:, 2] * (max(ynn_f2[:, 2]) - min(ynn_f2[:, 2])) + min(ynn_f2[:, 2]))


# PLOTS
lgd1 = []
fig1 = plt.figure(1)
ax1 = fig1.add_subplot(projection='3d')
for plt_idx in range(0, len(rho)):
    ax1.plot(x_plt[plt_idx][:, 0], y_plt[plt_idx][:, 0], z_plt[plt_idx][:, 0])
    lgd1.append('rho %d' % rho[plt_idx])
ax1.scatter(x0[0], x0[1], x0[2], 'o', color='r')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.legend(lgd1, fontsize = 12, loc='upper left')
fig1.suptitle('Real Behavior')

lgd2 = []
fig2 = plt.figure(2)
ax2 = fig2.add_subplot(projection='3d')
for plt_idx in range(0, len(rho)):
    #ax2.plot(x_plt[plt_idx][:, 0], y_plt[plt_idx][:, 0], z_plt[plt_idx][:, 0])
    ax2.plot(ynn_plt[plt_idx][:, 0], ynn_plt[plt_idx][:, 1], ynn_plt[plt_idx][:, 2], '-.')
    lgd2.append('rho %d' % rho[plt_idx])
ax2.scatter(x0[0], x0[1], x0[2], 'o', color='r')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.legend(lgd2, fontsize = 12, loc='upper left')
fig2.suptitle('Neural Network')

if EXTRAPOLATE:
    lgd3 = []
    fig3 = plt.figure(3)
    ax3 = fig3.add_subplot(projection='3d')
    ax3.plot(x_f1[:, 0], y_f1[:, 0], z_f1[:, 0])
    lgd3.append('real')
    ax3.plot(ynn_f1[:, 0], ynn_f1[:, 1], ynn_f1[:, 2])
    lgd3.append('NN Forecast')
    ax3.scatter(x0[0], x0[1], x0[2], 'o', color='r')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('z')
    ax3.legend(lgd3, fontsize = 12, loc='upper left')
    fig3.suptitle('Future Prediction rho=17')

    lgd4 = []
    fig4 = plt.figure(4)
    ax4 = fig4.add_subplot(projection='3d')
    ax4.plot(x_f2[:, 0], y_f2[:, 0], z_f2[:, 0])
    lgd4.append('real')
    ax4.plot(ynn_f2[:, 0], ynn_f2[:, 1], ynn_f2[:, 2])
    lgd4.append('NN Forecast')
    ax4.scatter(x0[0], x0[1], x0[2], 'o', color='r')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.legend(lgd4, fontsize = 12, loc='upper left')
    fig4.suptitle('Future Prediction rho=40')

plt.show()
