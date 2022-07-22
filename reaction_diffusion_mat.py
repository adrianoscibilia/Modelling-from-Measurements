import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
import matplotlib.gridspec as gridspec
from scipy import integrate
import scipy.io as sio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
import torch.utils.data as Data
import torch.nn.functional as F


data = sio.loadmat('./reaction_diffusion_big.mat')

t = data['t'][:,0]
x = data['x'][0,:]
y = data['y'][0,:]
U = data['u']
V = data['v']

n = len(x) # also the length of y
steps = len(t)
dx = x[2]-x[1]
dy = y[2]-y[1]
dt = t[2]-t[1]

xy_dim = V.shape[0]
n_test_time = int(steps * dt)
v_flat = np.reshape(V, (xy_dim ** 2, steps))
X = np.copy(v_flat)

# Do SVD
U, S, Vh = np.linalg.svd(X, full_matrices=False)
V = Vh.conj().T
r = 8
Ur = U[:, :r]
Vr = V[:, :r]
Sr = S[:r]

# Project data in the reduced space
_X_reduced = Ur.conj().T @ X
X_reduced = _X_reduced[:, :-1]
dX_reduced = _X_reduced[:, 1:]

# train data
X_train = X_reduced[:, :-n_test_time].T
dX_train = dX_reduced[:, 1:-n_test_time+1].T
# test data
X_test = X_reduced[:, -n_test_time:-1].T
dX_test = dX_reduced[:,-n_test_time+1:].T

# Set Parameters
torch.manual_seed(113)    # reproducible results
np.random.seed(113)
seed = 113

batch = 32

train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(dX_train).float())
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)

test_dataset = Data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(dX_test).float())
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=False)


class MLPNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLPNN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear1(F.relu(x))
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        y_pred = self.linear4(torch.tanh(x))*2
        return y_pred


n_x = X_train.shape[1]
D_in, H, D_out = n_x, 60, n_x
epochs = 1000

model = MLPNN(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
train_loss = []
best_so_far = np.inf
counter = 0
for epoch in range(epochs):
    for step, (batch_x, batch_y) in enumerate(train_loader):
        y_pred = model(batch_x)
        loss = criterion(y_pred, batch_y)
        train_loss.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.eval()
        for step, (test_x, test_y) in enumerate(test_loader):
            test_pred = model(test_x)
            loss_test = criterion(test_pred, test_y)
            if loss_test < best_so_far:
                best_so_far = loss_test
                counter = 0
            else:
                counter +=1
            if counter >10:
                break
        model.train()
        print("Iteration: ", epoch, " Loss: ", loss.item(), " Test loss: ", loss_test.item())

_X_test = torch.from_numpy(X_test).float()
_dX_test = torch.from_numpy(dX_test).float()

test_prediction = X_test[None, 0, :]
with torch.no_grad():
    for i in range(1, _X_test.shape[0]):
        curr_pred = model(torch.from_numpy(test_prediction[i-1, :]).float())
        test_prediction = np.vstack((test_prediction, curr_pred))
_forecast = Ur @ test_prediction.T
_real = Ur @ dX_test.T
forecast = np.reshape(_forecast, (xy_dim, xy_dim, -1))
real = np.reshape(_real, (xy_dim, xy_dim, -1))

# PLOTS
plt.figure(1)
plt.plot(range(1,len(S)+1), S / np.sum(S), 'o')
plt.title('Singual values')
plt.xlim([0, 30])
plt.xlabel('eigenvalues')
plt.ylabel('energy')
plt.grid()

plt.figure(2)
plt.imshow(real[:, :, 0], interpolation='lanczos', cmap='inferno')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Real')

plt.figure(3)
plt.imshow(forecast[:, :, 0], interpolation='lanczos', cmap='inferno')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Forecast')

plt.show()
