import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, rc
import matplotlib.gridspec as gridspec
from scipy import integrate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
import os
import torch
import torch.utils.data as Data
import torch.nn.functional as F


def rhs_reaction_diffusion(t, uvt, K22, d1, d2, beta, n, N):
    # Calculate u and v terms
    ut = np.reshape(uvt[:N], (n, n))
    vt = np.reshape(uvt[N:], (n, n))
    u = np.real(np.fft.ifft2(ut))
    v = np.real(np.fft.ifft2(vt))

    # reaction terms
    u3 = u ** 3
    v3 = v ** 3
    u2v = (u ** 2) * v
    uv2 = u * (v ** 2)
    utrhs = np.reshape((np.fft.fft2(u - u3 - uv2 + beta * u2v + beta * v3)), (N, 1))
    vtrhs = np.reshape((np.fft.fft2(v - v3 - u2v - beta * u3 - beta * uv2)), (N, 1))

    rhs = np.concatenate([-d1 * K22 * uvt[:N] + utrhs,
                          -d2 * K22 * uvt[N:] + vtrhs])[:, 0]

    return rhs


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


# Set Parameters
torch.manual_seed(113)    # reproducible results
np.random.seed(113)
seed = 113
dt = 0.05
final_t = 10  # Set final time horizon to forecast
t = np.linspace(0, final_t, int(final_t / dt) + 1)
d1 = 0.1
d2 = 0.1
beta = 1.0
L = 20
n = 32
N = n * n
x2 = np.linspace(-L / 2, L / 2, n + 1)
x = x2[0: n]
y = x
kx = (2 * np.pi / L) * np.concatenate([np.arange(0, n / 2), np.arange(-n / 2, 0)])
ky = kx

# Initial conditions
[X, Y] = np.meshgrid(x, y)
[KX, KY] = np.meshgrid(kx, ky)
K2 = KX ** 2 + KY ** 2
K22 = np.reshape(K2, (N, 1))

m = 1  # number of spirals

u = np.zeros((len(x), len(y), len(t)))
v = np.zeros((len(x), len(y), len(t)))

u[:, :, 0] = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.cos(m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2)))
v[:, :, 0] = np.tanh(np.sqrt(X ** 2 + Y ** 2)) * np.sin(m * np.angle(X + 1j * Y) - (np.sqrt(X ** 2 + Y ** 2)))

# Compute Reaction diffusion solution
uvt = np.concatenate([np.reshape(np.fft.fft2(u[:, :, 0]).T, (N, 1)), np.reshape(np.fft.fft2(v[:, :, 0]).T, (N, 1))])
sol = integrate.solve_ivp(rhs_reaction_diffusion, [0, final_t], uvt[:, 0], args=(K22, d1, d2, beta, n, N),
                          method='RK45', t_eval=t)
uvsol = sol.y

for j in range(len(t) - 1):
    ut = np.reshape(uvsol[:N, j + 1], (n, n))
    vt = np.reshape(uvsol[N:, j + 1], (n, n))
    u[:, :, j + 1] = np.real(np.fft.ifft2(ut))
    v[:, :, j + 1] = np.real(np.fft.ifft2(vt))

xy_dim = v.shape[0]
t_dim = v.shape[2]
n_test_time = int(t_dim * 0.2)
v_flat = np.reshape(v, (xy_dim ** 2, t_dim))
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

batch = 32

train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(dX_train).float())
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=False)

test_dataset = Data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(dX_test).float())
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=False)


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
