import numpy as np
from numpy.random import default_rng
from numpy.fft import fft, ifft, fftfreq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sklearn
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def KS(lenght, x_dim, final_t):
    N = x_dim
    x = (lenght * np.pi * np.arange(1, N + 1) / N)
    u = np.cos(x / 16) * (1 + np.sin(x / 16))
    v = fft(u)

    # spatial grid and initial conditions
    h = 0.025 # choose time step size
    k = (np.r_[np.arange(0, N / 2), np.array([0]), np.arange(-N / 2 + 1, 0)] / 16).astype(np.float64)
    L = k ** 2 - k ** 4
    exp1 = np.exp(h * L)
    exp2 = np.exp(h * L / 2)
    M = 16
    r = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = h * np.repeat([L], M, axis=0).T + np.repeat([r], N, axis=0)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=1))
    f1 = h * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR ** 2)) / LR ** 3, axis=1))
    f2 = h * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR ** 3, axis=1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR ** 2 + np.exp(LR) * (4 - LR)) / LR ** 3, axis=1))

    tmax = final_t
    step_max = round(tmax / h)
    # step_plt = int(tmax / (1000 * h))
    step_plt = 32
    g = -0.5j * k
    uu = u
    tt = 0

    for step in range(1, step_max):
        t = step * h
        Nv = g * fft(np.real(ifft(v)) ** 2)
        a = exp2 * v + Q * Nv
        Na = g * fft(np.real(ifft(a)) ** 2)
        b = exp2 * v + Q * Na
        Nb = g * fft(np.real(ifft(b)) ** 2)
        c = exp2 * a + Q * (2 * Nb - Nv)
        Nc = g * fft(np.real(ifft(c)) ** 2)
        v = exp1 * v + Nv * f1 + 2 * (Na + Nb) * f2 + Nc * f3
        if step % step_plt == 0:
            u = np.real(ifft(v))
            uu = np.vstack([uu, u])
            tt = np.hstack((tt, t))
    return x, uu, tt


torch.manual_seed(113)    # set seed for allowing reproducible results
np.random.seed(113)
seed = 113
n_x = 128
t_max_training = 100
t_removed = 0   # remove initial transient from data
X_train, y_train = np.empty((0, n_x)), np.empty((0, n_x))
X_val, y_val = np.empty((0, n_x)), np.empty((0, n_x))
X_test, y_test = np.empty((0, n_x)), np.empty((0, n_x))
n_iterations = 1  # Set greater than 1 for augment dataset dimensions

L_training = [32]

if n_iterations>1:
    for idx in range(0, n_iterations):
        L_training.append(np.random.randint(low=10, high=50))

for _L in L_training:
    x_tmp, u_tmp, t = KS(_L, n_x, t_max_training)
    n_t = len(t)  # 1875
    t_max = t_max_training  # t[-1]
    X = u_tmp[:-1, :]
    y = u_tmp[1:, :]
    # Train / Test split
    _X_train, _X_tmp, _y_train, _y_tmp = train_test_split(X, y, test_size=0.2, random_state=seed)
    _X_val, _X_test, _y_val, _y_test = train_test_split(_X_tmp, _y_tmp, test_size=0.5, random_state=seed+1)
    # Generate Data Matrices
    X_train = np.append(X_train, _X_train, axis=0)
    y_train = np.append(y_train, _y_train, axis=0)
    X_val = np.append(X_val, _X_val, axis=0)
    y_val = np.append(y_val, _y_val, axis=0)
    X_test = np.append(X_test, _X_test, axis=0)
    y_test = np.append(y_test, _y_test, axis=0)
    t_stop_train = int(t_max * 0.8)
    t_train = t[t_removed:t_stop_train]


class MLPNN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(MLPNN, self).__init__()
        self.linear2 = torch.nn.Linear(D_in, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = self.linear2(F.relu(x))
        x = self.linear3(F.relu(x))
        x = self.linear4(F.relu(x))
        y_pred = self.linear5(x)
        return y_pred


batch = 64

train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
val_dataset = Data.TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())
test_dataset = Data.TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
val_loader = Data.DataLoader(dataset=val_dataset, batch_size=X_val.shape[0], shuffle=False)
test_loader = Data.DataLoader(dataset=test_dataset, batch_size=X_test.shape[0], shuffle=False)

D_in, H, D_out = n_x, int(2*n_x), n_x
epochs = 1000
model1 = MLPNN(D_in, H, D_out)

# Training Loop
criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.Adam(model1.parameters(), lr=1e-4)
train_loss = []
best_so_far = np.inf
counter = 0
for epoch in range(epochs):
    epoch_val_loss = 0
    train_loss = []
    for step, (batch_x, batch_y) in enumerate(train_loader):
        y_pred = model1(batch_x)
        loss = criterion(y_pred, batch_y)
        train_loss.append(loss.item())
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    model1.eval()
    epoch_train_loss = np.sum(train_loss)/len(train_loader)
    for step, (val_x, val_y) in enumerate(val_loader):
        val_pred = model1(val_x)
        loss_val = criterion(val_pred, val_y)
        epoch_val_loss = (epoch_val_loss + loss_val)/len(val_loader)
    if epoch_val_loss < best_so_far:
        best_so_far = epoch_val_loss
        counter = 0
    else:
        counter += 1
    if counter > 20:
        break
    model1.train()
    print("Iteration: ", epoch, " Loss: ", epoch_train_loss, " Validation loss: ", epoch_val_loss)

# Test NN by varying time step

# time 100
L_test = 32
t_max1 = t_max_training
x_f1, u_f1, t_f1 = KS(L_test, n_x, t_max1)
n_t1 = len(t_f1)
u_frc1 = np.empty((n_t1, n_x))
u_frc1[0, :] = u_f1[0, :]
for i in range(n_t1 - 1):
    with torch.no_grad():
        u_frc1[i + 1, :] = model1(torch.from_numpy(u_frc1[i, :]).squeeze(0).float())

# NN vs ODE
L1 = 24     # Vary initial condition u (through x)
x_f3, u_f3, t_f3 = KS(L1, n_x, t_max1)
n_t3 = len(t_f3)
u_frc3 = np.empty((n_t3, n_x))
u_frc3[0, :] = u_f3[0, :]
u_frc3[:, -1] = L1
for i in range(n_t3 - 1):
    with torch.no_grad():
        u_frc3[i + 1, :] = model1(torch.from_numpy(u_frc3[i, :]).squeeze(0).float())

L2 = 36
x_f4, u_f4, t_f4 = KS(L2, n_x, t_max1)
n_t4 = len(t_f4)
u_frc4 = np.empty((n_t4, n_x))
u_frc4[0, :] = u_f4[0, :]
u_frc4[:, -1] = L2
for i in range(n_t4 - 1):
    with torch.no_grad():
        u_frc4[i + 1, :] = model1(torch.from_numpy(u_frc4[i, :]).squeeze(0).float())


# PLOT
fig = plt.figure(figsize=(7, 4))
gs = gridspec.GridSpec(1, 1)
gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.3)
ax = plt.subplot(gs[0, 0], projection='3d')
T, XX = np.meshgrid(t, x_f1)
h = ax.plot_surface(T, XX, u_f1.T, cmap='inferno')
ax.view_init(elev=30., azim=-120)
ax.set_xlabel('time', fontsize = 14)
ax.set_ylabel('x', fontsize = 14)
ax.set_title('Kuramoto-Sivashinsky equation', fontsize = 18)
gs.tight_layout(fig)


fig1 = plt.figure(figsize=(14, 4))
fig1.suptitle("ANN forecasting time variation", fontsize=18)
norm = mpl.colors.Normalize(vmin=-4, vmax=4)

gs = gridspec.GridSpec(1, 2)
gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.2)

ax = plt.subplot(gs[0, 0])
h = ax.imshow(u_f1.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L_test*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
#ax.fill([0, t_stop_train, t_stop_train, 0], [-100, -100, 200, 200], color='k', alpha=0.3)
ax.set_ylim([0, L_test*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig1.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('Real behavior %d s' % t_max1, fontsize = 12)

ax = plt.subplot(gs[0, 1])
h = ax.imshow(u_frc1.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L_test*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
#ax.fill([0, t_stop_train, t_stop_train, 0], [-100, -100, 200, 200], color='k', alpha=0.3)
ax.set_ylim([0, L_test*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.04)
fig1.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('ANN Forecasting %d s' % t_max1, fontsize = 12)
gs.tight_layout(fig1)


fig2 = plt.figure(figsize=(14, 8))
fig2.suptitle("ANN forecasting initial condition variation", fontsize=18)
norm = mpl.colors.Normalize(vmin=-4, vmax=4)

gs = gridspec.GridSpec(2, 2)
gs.update(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.3, hspace=0.2)

ax = plt.subplot(gs[0, 0])
h = ax.imshow(u_f3.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L1*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
ax.set_ylim([0, L1*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig2.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('Real behavior L %d' % L1, fontsize = 12)

ax = plt.subplot(gs[0, 1])
h = ax.imshow(u_frc3.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L1*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
ax.set_ylim([0, L1*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.04)
fig2.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('ANN Forecasting L %d' % L1, fontsize = 12)

ax = plt.subplot(gs[1, 0])
h = ax.imshow(u_f4.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L2*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
ax.set_ylim([0, L2*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.05)
fig2.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('Real behavior L %d' % L2, fontsize = 12)

ax = plt.subplot(gs[1, 1])
h = ax.imshow(u_frc4.T, interpolation='nearest', cmap='inferno', extent=[0, t_max1, 0, L2*np.pi], norm=norm, origin='lower', aspect='auto')
divider = make_axes_locatable(ax)
ax.set_ylim([0, L2*np.pi])
cax = divider.append_axes("right", size="5%", pad=0.04)
fig2.colorbar(h, cax=cax)
ax.set_xlabel('time', fontsize = 11)
ax.set_ylabel('x', fontsize = 11)
ax.set_title('ANN Forecasting L %d' % L2, fontsize = 12)

gs.tight_layout(fig2, rect=[0, 0.03, 1, 0.95])

plt.show()
