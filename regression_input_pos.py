from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import os
no_cost = "checkpoints_space_cartesian_limit_128_2_5e-10_0.05_62_1e-09_0_"
space_cost = "checkpoints_space_cartesian_limit_128_2_5e-12_0.3_62_1e-10_1_"
no_limit = "checkpoints_space_cartesian128_2_1.0_5e-13_0.01_"

x_r2, y_r2 = [],[]
for i in range(20):
    checkpoint_dir = no_cost + str(i)
    checkpoints = []
    for f in os.scandir(checkpoint_dir):
        fname = f.name
        if fname.split("-")[0].isdigit():
            checkpoints.append(int(fname.split("-")[0]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    if max(checkpoints) == 299:
        checkpoint = 299
    else:   
        checkpoint = max(checkpoints) - 16
    #print(checkpoint)
    W_in =np.load(no_cost + str(i) + "/" + str(checkpoint) + "-Conn_Pop0_Pop1-g.npy").reshape(700,128)
    X =np.load(no_cost + str(i) + "/" + str(checkpoint) + "-Pos0.npy")
    Y =np.load(no_cost + str(i) + "/" + str(checkpoint) + "-Pos1.npy")
    Xreg = W_in.T
    rg_x = Ridge(alpha=1.0).fit(Xreg, X)
    rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
    pred_x = rg_x.predict(Xreg)
    pred_y = rg_y.predict(Xreg)
    r2_x = r2_score(X, pred_x)
    r2_y = r2_score(Y, pred_y)
    x_r2.append(r2_x)
    y_r2.append(r2_y)
print("No cost: x R2: ", np.mean(x_r2), np.std(x_r2), "y R2: ", np.mean(y_r2), np.std(y_r2))

x_r2, y_r2 = [],[]
for i in range(20):
    checkpoint_dir = space_cost + str(i)
    checkpoints = []
    for f in os.scandir(checkpoint_dir):
        fname = f.name
        if fname.split("-")[0].isdigit():
            checkpoints.append(int(fname.split("-")[0]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    if max(checkpoints) == 299:
        checkpoint = 299
    else:   
        checkpoint = max(checkpoints) - 16
    #print(checkpoint)
    W_in =np.load(space_cost + str(i) + "/" + str(checkpoint) + "-Conn_Pop0_Pop1-g.npy").reshape(700,128)
    X =np.load(space_cost + str(i) + "/" + str(checkpoint) + "-Pos0.npy")
    Y =np.load(space_cost + str(i) + "/" + str(checkpoint) + "-Pos1.npy")
    Xreg = W_in.T
    rg_x = Ridge(alpha=1.0).fit(Xreg, X)
    rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
    pred_x = rg_x.predict(Xreg)
    pred_y = rg_y.predict(Xreg)
    r2_x = r2_score(X, pred_x)
    r2_y = r2_score(Y, pred_y)
    x_r2.append(r2_x)
    y_r2.append(r2_y)
print("Space cost: x R2: ", np.mean(x_r2), np.std(x_r2), "y R2: ", np.mean(y_r2), np.std(y_r2))

x_r2, y_r2 = [],[]
for i in range(20):
    checkpoint_dir = no_limit + str(i)
    checkpoints = []
    for f in os.scandir(checkpoint_dir):
        fname = f.name
        if fname.split("-")[0].isdigit():
            checkpoints.append(int(fname.split("-")[0]))
    if not checkpoints:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    if max(checkpoints) == 299:
        checkpoint = 299
    else:   
        checkpoint = max(checkpoints) - 16
    #print(checkpoint)
    W_in =np.load(no_limit + str(i) + "/" + str(checkpoint) + "-Conn_Pop0_Pop1-g.npy").reshape(700,128)
    X =np.load(no_limit + str(i) + "/" + str(checkpoint) + "_Pos0.npy")
    Y =np.load(no_limit + str(i) + "/" + str(checkpoint) + "_Pos1.npy")
    Xreg = W_in.T
    rg_x = Ridge(alpha=1.0).fit(Xreg, X)
    rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
    pred_x = rg_x.predict(Xreg)
    pred_y = rg_y.predict(Xreg)
    r2_x = r2_score(X, pred_x)
    r2_y = r2_score(Y, pred_y)
    x_r2.append(r2_x)
    y_r2.append(r2_y)
print("No limit: x R2: ", np.mean(x_r2), np.std(x_r2), "y R2: ", np.mean(y_r2), np.std(y_r2))



x_r2, y_r2 = [],[]
for i in range(20):
    W_in =np.random.normal(0.03, 0.02, (700,128))
    X =np.random.uniform(low=-1.0, high=1.0, size=(700))
    Y =np.random.uniform(low=-1.0, high=1.0, size=(700))
    Xreg = W_in
    rg_x = Ridge(alpha=1.0).fit(Xreg, X)
    rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
    pred_x = rg_x.predict(Xreg)
    pred_y = rg_y.predict(Xreg)
    r2_x = r2_score(X, pred_x)
    r2_y = r2_score(Y, pred_y)
    x_r2.append(r2_x)
    y_r2.append(r2_y)
print("Init x R2: ", np.mean(x_r2), np.std(x_r2), "y R2: ", np.mean(y_r2), np.std(y_r2))
