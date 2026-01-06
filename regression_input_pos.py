from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import os



no_cost = []
space_cost = []

no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_0.0_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_0.0_0_")

no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-10_2_")

no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-10_0.05_1e-09_2_")

no_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_0_")
space_cost.append("checkpoints_space_cartesian_nolimit_dynamic_128_2_5e-13_0.05_1e-08_2_")

for j in range(3):
    r2 = []
    for i in range(20):
        checkpoint_dir = no_cost[j] + str(i)
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
        W_in =np.load(no_cost[j] + str(i) + "/" + str(checkpoint) + "-Conn_Pop0_Pop1-g.npy").reshape(700,128)
        X =np.load(no_cost[j] + str(i) + "/" + str(checkpoint) + "-Pos0.npy")
        Y =np.load(no_cost[j] + str(i) + "/" + str(checkpoint) + "-Pos1.npy")
        Xreg = W_in.T
        rg_x = Ridge(alpha=1.0).fit(Xreg, X)
        rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
        pred_x = rg_x.predict(Xreg)
        pred_y = rg_y.predict(Xreg)
        r2_x = r2_score(X, pred_x)
        r2_y = r2_score(Y, pred_y)
        r2.append((r2_x+r2_y)/2)
    print("No cost: R2: ", np.mean(r2), np.std(r2))

for j in range(3):
    r2 = []
    for i in range(20):
        checkpoint_dir = space_cost[j] + str(i)
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
        W_in =np.load(space_cost[j] + str(i) + "/" + str(checkpoint) + "-Conn_Pop0_Pop1-g.npy").reshape(700,128)
        X =np.load(space_cost[j] + str(i) + "/" + str(checkpoint) + "-Pos0.npy")
        Y =np.load(space_cost[j] + str(i) + "/" + str(checkpoint) + "-Pos1.npy")
        Xreg = W_in.T
        rg_x = Ridge(alpha=1.0).fit(Xreg, X)
        rg_y = Ridge(alpha=1.0).fit(Xreg, Y)
        pred_x = rg_x.predict(Xreg)
        pred_y = rg_y.predict(Xreg)
        r2_x = r2_score(X, pred_x)
        r2_y = r2_score(Y, pred_y)
        r2.append((r2_x+r2_y)/2)
    print("Space cost: R2: ", np.mean(r2), np.std(r2))
