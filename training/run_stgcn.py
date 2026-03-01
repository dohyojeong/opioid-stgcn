# ============================================
# 1. Environment Setup and Libraries (Seed Fixing)
# ============================================
import random
import numpy as np
import torch

seed = 2025
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import geopandas as gpd
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from math import sqrt

from libpysal.weights import Rook

import torch.nn as nn
import torch.nn.functional as F

print("Environment ready.")

# ===================================================
# 2. Load Data (Cook County 1km Monthly Panel Example)
# ===================================================
shp_path = "data/Cook_1km_panel_wide.shp"

gdf = gpd.read_file(shp_path)

grid_id_col = "TARGET_FID"

month_cols = [
    "Month_1", "Month_2", "Month_3", "Month_4",
    "Month_5", "Month_6", "Month_7", "Month_8",
    "Month_9", "Month_10", "Month_11", "Month_12"
]
month_cols = [c for c in month_cols if c in gdf.columns]

if len(month_cols) < 2:
    raise ValueError("At least two monthly columns are required for forecasting.")

print("Monthly columns used:", month_cols)

month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"][:len(month_cols)]

gdf_sorted = gdf.sort_values(grid_id_col).reset_index(drop=True)

node_ids = gdf_sorted[grid_id_col].values
data = gdf_sorted[month_cols].values.astype(np.float32)
data = np.nan_to_num(data)

num_nodes, num_timesteps = data.shape
print(f"Panel data shape (N,T): {data.shape}")

# ============================================
# 3. Forecast Tensor Construction
# ============================================
X_in = data[:, :-1]
Y_out = data[:, 1:]
T_out = X_in.shape[1]

print("Forecast input shape (N,T-1):", X_in.shape)
print("Forecast target shape (N,T-1):", Y_out.shape)

# Optional stabilization for zero-heavy count data
use_log1p = True
if use_log1p:
    X_in = np.log1p(X_in)
    Y_out = np.log1p(Y_out)

data_tensor = torch.tensor(X_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
target_tensor = torch.tensor(Y_out, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# ============================================
# 4. Rook Adjacency Matrix Construction (with normalization)
# ============================================
rook_w = Rook.from_dataframe(gdf_sorted, ids=gdf_sorted[grid_id_col])

A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
id_to_idx = {gid: idx for idx, gid in enumerate(node_ids)}

for gid, neighbors in rook_w.neighbors.items():
    i = id_to_idx[gid]
    for n_gid in neighbors:
        j = id_to_idx[n_gid]
        A[i, j] = 1.0
        A[j, i] = 1.0

np.fill_diagonal(A, 1.0)
A = torch.tensor(A, dtype=torch.float32)

# Symmetric normalization: D^{-1/2} A D^{-1/2}
deg = A.sum(dim=1)
deg_inv_sqrt = torch.pow(deg, -0.5)
deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
D_inv_sqrt = torch.diag(deg_inv_sqrt)
A_norm = D_inv_sqrt @ A @ D_inv_sqrt

print("A_norm shape:", A_norm.shape)

# ============================================
# 5. ST-GCN Model Definition
# ============================================
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size, A):
        super().__init__()
        self.A = A
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1,1))
        self.temporal_conv = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_kernel_size // 2)
        )

    def forward(self, x):
        x_spatial = torch.einsum('ij,bcjt->bcit', self.A, x)
        x_spatial = self.spatial_conv(x_spatial)
        x_temporal = self.temporal_conv(x_spatial)
        return F.relu(x_temporal)

class STGCN(nn.Module):
    def __init__(self, A):
        super().__init__()
        self.block1 = STGCNBlock(1, 16, temporal_kernel_size=3, A=A)
        self.block2 = STGCNBlock(16, 32, temporal_kernel_size=3, A=A)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1,1)),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.final_conv(x)
        return x

criterion = nn.MSELoss()

def safe_mape(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.maximum(np.abs(y_true), eps)
    return np.mean(np.abs((y_true - y_pred) / denom))

def eval_metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = safe_mape(y_true, y_pred)
    evs = explained_variance_score(y_true, y_pred)
    return rmse, mae, r2, mape, evs

# ============================================
# 6. Train/Test Split (Node-level 80/20)
# ============================================
node_indices = np.arange(num_nodes)
train_nodes, test_nodes = train_test_split(node_indices, test_size=0.2, random_state=seed)

train_mask_global = torch.zeros(num_nodes, dtype=torch.bool)
test_mask_global = torch.zeros(num_nodes, dtype=torch.bool)
train_mask_global[train_nodes] = True
test_mask_global[test_nodes] = True

print(f"Train nodes: {len(train_nodes)}, Test nodes: {len(test_nodes)}")

# ============================================
# 7. 5-Fold Cross-Validation (Node-level)
# ============================================
kf = KFold(n_splits=5, shuffle=True, random_state=seed)

lr = 0.01
num_epochs = 200

best_rmse_val = float("inf")
best_state = None
fold_metrics = []

for fold, (tr_sub, va_sub) in enumerate(kf.split(train_nodes), start=1):
    tr_idx = train_nodes[tr_sub]
    va_idx = train_nodes[va_sub]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[tr_idx] = True
    val_mask[va_idx] = True

    model = STGCN(A_norm)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        out = model(data_tensor)[0, 0, :, :]
        tgt = target_tensor[0, 0, :, :]

        train_mask_T = train_mask.unsqueeze(1).expand_as(tgt)
        loss = criterion(out[train_mask_T], tgt[train_mask_T])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data_tensor)[0, 0, :, :].cpu().numpy()
    true = target_tensor[0, 0, :, :].cpu().numpy()

    val_mask_T = val_mask.unsqueeze(1).expand_as(target_tensor[0, 0, :, :]).cpu().numpy()
    preds_val = pred[val_mask_T]
    trues_val = true[val_mask_T]

    rmse, mae, r2, mape, evs = eval_metrics(trues_val, preds_val)
    fold_metrics.append((rmse, mae, r2, mape, evs))

    if rmse < best_rmse_val:
        best_rmse_val = rmse
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

# ============================================
# 8. Test Evaluation
# ============================================
best_model = STGCN(A_norm)
best_model.load_state_dict(best_state)
best_model.eval()

with torch.no_grad():
    pred_all = best_model(data_tensor)[0, 0, :, :].cpu().numpy()
true_all = target_tensor[0, 0, :, :].cpu().numpy()

test_mask_T = test_mask_global.unsqueeze(1).expand_as(target_tensor[0, 0, :, :]).cpu().numpy()
preds_test = pred_all[test_mask_T]
trues_test = true_all[test_mask_T]

rmse_t, mae_t, r2_t, mape_t, evs_t = eval_metrics(trues_test, preds_test)

print("Test Performance:")
print(f"RMSE={rmse_t:.4f}, MAE={mae_t:.4f}, R2={r2_t:.4f}, MAPE={mape_t:.4f}, EVS={evs_t:.4f}")

# ============================================
# 9. Save Forecast Output (Wide Format)
# ============================================
if use_log1p:
    pred_all_out = np.expm1(pred_all)
    pred_all_out[pred_all_out < 0] = 0.0
else:
    pred_all_out = pred_all

pred_wide = pd.DataFrame({grid_id_col: node_ids})
true_wide = pd.DataFrame({grid_id_col: node_ids})

for t in range(T_out):
    in_m = month_names[t]
    out_m = month_names[t+1]
    pred_wide[f"pred_{out_m}_from_{in_m}"] = pred_all_out[:, t]
    true_wide[f"true_{out_m}"] = data[:, t+1]

# Save inside outputs/ folder
out_pred = "outputs/cook_stgcn_forecast_predictions_wide.csv"
out_true = "outputs/cook_stgcn_forecast_true_wide.csv"

pred_wide.to_csv(out_pred, index=False, float_format="%.6f")
true_wide.to_csv(out_true, index=False, float_format="%.6f")

print("Saved prediction and true CSV files in outputs/ folder.")
