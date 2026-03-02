"""
Cross-region ST-GCN (Cook -> Milwaukee) one-step-ahead forecasting.

- Task: y_t -> y_{t+1} (monthly)
- Train: Cook County only
  - Node-based 80/20 split (within Cook)
  - 5-fold CV within Cook training nodes
  - Best fold selected by validation RMSE
- Test: Milwaukee (no training)
  - Apply Cook-trained weights to Milwaukee using Milwaukee-specific rook adjacency

"""

import os
import random
from math import sqrt

import numpy as np
import pandas as pd
import geopandas as gpd

import torch
import torch.nn as nn
import torch.nn.functional as F

from libpysal.weights import Rook
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
)


# ================================
# 0) Reproducibility
# ================================
def set_seed(seed: int = 2025) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ================================
# 1) Metrics
# ================================
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


# ================================
# 2) Data utilities
# ================================
def build_panel_tensors(gdf, grid_id_col, month_cols, use_log1p=True):
    """
    Returns:
      gdf_sorted
      node_ids
      data_raw (N,T) raw scale
      X (1,1,N,T-1) model scale
      Y (1,1,N,T-1) model scale
    """
    gdf_sorted = gdf.sort_values(grid_id_col).reset_index(drop=True)
    node_ids = gdf_sorted[grid_id_col].values

    data = gdf_sorted[month_cols].values.astype(np.float32)
    data = np.nan_to_num(data)

    X_in = data[:, :-1]  # (N,T-1)
    Y_out = data[:, 1:]  # (N,T-1)

    if use_log1p:
        X_in = np.log1p(X_in)
        Y_out = np.log1p(Y_out)

    X = torch.tensor(X_in, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    Y = torch.tensor(Y_out, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return gdf_sorted, node_ids, data, X, Y


def build_rook_A_norm(gdf_sorted, grid_id_col):
    """
    Rook contiguity adjacency with symmetric normalization: D^{-1/2} A D^{-1/2}
    """
    rook_w = Rook.from_dataframe(gdf_sorted, ids=gdf_sorted[grid_id_col])
    node_ids = gdf_sorted[grid_id_col].values
    num_nodes = len(node_ids)

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

    deg = A.sum(dim=1)
    deg_inv_sqrt = torch.pow(deg, -0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A @ D_inv_sqrt


# ================================
# 3) Model 
# ================================
    
class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size):
        super().__init__()
        self.spatial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        self.temporal_conv = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(1, temporal_kernel_size),
            padding=(0, temporal_kernel_size // 2),
        )

    def forward(self, x, A_norm):
        # x: (B,C,N,T), A_norm: (N,N)
        x_spatial = torch.einsum("ij,bcjt->bcit", A_norm, x)
        x_spatial = self.spatial_conv(x_spatial)
        x_temporal = self.temporal_conv(x_spatial)
        return F.relu(x_temporal)


class STGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = STGCNBlock(1, 16, temporal_kernel_size=3)
        self.block2 = STGCNBlock(16, 32, temporal_kernel_size=3)
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=(1, 1)),
            nn.ReLU(),
        )

    def forward(self, x, A_norm):
        x = self.block1(x, A_norm)
        x = self.block2(x, A_norm)
        x = self.final_conv(x)
        return x


# ================================
# 4) Main
# ================================
def main():
    # ---- config ----
    seed = 2025
    set_seed(seed)

    use_log1p = True
    lr = 0.01
    num_epochs = 200
    n_splits = 5

    run_cook_internal_test = False 

    # ---- paths  ----
    cook_shp = os.path.join("data", "Cook_1km_panel_wide.shp")
    mke_shp = os.path.join("data", "Milwaukee_1km_panel_wide.shp")
    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)

    # ---- columns ----
    grid_id_col = "TARGET_FID"
    month_cols_candidate = [
        "Month_1", "Month_2", "Month_3", "Month_4",
        "Month_5", "Month_6", "Month_7", "Month_8",
        "Month_9", "Month_10", "Month_11", "Month_12",
    ]

    # ---- load ----
    if not os.path.exists(cook_shp):
        raise FileNotFoundError(f"Cook shapefile not found: {cook_shp}")
    if not os.path.exists(mke_shp):
        raise FileNotFoundError(f"Milwaukee shapefile not found: {mke_shp}")

    gdf_cook = gpd.read_file(cook_shp)
    gdf_mke = gpd.read_file(mke_shp)

    # common month columns
    month_cols = [c for c in month_cols_candidate if (c in gdf_cook.columns) and (c in gdf_mke.columns)]
    if len(month_cols) < 2:
        raise ValueError("Cook/Milwaukee must share at least two monthly columns for forecasting.")
    print(f"[INFO] Common monthly columns ({len(month_cols)}): {month_cols}")

    # build tensors
    cook_sorted, cook_node_ids, cook_raw, cook_X, cook_Y = build_panel_tensors(
        gdf_cook, grid_id_col, month_cols, use_log1p=use_log1p
    )
    mke_sorted, mke_node_ids, mke_raw, mke_X, mke_Y = build_panel_tensors(
        gdf_mke, grid_id_col, month_cols, use_log1p=use_log1p
    )

    # adjacency
    A_cook = build_rook_A_norm(cook_sorted, grid_id_col)
    A_mke = build_rook_A_norm(mke_sorted, grid_id_col)

    print(f"[INFO] Cook X: {tuple(cook_X.shape)}, Cook Y: {tuple(cook_Y.shape)}, A_cook: {tuple(A_cook.shape)}")
    print(f"[INFO] MKE  X: {tuple(mke_X.shape)},  MKE  Y: {tuple(mke_Y.shape)},  A_mke:  {tuple(A_mke.shape)}")

    # ---- train on Cook  ----
    criterion = nn.MSELoss()

    num_nodes_cook = cook_X.shape[2]
    node_indices = np.arange(num_nodes_cook)

    train_nodes, test_nodes_internal = train_test_split(node_indices, test_size=0.2, random_state=seed)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    best_rmse_val = float("inf")
    best_state = None

    for fold, (tr_sub, va_sub) in enumerate(kf.split(train_nodes), start=1):
        tr_idx = train_nodes[tr_sub]
        va_idx = train_nodes[va_sub]

        train_mask = torch.zeros(num_nodes_cook, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes_cook, dtype=torch.bool)
        train_mask[tr_idx] = True
        val_mask[va_idx] = True

        model = STGCN()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        for _epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            out = model(cook_X, A_cook)[0, 0, :, :]  # (N, T-1)
            tgt = cook_Y[0, 0, :, :]                 # (N, T-1)

            train_mask_T = train_mask.unsqueeze(1).expand_as(tgt)
            loss = criterion(out[train_mask_T], tgt[train_mask_T])
            loss.backward()
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            pred = model(cook_X, A_cook)[0, 0, :, :].cpu().numpy()
        true = cook_Y[0, 0, :, :].cpu().numpy()

        val_mask_T = val_mask.unsqueeze(1).expand_as(cook_Y[0, 0, :, :]).cpu().numpy()
        preds_val = pred[val_mask_T]
        trues_val = true[val_mask_T]

        rmse, mae, r2, mape, evs = eval_metrics(trues_val, preds_val)
        print(f"[CV] Cook Fold {fold}: RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f} MAPE={mape:.4f} EVS={evs:.4f}")

        if rmse < best_rmse_val:
            best_rmse_val = rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"\n[INFO] Best Cook CV RMSE (VAL): {best_rmse_val:.4f}")

    # final Cook-trained model
    best_model = STGCN()
    best_model.load_state_dict(best_state)
    best_model.eval()

    
    if run_cook_internal_test:
        with torch.no_grad():
            pred_cook = best_model(cook_X, A_cook)[0, 0, :, :].cpu().numpy()
        true_cook = cook_Y[0, 0, :, :].cpu().numpy()

        test_mask = torch.zeros(num_nodes_cook, dtype=torch.bool)
        test_mask[test_nodes_internal] = True
        test_mask_T = test_mask.unsqueeze(1).expand_as(cook_Y[0, 0, :, :]).cpu().numpy()

        rmse_t, mae_t, r2_t, mape_t, evs_t = eval_metrics(true_cook[test_mask_T], pred_cook[test_mask_T])
        print(f"[TEST] Cook internal (node-holdout) RMSE={rmse_t:.4f} MAE={mae_t:.4f} R2={r2_t:.4f} MAPE={mape_t:.4f} EVS={evs_t:.4f}")

    # ---- cross-region test on Milwaukee ----
    with torch.no_grad():
        pred_mke = best_model(mke_X, A_mke)[0, 0, :, :].cpu().numpy()
    true_mke = mke_Y[0, 0, :, :].cpu().numpy()

    rmse_m, mae_m, r2_m, mape_m, evs_m = eval_metrics(true_mke.flatten(), pred_mke.flatten())
    print("\n[TEST] Cross-region: Train=Cook, Test=Milwaukee")
    print(f"[TEST] RMSE={rmse_m:.4f} MAE={mae_m:.4f} R2={r2_m:.4f} MAPE={mape_m:.4f} EVS={evs_m:.4f}")

    # ---- save wide outputs (Milwaukee) ----
    if use_log1p:
        pred_mke_out = np.expm1(pred_mke)
        pred_mke_out[pred_mke_out < 0] = 0.0
    else:
        pred_mke_out = pred_mke

    T_out = pred_mke_out.shape[1]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][: (T_out + 1)]

    pred_wide = pd.DataFrame({grid_id_col: mke_node_ids})
    true_wide = pd.DataFrame({grid_id_col: mke_node_ids})

    for t in range(T_out):
        in_m = month_names[t]
        out_m = month_names[t + 1]
        pred_wide[f"pred_{out_m}_from_{in_m}"] = pred_mke_out[:, t]
        true_wide[f"true_{out_m}"] = mke_raw[:, t + 1]  

    out_pred = os.path.join(out_dir, "MKE_pred_from_Cook_weights_wide.csv")
    out_true = os.path.join(out_dir, "MKE_true_wide.csv")

    pred_wide.to_csv(out_pred, index=False, float_format="%.6f")
    true_wide.to_csv(out_true, index=False, float_format="%.6f")

    print(f"\n[INFO] Saved predictions: {out_pred}")
    print(f"[INFO] Saved true values: {out_true}")


if __name__ == "__main__":
    main()
