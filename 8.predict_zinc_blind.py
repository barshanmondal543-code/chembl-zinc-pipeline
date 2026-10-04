# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import numpy as np
import pandas as pd

# --- Cheminformatics / ML stack (same libraries as Scripts 2 and 3/v2) ---
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, DataStructs

from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ==============================================================================
# SCRIPT 8 — BLIND PREDICTION  (aligned to molecular_pipeline_code-v2.py)
#
# Trains ONE of the models from the v2 pipeline on the labelled ChEMBL feature
# matrix (Script 2 output) and blind-predicts activity for the unlabelled ZINC
# library (Script 7 output, SMILES + ZINC_ID only).
#
# What is kept IDENTICAL to v2 so the deployed model sees exactly the same
# feature space it was designed around:
#   * same sheet, same target, same train/test split (RANDOM_SEED)
#   * same variance filter
#   * same multicollinearity pruning  (Pearson |r| > CORR_THRESHOLD = 0.9)
#   * same RFECV + averaged RF/XGB importance selection of the TOP_N_FEATURES (30)
#   * same regularized model definitions
# The selection chain is re-run here (not hard-coded) with v2's exact settings,
# so the SAME features and the SAME number of features are identified, then only
# those features are computed for the ZINC molecules and fed to the model.
# ==============================================================================


# ==============================================================================
# CONFIGURATION BLOCK  (mirrors molecular_pipeline_code-v2.py)
# ==============================================================================
# --- Training data (Script 2 output) ---
TRAINING_EXCEL = "generated/molecular_camble_database_for_ic50.xlsx"

# Feature sheet — MUST match a sheet written by Script 2 and should be the SAME
# one you used in v2 so the selected features exist:
#   "Master Combined Matrix" [RECOMMENDED] | "Physicochemical Descriptors"
#   | "MACCS Keys" | "Morgan Fingerprints"
TARGET_SHEET_NAME = "Physicochemical Descriptors"

# Target endpoint column (same label as v2).
TARGET_COLUMN = "pIC50 (-log10(M))"

# --- Feature-selection parameters — keep identical to v2 -----------------------
TOP_N_FEATURES = 30       # number of top features to retain (v2: 30)
CORR_THRESHOLD = 0.9     # Pearson |r| multicollinearity cut-off (v2: 0.9)
TEST_RATIO = 0.25         # train/test split (v2)
CV_FOLDS = 5              # cross-validation folds (v2)
RANDOM_SEED = 42          # reproducibility — MUST equal v2 to reproduce the selection

# ------------------------------------------------------------------------------
# >>> MODEL SELECTION <<<  — which trained model performs the blind prediction.
# Set MODEL_NAME to EXACTLY ONE of the strings below (identical to v2's models):
#
#   "Linear Regression"    -> Multiple Linear Regression (LinearRegression)
#   "Ridge Regression"     -> Ridge (alpha=10.0)
#   "Lasso Regression"     -> Lasso (alpha=0.02)
#   "Random Forest (Reg)"  -> RandomForestRegressor (200 trees, depth 8, sqrt)
#   "XGBoost (Reg)"        -> XGBRegressor (150 trees, regularized)
#
# Change only this one line to switch models.
# ------------------------------------------------------------------------------
MODEL_NAME = "XGBoost (Reg)"

# --- Blind set (Script 7 output, training molecules screened out) ---
ZINC_INPUT = "generated/ZINC20_screened_for_ic50.csv"
ZINC_SMILES_COL = "SMILES"
ZINC_ID_COL = "ZINC_ID"
MAX_MOLECULES = None            # cap ZINC rows scored (int) or None for all
ZINC_CHUNKSIZE = 5000           # stream the (large) ZINC file in chunks

# --- Feature parameters (MUST match Script 2 exactly) ---
MORGAN_RADIUS = 2
MORGAN_BITS = 2048

# --- Final deployment model ---
# True  -> after the features are identified on the training split (leakage-free,
#          exactly as v2), retrain the chosen model on ALL labelled compounds for
#          the actual ZINC prediction. The FEATURE SET stays identical either way.
# False -> deploy the model trained only on v2's 75% training split.
RETRAIN_ON_FULL_DATA = True
SHOW_HOLDOUT_METRICS = True     # echo v2-style 5-fold CV + holdout test scores

# --- Applicability domain (AD), computed in the SELECTED-feature space ---
APPLICABILITY_DOMAIN = True
AD_K = 5
AD_PERCENTILE = 95

# --- Output ---
OUTPUT_DIR = "generated"
OUTPUT_CSV = f"{OUTPUT_DIR}/blind_prediction_final.csv"
SELECTED_FEATURES_TXT = f"{OUTPUT_DIR}/blind_selected_features.txt"
TOP_N_PRINT = 20
# ==============================================================================


RDLogger.DisableLog("rdApp.*")
DESCRIPTOR_NAMES_2D = [name for name, _ in Descriptors._descList]


# ------------------------------------------------------------------------------
# v2's multicollinearity filter — copied verbatim so results are identical.
# ------------------------------------------------------------------------------
def remove_multicollinear_features(df, threshold=0.85):
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop), to_drop


# ------------------------------------------------------------------------------
# Feature computation for a single molecule — mirrors Script 2 bit-for-bit.
# ------------------------------------------------------------------------------
def featurize_mol(mol, need_morgan, need_maccs, need_desc):
    feats = {}
    if need_morgan:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
        arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(bv, arr)
        for i in range(MORGAN_BITS):
            feats[f"morgan_bit_{i}"] = int(arr[i])
    if need_maccs:
        maccs_bv = MACCSkeys.GenMACCSKeys(mol)
        maccs_arr = np.zeros((1,), dtype=int)
        DataStructs.ConvertToNumpyArray(maccs_bv, maccs_arr)
        maccs_arr = maccs_arr[1:]  # drop dummy bit 0, exactly like Script 2
        for i in range(1, 167):
            feats[f"maccs_bit_{i}"] = int(maccs_arr[i - 1])
    if need_desc:
        for name in DESCRIPTOR_NAMES_2D:
            try:
                feats[name] = getattr(Descriptors, name)(mol)
            except Exception:
                feats[name] = np.nan
    return feats


def featurize_smiles_frame(smiles_series, id_series, need_morgan, need_maccs, need_desc):
    rows, kept_ids, kept_smiles, n_invalid = [], [], [], 0
    for smi, mol_id in zip(smiles_series, id_series):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            n_invalid += 1
            continue
        rows.append(featurize_mol(mol, need_morgan, need_maccs, need_desc))
        kept_ids.append(mol_id)
        kept_smiles.append(Chem.MolToSmiles(mol))
    return pd.DataFrame(rows), kept_ids, kept_smiles, n_invalid


# ------------------------------------------------------------------------------
# Training-matrix assembly — identical leak-column logic to v2.
# ------------------------------------------------------------------------------
def build_training_matrix(sheet_df, target_column):
    if target_column not in sheet_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in sheet '{TARGET_SHEET_NAME}'."
        )
    df = sheet_df.replace("NaN", np.nan)
    df = df.dropna(subset=[target_column]).copy()

    non_feature_cols = ["molecule_chembl_id", "2D_Structure", "canonical_smiles"]
    activity_leak_cols = [c for c in df.columns if "(" in c or c.startswith("p")]
    leak_cols = list(set(non_feature_cols + activity_leak_cols))

    X_raw = df.drop(columns=leak_cols, errors="ignore")
    X_raw = X_raw.apply(pd.to_numeric, errors="coerce")
    y = df[target_column].astype(float).values
    return X_raw, y


def detect_feature_families(feature_names):
    need_morgan = any(str(c).startswith("morgan_bit_") for c in feature_names)
    need_maccs = any(str(c).startswith("maccs_bit_") for c in feature_names)
    need_desc = any(
        not (str(c).startswith("morgan_bit_") or str(c).startswith("maccs_bit_"))
        for c in feature_names
    )
    return need_morgan, need_maccs, need_desc


def make_model(name):
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Lasso Regression": Lasso(alpha=0.02),
        "Random Forest (Reg)": RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=3,
            max_features="sqrt", random_state=RANDOM_SEED, n_jobs=-1
        ),
        "XGBoost (Reg)": XGBRegressor(
            n_estimators=150, max_depth=4, learning_rate=0.03, min_child_weight=5,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, subsample=0.8,
            colsample_bytree=0.8, random_state=RANDOM_SEED, n_jobs=-1
        ),
    }
    if name not in models:
        raise ValueError(f"Unknown MODEL_NAME '{name}'. Choose one of: {list(models)}")
    return models[name]


# ------------------------------------------------------------------------------
# Reproduce v2's feature selection EXACTLY -> the definitive feature list.
# ------------------------------------------------------------------------------
def select_features_like_v2(X_raw, y):
    """Returns (selected_feature_names, artefacts dict) using v2's identical
    train-split-only variance -> correlation -> RFECV -> top-N chain."""
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_RATIO, random_state=RANDOM_SEED
    )

    # Variance filter (train fold only)
    variance_mask = X_train_raw.var(axis=0) > 0
    X_train_filter = X_train_raw.loc[:, variance_mask]

    # Multicollinearity pruning (train fold only), Pearson |r| > CORR_THRESHOLD
    X_train_no_corr, dropped_corr_cols = remove_multicollinear_features(
        X_train_filter, threshold=CORR_THRESHOLD
    )
    filtered_feature_names = np.array(X_train_no_corr.columns)

    # Impute + scale (fit on train fold)
    imp_sel = SimpleImputer(strategy="median").fit(X_train_no_corr)
    X_train_imp = imp_sel.transform(X_train_no_corr)
    sc_sel = StandardScaler().fit(X_train_imp)
    X_train_scaled = sc_sel.transform(X_train_imp)

    # RFECV feature selection
    rf_selector = RandomForestRegressor(
        n_estimators=100, max_depth=8, min_samples_leaf=3,
        random_state=RANDOM_SEED, n_jobs=-1
    )
    rfecv = RFECV(
        estimator=rf_selector, step=5,
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="r2",
        min_features_to_select=min(TOP_N_FEATURES, X_train_scaled.shape[1]),
        n_jobs=-1,
    )
    rfecv.fit(X_train_scaled, y_train)

    # Averaged RF + XGB importance on the RFECV-supported columns
    rf_selector.fit(X_train_scaled[:, rfecv.support_], y_train)
    xgb_selector = XGBRegressor(
        n_estimators=100, max_depth=4, learning_rate=0.03,
        min_child_weight=5, random_state=RANDOM_SEED, n_jobs=-1
    )
    xgb_selector.fit(X_train_scaled[:, rfecv.support_], y_train)

    avg_imp = (rf_selector.feature_importances_ + xgb_selector.feature_importances_) / 2.0
    top_idx_within_selected = np.argsort(avg_imp)[::-1][:TOP_N_FEATURES]
    selected_indices = np.where(rfecv.support_)[0][top_idx_within_selected]
    selected_feature_names = list(filtered_feature_names[selected_indices])
    top_importances = avg_imp[top_idx_within_selected]

    artefacts = {
        "X_train_raw": X_train_raw, "X_test_raw": X_test_raw,
        "y_train": y_train, "y_test": y_test,
        "variance_mask": variance_mask, "dropped_corr_cols": dropped_corr_cols,
        "filtered_feature_names": filtered_feature_names,
        "imp_sel": imp_sel, "sc_sel": sc_sel,
        "X_train_scaled": X_train_scaled, "selected_indices": selected_indices,
        "top_importances": top_importances,
        "n_after_corr": len(filtered_feature_names),
    }
    return selected_feature_names, artefacts


def holdout_report(art, selected_indices, model_name):
    """Reproduce v2's 5-fold CV + holdout test metrics for the chosen model on
    the selected features (same split, same transforms, same model)."""
    X_train_top = art["X_train_scaled"][:, selected_indices]

    X_test_filter = art["X_test_raw"].loc[:, art["variance_mask"]]
    X_test_no_corr = X_test_filter.drop(columns=art["dropped_corr_cols"], errors="ignore")
    X_test_no_corr = X_test_no_corr[art["filtered_feature_names"]]
    X_test_scaled = art["sc_sel"].transform(art["imp_sel"].transform(X_test_no_corr))
    X_test_top = X_test_scaled[:, selected_indices]

    model = make_model(model_name)
    cv = cross_validate(
        model, X_train_top, art["y_train"],
        cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
        scoring="r2", n_jobs=-1,
    )
    model.fit(X_train_top, art["y_train"])
    test_preds = model.predict(X_test_top)
    return {
        "cv_r2_mean": float(np.mean(cv["test_score"])),
        "cv_r2_std": float(np.std(cv["test_score"])),
        "test_r2": float(r2_score(art["y_test"], test_preds)),
        "test_mae": float(mean_absolute_error(art["y_test"], test_preds)),
        "test_rmse": float(np.sqrt(mean_squared_error(art["y_test"], test_preds))),
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1. Load labelled training matrix -------------------------------------
    if not os.path.exists(TRAINING_EXCEL):
        raise FileNotFoundError(f"Training feature file not found: {TRAINING_EXCEL}")
    print(f"Loading training sheet '{TARGET_SHEET_NAME}' from '{TRAINING_EXCEL}'...")
    sheet_df = pd.read_excel(TRAINING_EXCEL, sheet_name=TARGET_SHEET_NAME)
    X_raw, y = build_training_matrix(sheet_df, TARGET_COLUMN)
    print(f"  Labelled compounds: {X_raw.shape[0]} | candidate feature columns: {X_raw.shape[1]}")

    # ---- 2. Reproduce v2 feature selection (identical settings) ---------------
    print(f"\nReproducing v2 feature selection "
          f"(|r|>{CORR_THRESHOLD} pruning, RFECV, top {TOP_N_FEATURES})...")
    selected_feature_names, art = select_features_like_v2(X_raw, y)
    print(f"  Features after variance + correlation pruning: {art['n_after_corr']} "
          f"(dropped {len(art['dropped_corr_cols'])} collinear @ |r|>{CORR_THRESHOLD})")
    print(f"  Final selected features: {len(selected_feature_names)}")

    # Persist the exact feature list used for the blind prediction.
    with open(SELECTED_FEATURES_TXT, "w", encoding="utf-8") as fh:
        fh.write(f"# Blind-prediction feature set (identical to v2 selection)\n")
        fh.write(f"# sheet={TARGET_SHEET_NAME} target={TARGET_COLUMN} "
                 f"corr_threshold={CORR_THRESHOLD} top_n={TOP_N_FEATURES} seed={RANDOM_SEED}\n")
        for rank, (feat, imp) in enumerate(zip(selected_feature_names, art["top_importances"]), 1):
            fh.write(f"{rank:02d}\t{feat}\timportance={imp:.6f}\n")
    print(f"  Feature list written to: {SELECTED_FEATURES_TXT}")

    need_morgan, need_maccs, need_desc = detect_feature_families(selected_feature_names)
    fam = [n for n, on in (("Morgan", need_morgan), ("MACCS", need_maccs),
                           ("2D-descriptors", need_desc)) if on]
    print(f"  Feature families to compute for ZINC: {', '.join(fam)}")

    # ---- 3. (Optional) v2-style honest evaluation of the chosen model ---------
    if SHOW_HOLDOUT_METRICS:
        m = holdout_report(art, art["selected_indices"], MODEL_NAME)
        print(f"\nHoldout evaluation of '{MODEL_NAME}' on the selected features "
              f"(same 75/25 split as v2):")
        print(f"  5-fold CV R2: {m['cv_r2_mean']:.3f} (+/- {m['cv_r2_std']:.3f}) | "
              f"Test R2: {m['test_r2']:.3f} | Test MAE: {m['test_mae']:.3f} | "
              f"Test RMSE: {m['test_rmse']:.3f}")

    # ---- 4. Build the deployment pipeline on the SELECTED features ------------
    if RETRAIN_ON_FULL_DATA:
        X_fit_df = X_raw[selected_feature_names]
        y_fit = y
        fit_note = "all labelled compounds"
    else:
        X_fit_df = art["X_train_raw"][selected_feature_names]
        y_fit = art["y_train"]
        fit_note = "v2 75% training split"

    X_fit_arr = X_fit_df.values
    imputer = SimpleImputer(strategy="median").fit(X_fit_arr)
    X_fit_imp = imputer.transform(X_fit_arr)
    scaler = StandardScaler().fit(X_fit_imp)
    X_fit_scaled = scaler.transform(X_fit_imp)

    model = make_model(MODEL_NAME)
    print(f"\nTraining deployment model '{MODEL_NAME}' on {fit_note} "
          f"({len(selected_feature_names)} features)...")
    model.fit(X_fit_scaled, y_fit)

    # ---- 5. Applicability-domain reference (selected-feature space) -----------
    ad_nn = ad_threshold = ad_k_query = None
    if APPLICABILITY_DOMAIN and X_fit_scaled.shape[0] > 1:
        k = min(AD_K, X_fit_scaled.shape[0] - 1)
        ad_nn = NearestNeighbors(n_neighbors=k + 1).fit(X_fit_scaled)
        d, _ = ad_nn.kneighbors(X_fit_scaled)
        thr = np.percentile(d[:, 1:].mean(axis=1), AD_PERCENTILE)
        ad_threshold, ad_k_query = thr, k
        print(f"  Applicability domain: k={k}, in-domain threshold "
              f"(<= P{AD_PERCENTILE}) = {thr:.3f}")

    # ---- 6. Stream, featurize and predict the ZINC set ------------------------
    if not os.path.exists(ZINC_INPUT):
        raise FileNotFoundError(f"ZINC dataset not found: {ZINC_INPUT}")
    print(f"\nScoring ZINC molecules from '{ZINC_INPUT}' (chunk size {ZINC_CHUNKSIZE})...")
    is_p_target = TARGET_COLUMN.strip().startswith("p")

    results = []
    n_seen = n_scored = n_invalid = 0
    for chunk in pd.read_csv(ZINC_INPUT, chunksize=ZINC_CHUNKSIZE):
        if ZINC_SMILES_COL not in chunk.columns:
            raise ValueError(f"Column '{ZINC_SMILES_COL}' not in {ZINC_INPUT}. "
                             f"Found: {list(chunk.columns)}")
        if ZINC_ID_COL not in chunk.columns:
            chunk[ZINC_ID_COL] = [f"ROW_{n_seen + j}" for j in range(len(chunk))]

        if MAX_MOLECULES is not None:
            remaining = MAX_MOLECULES - n_seen
            if remaining <= 0:
                break
            if remaining < len(chunk):
                chunk = chunk.iloc[:remaining]
        n_seen += len(chunk)

        feat_df, kept_ids, kept_smiles, inv = featurize_smiles_frame(
            chunk[ZINC_SMILES_COL], chunk[ZINC_ID_COL],
            need_morgan, need_maccs, need_desc,
        )
        n_invalid += inv
        if feat_df.empty:
            continue

        # Align to the SELECTED features (identical to v2), then transform.
        aligned = feat_df.reindex(columns=selected_feature_names)
        aligned = aligned.apply(pd.to_numeric, errors="coerce")
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        X_zinc = scaler.transform(imputer.transform(aligned.values))

        preds = model.predict(X_zinc)

        ad_status = ad_dist = None
        if APPLICABILITY_DOMAIN and ad_nn is not None:
            dd, _ = ad_nn.kneighbors(X_zinc, n_neighbors=ad_k_query)
            ad_dist = dd.mean(axis=1)
            ad_status = np.where(ad_dist <= ad_threshold, "In-Domain", "Out-of-Domain")

        for j in range(len(preds)):
            rec = {
                ZINC_ID_COL: kept_ids[j],
                "canonical_smiles": kept_smiles[j],
                f"predicted_{TARGET_COLUMN}": float(preds[j]),
            }
            if is_p_target:
                rec["predicted_IC50 (nM)"] = float(10.0 ** (9.0 - preds[j]))
            if APPLICABILITY_DOMAIN and ad_status is not None:
                rec["applicability_domain"] = ad_status[j]
                rec["AD_mean_dist"] = float(ad_dist[j])
            results.append(rec)
            n_scored += 1

        print(f"  ...scored {n_scored} (scanned {n_seen}, invalid {n_invalid})", end="\r")
    print()

    if not results:
        print("No molecules scored — check the ZINC file / SMILES column.")
        return

    # ---- 7. Rank and export ---------------------------------------------------
    out = pd.DataFrame(results)
    sort_col = f"predicted_{TARGET_COLUMN}"
    ascending = not is_p_target  # higher pIC50 = more potent
    out = out.sort_values(by=sort_col, ascending=ascending).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    out.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 78)
    print("BLIND PREDICTION COMPLETE")
    print("=" * 78)
    print(f"  Model              : {MODEL_NAME}")
    print(f"  Features used      : {len(selected_feature_names)} (v2-identical selection)")
    print(f"  Molecules scanned  : {n_seen}")
    print(f"  Invalid SMILES     : {n_invalid}")
    print(f"  Molecules scored   : {n_scored}")
    if APPLICABILITY_DOMAIN and ad_nn is not None:
        in_dom = int((out["applicability_domain"] == "In-Domain").sum())
        print(f"  In-domain hits     : {in_dom} / {n_scored}")
    print(f"  Predicted {sort_col}: min={out[sort_col].min():.2f}, "
          f"max={out[sort_col].max():.2f}, mean={out[sort_col].mean():.2f}")
    print(f"  Output written to  : {OUTPUT_CSV}")

    show_cols = ["rank", ZINC_ID_COL, sort_col]
    if "predicted_IC50 (nM)" in out.columns:
        show_cols.append("predicted_IC50 (nM)")
    if APPLICABILITY_DOMAIN and ad_nn is not None:
        show_cols.append("applicability_domain")
    print(f"\nTop {min(TOP_N_PRINT, len(out))} predicted compounds:")
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(out[show_cols].head(TOP_N_PRINT).to_string(index=False))


if __name__ == "__main__":
    main()
