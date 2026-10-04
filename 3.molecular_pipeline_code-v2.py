# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# RDKit imports for backmapping fingerprint bits
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# ==============================================================================
# CONFIGURATION BLOCK: DATASET & MODEL PARAMETERS
# ==============================================================================
INPUT_FILENAME = "generated/molecular_camble_database_for_ic50.xlsx"

# 1. Feature Sheet Selection (Uncomment the desired representation layer):
# - "Master Combined Matrix": Hybrid dataset (Physicochemical + MACCS + Morgan Fingerprints)
# - "Physicochemical Descriptors": 2D/3D macro physical property features only
# - "MACCS Keys": Structural 166-bit predefined key patterns
# - "Morgan Fingerprints": Radius-2 topological circular atom environments
TARGET_SHEET_NAME = "Physicochemical Descriptors"  # sheet used for the committed results

# 2. Target Endpoint Column to Predict
TARGET_COLUMNS = ["pIC50 (-log10(M))"]  

# 3. Maximum Number of Top Predictive Features to Retain Post-Selection
TOP_N_FEATURES = 30

# 4. Multicollinearity Threshold (|r| Pearson Correlation Coefficient)
CORR_THRESHOLD = 0.9

# 5. Model Validation & Reproducibility Settings
TEST_RATIO = 0.25
CV_FOLDS = 5
RANDOM_SEED = 42

# Output File Directories & Target File Paths
OUTPUT_DIR = "top_feature_of_ic50"
CHART_OUTPUT_PATH = f"{OUTPUT_DIR}/top_feature_importances_Physicochemical Descriptors.png"
FRAGMENT_IMAGE_PATH = f"{OUTPUT_DIR}/top_fingerprint_fragments_Physicochemical Descriptors.png"
TEXT_REPORT_PATH = f"{OUTPUT_DIR}/top_feature_explanations_Physicochemical Descriptors.txt"
# ==============================================================================

MACCS_DEFINITIONS = {
    1:  "Isotope", 2: "Group IVa, Va, VIa Period 4-6", 3: "Ge, As, Se, Sn, Sb, Te, Pb, Bi, Po",
    4:  "Actinide/Lanthanide", 5: "Group IVb-VIIb, VIII Period 4-6", 6: "Subhalogen (F, Cl, Br, I)", 7: "Group IIIa, IVa-VIIa (Period 2-3)",
    8:  "QCH4 (Q=heteroatom)", 9: "Heteroatom-Hydrogen (e.g., OH, NH)", 10: "Halogen-Heteroatom", 11: "Expression: 4M-Ring",
    12: "Other group XIV (Si, Ge, Sn, Pb)", 13: "ON(O)C (Nitro derivative)", 14: "S-O Bond", 15: "N-Bonds (7+ structural connections)",
    16: "Other valence", 17: "Expression: 3M-Ring", 18: "S-N Bond", 19: "7-membered or larger ring",
    20: "Expression: 8-membered or larger ring", 21: "Expression: 3M-heteroring", 22: "Expression: 3M-heteroring containing O, N, S",
    23: "NC(O)N (Urea derivative)", 24: "Expression: 4M-heteroring", 25: "Expression: 4M-heteroring containing O, N, S",
    26: "Aromatic Ring size > 6", 27: "Expression: 5M-Ring", 28: "Expression: 5M-heteroring", 29: "Expression: 5M-heteroring containing O, N, S",
    30: "Expression: 6M-Ring", 31: "Expression: 6M-heteroring", 32: "Expression: 6M-heteroring containing O, N, S",
    33: "Expression: 7M-Ring or larger", 34: "Expression: CH2=CH2 / Olefinic Alkene system", 35: "Expression: F, Cl, Br, I (Halogen)",
    36: "S-Heteroatom Bond", 37: "NC(O)O (Carbamate / Urethane moiety)", 38: "NC(O)C (Amide connection)", 39: "OS(O)O (Sulfator / Sulfonate group)",
    40: "S-C Bond", 41: "C=N Bond (Imine / Hydrazone matrix)", 42: "Expression: Keton / Aldehyde C=O structural cluster",
    43: "N-O Bond", 44: "Expression: Primary/Secondary/Tertiary Amine configurations", 45: "C=C Bond (Alkene link)",
    46: "Expression: Br (Bromine atom)", 47: "Sanitized expression: Ring junction or shared vertices", 48: "Expression: C-O Bond",
    49: "Expression: Charge indicator / Zwitterionic system", 50: "Expression: C-N Bond", 51: "Expression: C-S Bond",
    52: "Expression: N-N Bond", 53: "Expression: RO-O-R / Peroxide or dynamic Oxygen stack", 54: "Expression: O-O Bond",
    55: "Expression: PO4 / Phosphonate or Phosphorus oxide link", 56: "Expression: P-Heteroatom Bond", 57: "Expression: P-C Bond",
    58: "Expression: P-N Bond", 59: "Expression: P-O Bond", 60: "Expression: P-S Bond", 61: "Expression: Pyridine ring system",
    62: "Expression: Imidazole or Azole framework", 63: "Expression: Thiophene motif", 64: "Expression: Furan motif",
    65: "Expression: Pyrrole motif", 66: "Expression: Benzene core", 67: "Expression: Iodine atom", 68: "Expression: Chlorine atom",
    69: "Expression: Fluorine atom", 70: "Expression: Bromine atom", 71: "Expression: Secondary aliphatic carbon center",
    72: "Expression: Tertiary aliphatic carbon vertex", 73: "Expression: Quaternary carbon matrix node", 74: "Expression: CH3 group",
    75: "Expression: Structural Ester connector (C-O-C=O)", 76: "Expression: Carboxylic acid group (COOH)", 77: "Expression: Alkynes structural system (C#C)",
    78: "Expression: Nitrile/Cyano cluster (C#N)", 79: "Expression: Nitro functional structural node (NO2)",
    80: "Expression: Sulfonamide cluster (SO2N)", 81: "Expression: Sulfone or Sulfoxide segment (S=O)", 82: "Expression: Thiol derivative cluster (-SH)",
    83: "Expression: Ether connector (C-O-C)", 84: "Expression: Hydroxyl cluster (-OH)", 85: "Expression: Primary Amine (-NH2)",
    86: "Expression: Secondary Amine (-NH-)", 87: "Expression: Tertiary Amine (N clustered with 3 Carbons)", 88: "Expression: Ring atom count matching 3 nodes",
    89: "Expression: Ring atom count matching 4 nodes", 90: "Expression: Ring atom count matching 5 nodes", 91: "Expression: Ring atom count matching 6 nodes",
    92: "Expression: Ring atom count matching 7 nodes", 93: "Expression: Ring atom count matching 8 or greater nodes",
    94: "Expression: Core contains 1 or more halogen nodes", 95: "Expression: Core contains 2 or more halogen nodes",
    96: "Expression: Core contains 3 or more halogen nodes", 97: "Expression: Core contains 4 or more halogen nodes",
    98: "Expression: Molecule contains 1 or more O atoms", 99: "Expression: Molecule contains 2 or more O atoms",
    100: "Expression: Molecule contains 3 or more O atoms", 101: "Expression: Molecule contains 4 or more O atoms",
    102: "Expression: Molecule contains 1 or more N atoms", 103: "Expression: Molecule contains 2 or more N atoms",
    104: "Expression: Molecule contains 3 or more N atoms", 105: "Expression: Molecule contains 4 or more N atoms",
    106: "Expression: Molecule contains 1 or more S atoms", 107: "Expression: Molecule contains 2 or more S atoms",
    108: "Expression: Molecule contains 3 or more S atoms", 109: "Expression: Molecule contains 4 or more S atoms",
    110: "Expression: Molecule contains 1 or more Ring structures", 111: "Expression: Molecule contains 2 or more Ring structures",
    112: "Expression: Molecule contains 3 or more Ring structures", 113: "Expression: Molecule contains 4 or more Ring structures",
    114: "Expression: Heteroatom attached to an aromatic ring", 115: "Expression: CH3 attached to an oxygen atom (Methoxy or Ester)",
    116: "Expression: CH3 attached to a nitrogen atom", 117: "Expression: CH3 attached to a sulfur atom", 118: "Expression: CH3 attached to another carbon",
    119: "Expression: Fragment containing N=O", 120: "Expression: Fragment containing Aromatic Carbon-Nitrogen bond",
    121: "Expression: Fragment containing Aromatic Carbon-Oxygen bond", 122: "Expression: Fragment containing Aromatic Carbon-Sulfur bond",
    123: "Expression: O-Aromatic Ring atom link", 124: "Expression: N-Aromatic Ring atom link", 125: "Expression: S-Aromatic Ring atom link",
    126: "Expression: Aromatic ring node bound to another Aromatic node", 127: "Expression: Carbonyl cluster bound directly to Aromatic system",
    128: "Expression: Halogen bound directly to an Aromatic ring atom", 129: "Expression: Heteroatom nested inside an Aromatic ring system",
    130: "Expression: Dynamic Ring size identifier match", 131: "Expression: Ring system incorporating exactly two heteroatoms",
    132: "Expression: Carbon double bonded to Carbon (C=C)", 133: "Expression: Carbon double bonded to Oxygen (C=O)",
    134: "Expression: Carbon double bonded to Nitrogen (C=N)", 135: "Expression: Nitrogen double bonded to Nitrogen (N=N)",
    136: "Expression: Nitrogen double bonded to Oxygen (N=O)", 137: "Expression: Heteroatom double bonded to Heteroatom",
    138: "Expression: Heteroatom double bonded to Carbon", 139: "Expression: Carbon single bonded to Oxygen (C-O)",
    140: "Expression: Carbon single bonded to Nitrogen (C-N)", 141: "Expression: Carbon single bonded to Sulfur (C-S)",
    142: "Expression: Nitrogen single bonded to Nitrogen (N-N)", 143: "Expression: Nitrogen single bonded to Oxygen (N-O)",
    144: "Expression: Oxygen single bonded to Oxygen (O-O)", 145: "Expression: Heteroatom single bonded to Heteroatom",
    146: "Expression: Valency/Coordination check for internal lines", 147: "Expression: Aromatic atom bound to non-aromatic atom",
    148: "Expression: Aromatic ring containing 1 heteroatom", 149: "Expression: Aromatic ring containing 2 heteroatoms",
    150: "Expression: Aromatic ring containing 3 or more heteroatoms", 151: "Expression: Ring containing exactly 1 heteroatom",
    152: "Expression: Ring containing exactly 2 heteroatoms", 153: "Expression: Ring containing 3 or more heteroatoms",
    154: "Expression: Aromatic ring fused with another ring structure", 155: "Expression: Conjugated double bond system (C=C-C=C)",
    156: "Expression: Chiral center / Stereochemical node present", 157: "Expression: Alcohol functional cluster (-OH on aliphatic carbon)",
    158: "Expression: Phenol structural segment (-OH on aromatic carbon)", 159: "Expression: Carbonyl linked directly to Nitrogen (Amide framework)",
    160: "Expression: Carbonyl linked directly to Oxygen (Ester / Carboxylic acid)", 161: "Expression: Nitrogen linked to Aromatic carbon node",
    162: "Expression: Iodine atom present", 163: "Expression: Chlorine atom present", 164: "Expression: Fluorine atom present",
    165: "Expression: Bromine atom present", 166: "Expression: Fragments matching generic organic molecules"
}

def remove_multicollinear_features(df, threshold=0.85):
    """
    MULTICOLLINEARITY FILTER EXPLANATION:
    ----------------------------------------------------------------------------
    Multicollinearity occurs when two or more independent molecular descriptors 
    are highly linearly correlated (e.g., Pearson |r| > 0.85). 
    
    Why this is problematic:
    1. Redundancy & Noise: Features carrying near-identical information (like Molecular 
       Weight vs. Heavy Atom Count) confuse models and split feature importance values.
    2. Model Instability: Linear regressions become numerically unstable (inflated variance),
       and tree ensembles can arbitrarily pick one correlated feature over another.
    
    How this function works:
    Calculates pairwise absolute Pearson correlations across all descriptors, inspects
    the upper triangular correlation matrix, and drops redundant feature columns exceeding 
    the defined threshold while keeping the first occurrence.
    ----------------------------------------------------------------------------
    """
    corr_matrix = df.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    return df.drop(columns=to_drop), to_drop

print(f"Loading matrix sheet '{TARGET_SHEET_NAME}' from Excel file: '{INPUT_FILENAME}'...")
if not os.path.exists(INPUT_FILENAME):
    raise FileNotFoundError(f"Missing base matrix data sheet at path: {INPUT_FILENAME}")

df_raw_sheet = pd.read_excel(INPUT_FILENAME, sheet_name=TARGET_SHEET_NAME)
df_raw_sheet = df_raw_sheet.replace("NaN", np.nan)

for target_col in TARGET_COLUMNS:
    if target_col not in df_raw_sheet.columns:
        raise ValueError(f"Requested target column '{target_col}' not found in sheet '{TARGET_SHEET_NAME}'.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(TEXT_REPORT_PATH, "w", encoding="utf-8") as txt_report:
    txt_report.write("================================================================================\n")
    txt_report.write("       MOLECULAR REGRESSION FEATURE IMPORTANCE & MODEL EVALUATION REPORT\n")
    txt_report.write("================================================================================\n\n")
    txt_report.write(f"Source Excel Database Path: {INPUT_FILENAME}\n")
    txt_report.write(f"Target Feature Sheet Processed: {TARGET_SHEET_NAME}\n\n")

    for TARGET_COLUMN in TARGET_COLUMNS:
        print("\n" + "="*85)
        print(f"PROCESSING ENHANCED MACHINE LEARNING PIPELINE FOR TARGET: {TARGET_COLUMN}")
        print("="*85)
        
        df_clean = df_raw_sheet.dropna(subset=[TARGET_COLUMN]).copy()
        
        non_feature_cols = ["molecule_chembl_id", "2D_Structure", "canonical_smiles"]
        activity_leak_cols = [c for c in df_clean.columns if "(" in c or c.startswith("p")]
        leak_cols = list(set(non_feature_cols + activity_leak_cols))
        
        X_raw = df_clean.drop(columns=leak_cols, errors="ignore")
        y = df_clean[TARGET_COLUMN].values
        
        # Splitting BEFORE feature selection/filtering to prevent data leakage
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X_raw, y, test_size=TEST_RATIO, random_state=RANDOM_SEED
        )

        # Variance thresholding (drops zero-variance flat descriptors)
        variance_mask = X_train_raw.var(axis=0) > 0
        X_train_filter = X_train_raw.loc[:, variance_mask]
        X_test_filter = X_test_raw.loc[:, variance_mask]

        # Multicollinearity filtering calculated exclusively on training fold
        X_train_no_corr, dropped_corr_cols = remove_multicollinear_features(X_train_filter, threshold=CORR_THRESHOLD)
        X_test_no_corr = X_test_filter.drop(columns=dropped_corr_cols, errors="ignore")
        
        filtered_feature_names = np.array(X_train_no_corr.columns)
        print(f"Removed {len(dropped_corr_cols)} redundant collinear features (|r| > {CORR_THRESHOLD}).")

        # Imputation & Scaling
        imputer = SimpleImputer(strategy="median")
        X_train_imputed = imputer.fit_transform(X_train_no_corr)
        X_test_imputed = imputer.transform(X_test_no_corr)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        # Leakage-Free Recursive Feature Elimination (RFECV) for Feature Selection
        print(f"Executing Cross-Validated Feature Selection (targeting top {TOP_N_FEATURES} features)...")
        rf_selector = RandomForestRegressor(n_estimators=100, max_depth=8, min_samples_leaf=3, random_state=RANDOM_SEED, n_jobs=-1)
        
        rfecv = RFECV(
            estimator=rf_selector,
            step=5,
            cv=KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED),
            scoring="r2",
            min_features_to_select=min(TOP_N_FEATURES, X_train_scaled.shape[1]),
            n_jobs=-1
        )
        rfecv.fit(X_train_scaled, y_train)

        # Aggregate ensemble importances across top selected feature space
        rf_selector.fit(X_train_scaled[:, rfecv.support_], y_train)
        xgb_selector = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.03, min_child_weight=5, random_state=RANDOM_SEED, n_jobs=-1)
        xgb_selector.fit(X_train_scaled[:, rfecv.support_], y_train)

        avg_imp = (rf_selector.feature_importances_ + xgb_selector.feature_importances_) / 2.0
        top_idx_within_selected = np.argsort(avg_imp)[::-1][:TOP_N_FEATURES]

        selected_indices = np.where(rfecv.support_)[0][top_idx_within_selected]
        
        X_train_top = X_train_scaled[:, selected_indices]
        X_test_top = X_test_scaled[:, selected_indices]
        
        top_importances = avg_imp[top_idx_within_selected]
        top_feature_labels = filtered_feature_names[selected_indices]

        # Regularized Predictive Models Architecture
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=10.0),
            "Lasso Regression": Lasso(alpha=0.02),
            "Random Forest (Reg)": RandomForestRegressor(
                n_estimators=200, 
                max_depth=8, 
                min_samples_leaf=3, 
                max_features="sqrt", 
                random_state=RANDOM_SEED, 
                n_jobs=-1
            ),
            "XGBoost (Reg)": XGBRegressor(
                n_estimators=150, 
                max_depth=4, 
                learning_rate=0.03, 
                min_child_weight=5, 
                gamma=0.1, 
                reg_alpha=0.1, 
                reg_lambda=1.0, 
                subsample=0.8, 
                colsample_bytree=0.8, 
                random_state=RANDOM_SEED, 
                n_jobs=-1
            )
        }

        txt_report.write("--------------------------------------------------------------------------------\n")
        txt_report.write(f"TARGET ENDPOINT METRIC MODELING PHASE: {TARGET_COLUMN}\n")
        txt_report.write("--------------------------------------------------------------------------------\n")
        txt_report.write(f"Total training compounds used: {X_train_raw.shape[0]} molecules\n")
        txt_report.write(f"Features after correlation pruning: {X_train_scaled.shape[1]} features\n")
        txt_report.write(f"Final selected top feature count: {X_train_top.shape[1]} features\n\n")
        
        txt_report.write(f"MODEL PREDICTION QUALITY ({CV_FOLDS}-FOLD CV & HOLDOUT TEST):\n")
        print(f"\n--- Regularized Model Performance (5-Fold CV & Top {TOP_N_FEATURES} Features) ---")
        
        cv_splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        for name, model in models.items():
            # 5-Fold Cross Validation on Training Subset
            cv_results = cross_validate(model, X_train_top, y_train, cv=cv_splitter, scoring="r2", n_jobs=-1)
            cv_r2_mean = np.mean(cv_results['test_score'])
            cv_r2_std = np.std(cv_results['test_score'])

            # Fit and Test on Independent Holdout Split
            model.fit(X_train_top, y_train)
            train_preds = model.predict(X_train_top)
            test_preds = model.predict(X_test_top)
            
            r2_train = r2_score(y_train, train_preds)
            r2_test = r2_score(y_test, test_preds)
            mae_test = mean_absolute_error(y_test, test_preds)
            rmse_test = np.sqrt(mean_squared_error(y_test, test_preds))
            
            print(f"{name:<20} | 5-Fold CV R2: {cv_r2_mean:.3f} (+/- {cv_r2_std:.3f}) | Test R2: {r2_test:.3f} | Test MAE: {mae_test:.3f} | Test RMSE: {rmse_test:.3f}")
            
            txt_report.write(f"  > Model Framework: {name}\n")
            txt_report.write(f"    * 5-FOLD CV R2 -> Mean: {cv_r2_mean:6.4f} (Std Dev: {cv_r2_std:6.4f})\n")
            txt_report.write(f"    * HOLDOUT DATA -> Train R2: {r2_train:6.4f} | Test R2: {r2_test:6.4f} | MAE: {mae_test:6.4f} | RMSE: {rmse_test:6.4f}\n")
            txt_report.write(f"    * Generalization Gap (Train - Test R2): {(r2_train - r2_test):6.4f}\n\n")

        # Feature Importance Horizontal Bar Plot
        plt.figure(figsize=(12, 10))
        colors = plt.cm.viridis(np.linspace(0.4, 0.8, len(top_importances)))
        plt.barh(range(len(top_importances)), top_importances[::-1], color=colors, edgecolor='none')
        plt.yticks(range(len(top_importances)), top_feature_labels[::-1], fontsize=9)
        plt.xlabel("Mean Relative Importance Score", fontsize=12, fontweight='bold')
        plt.ylabel("Molecular Feature Descriptors / Fingerprints", fontsize=12, fontweight='bold')
        plt.title(f"Top {TOP_N_FEATURES} Predictive Features Matrix: {TARGET_COLUMN}\nSheet Context: {TARGET_SHEET_NAME}", fontsize=13, fontweight='bold', pad=15)
        plt.grid(axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(CHART_OUTPUT_PATH, dpi=300)
        plt.close()
        print(f"-> Generated Feature Importance chart at: '{CHART_OUTPUT_PATH}'")

        # Structural Fragment Visualizer
        is_morgan = "morgan" in TARGET_SHEET_NAME.lower() or "master" in TARGET_SHEET_NAME.lower()
        fragment_mols = []
        fragment_legends = []

        if is_morgan and "canonical_smiles" in df_clean.columns:
            try:
                raw_smiles = df_clean.dropna(subset=["canonical_smiles"]).sort_values(by=TARGET_COLUMN, ascending=False)["canonical_smiles"].unique()[:4]
                for idx, s in enumerate(raw_smiles):
                    m = Chem.MolFromSmiles(s)
                    if m:
                        fragment_mols.append(m)
                        fragment_legends.append(f"High Active Match #{idx+1}")
                
                if fragment_mols:
                    img = Draw.MolsToGridImage(fragment_mols, molsPerRow=2, subImgSize=(300, 300), legends=fragment_legends)
                    img.save(FRAGMENT_IMAGE_PATH)
                    print(f"-> Generated Structural Fragment canvas grid at: '{FRAGMENT_IMAGE_PATH}'")
                else:
                    with open(FRAGMENT_IMAGE_PATH, 'wb') as f: f.write(b"")
            except Exception as e:
                print(f"Skipping fragment generation due to environment setup: {e}")
                with open(FRAGMENT_IMAGE_PATH, 'wb') as f: f.write(b"")
        else:
            with open(FRAGMENT_IMAGE_PATH, 'wb') as f:
                f.write(b"")

        txt_report.write(f"TOP {TOP_N_FEATURES} CRITICAL PREDICTIVE MOLECULAR DESCRIPTORS RANKING MATRIX:\n")
        for rank_idx, (lbl, imp_score) in enumerate(zip(top_feature_labels, top_importances), start=1):
            txt_report.write(f"  Rank {rank_idx:02d} | Feature Name: {lbl:<35} | Mean Importance Score: {imp_score:.5f}\n")
        
        txt_report.write("\nSTRUCTURAL INTERPRETATION & CHEMICAL DECODING ANALYSIS:\n")
        
        for rank_idx, feature in enumerate(top_feature_labels[:5], start=1):
            txt_report.write(f"\n[Rank #{rank_idx}] Feature Identifier: '{feature}'\n")
            
            if "morgan_bit_" in feature:
                bit_id = int(feature.split("_")[-1])
                txt_report.write(f"  -> Type: RDKit Morgan Circular Fingerprint Bit (Radius=2)\n")
                txt_report.write(f"  -> Description: Topological atom environment centering on specific atom paths.\n")
                
                valid_hits = df_clean[df_clean[feature] == 1].sort_values(by=TARGET_COLUMN, ascending=False)
                if not valid_hits.empty:
                    txt_report.write(f"  -> Occurrences in Dataset: Found {len(valid_hits)} molecules carrying this fragment.\n")
                    txt_report.write(f"  -> Mapping Environment SMILES Traces from high-performing compounds:\n")
                    
                    found_env_count = 0
                    for _, row in valid_hits.head(3).iterrows():
                        mol = Chem.MolFromSmiles(row["canonical_smiles"])
                        if mol:
                            info_dict = {}
                            _ = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048, bitInfo=info_dict)
                            if bit_id in info_dict:
                                for match in info_dict[bit_id]:
                                    atom_idx, radius = tuple(match)[:2]
                                    if radius == 0:
                                        env_smiles = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                                    else:
                                        env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                                        amap = {}
                                        submol = Chem.PathToSubmol(mol, env, atomMap=amap)
                                        env_smiles = Chem.MolToSmiles(submol)
                                    txt_report.write(f"     * Molecule {row['molecule_chembl_id']} ({TARGET_COLUMN.split()[0]}: {row[TARGET_COLUMN]:.2f}) -> Path Environment: {env_smiles}\n")
                                    found_env_count += 1
                    if found_env_count == 0:
                        txt_report.write("     * (Environments transformed during filtering or scaling steps)\n")
                else:
                    txt_report.write("  -> Occurrences in Dataset: No active compounds found carrying this bit in the evaluated data partition.\n")
                    
            elif "maccs_bit_" in feature:
                bit_id = int(feature.split("_")[-1])
                structural_definition = MACCS_DEFINITIONS.get(bit_id, "Unknown generic MDL MACCS structural definition fragment pattern.")
                txt_report.write(f"  -> Type: MDL MACCS 166-Key Predefined Structural Bit Vector\n")
                txt_report.write(f"  -> Original Standardized Definition: \"{structural_definition}\"\n")
                
                valid_hits = df_clean[df_clean[feature] == 1].sort_values(by=TARGET_COLUMN, ascending=False)
                txt_report.write(f"  -> Occurrences in Dataset: Found {len(valid_hits)} compounds matching this pattern.\n")
                if not valid_hits.empty:
                    txt_report.write(f"  -> Highest Activity Compound IDs expressing this fragment rule:\n")
                    for _, row in valid_hits.head(3).iterrows():
                        txt_report.write(f"     * {row['molecule_chembl_id']} ({TARGET_COLUMN.split()[0]}: {row[TARGET_COLUMN]:.2f}) | Structure: {row['canonical_smiles']}\n")
            
            else:
                txt_report.write(f"  -> Type: Continuous Physicochemical/Topological 2D/3D Molecule Descriptor\n")
                txt_report.write(f"  -> Description: Global property calculation value reflecting macro molecular features.\n")
                
        txt_report.write("\n" + "="*80 + "\n\n")

print(f"\nPipeline execution complete! Written report to: '{TEXT_REPORT_PATH}'")