# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import numpy as np
import pandas as pd
import time
from chembl_webresource_client.new_client import new_client

# ==============================================================================
# CONFIGURATION BLOCK - Adjust these variables to customize your search
# ==============================================================================
SELECTED_TARGET_ID = "CHEMBL3024"  # Target ChEMBL ID: Serine/threonine-protein kinase PLK1

# Properties to extract
#DESIRED_ACTIVITIES = [
#    "EC50", "IC50", "Ki", "Kd", "Kon", "Koff",
#    "MIC50", "MIC90", "MIC"
#]

DESIRED_ACTIVITIES = [
    "IC50"
]

# Output file name
OUTPUT_FILENAME = "chembl_database_for_IC50.csv"
# ==============================================================================

activity_api = new_client.activity
data_list = []

def normalize_activity_units(value, unit, activity_type):
    """
    Normalizes property values into consistent structural baselines.
    - Affinity/Potency/MIC -> Standardized to 'nM'
    - Forward rates (Kon)  -> Standardized to 'M-1.s-1'
    - Reverse rates (Koff)  -> Standardized to 's-1'
    """
    if value is None:
        return None, None

    try:
        val = float(value)
    except (ValueError, TypeError):
        return None, None

    u_lower = str(unit).lower().strip() if unit else ""

    if activity_type.lower() == "kon":
        return val, "M-1.s-1"
    elif activity_type.lower() == "koff":
        return val, "s-1"
    else:
        # Concentration metrics (IC50, EC50, Ki, Kd, MIC) scaled directly to nM
        if u_lower in ["nm", "nmol/l", "nmol.l-1"]:
            return val, "nM"
        elif u_lower in ["um", "µm", "umol/l", "µmol/l", "umol.l-1"]:
            return val * 1000.0, "nM"
        elif u_lower in ["mm", "mmol/l", "mmol.l-1"]:
            return val * 1000000.0, "nM"
        elif u_lower in ["m", "mol/l", "mol.l-1"]:
            return val * 1000000000.0, "nM"
        elif u_lower in ["pm", "pmol/l", "pmol.l-1"]:
            return val / 1000.0, "nM"
        elif u_lower in ["ug/ml", "µg/ml"]:
            return val * 1000.0, "nM"
        else:
            return val, "nM"

print(f"Starting database generation pipeline for target: {SELECTED_TARGET_ID}...")

# Step 1: Extract and align raw values to common baseline units
for activity_type in DESIRED_ACTIVITIES:
    print(f"Fetching '{activity_type}' values...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            activities = activity_api.filter(
                target_chembl_id=SELECTED_TARGET_ID,
                standard_type=activity_type
            )

            count = 0
            for act in activities:
                time.sleep(0.1)
                raw_val = act.get("standard_value")
                raw_unit = act.get("standard_units")
                mol_id = act.get("molecule_chembl_id")

                if act.get("data_validity_comment") in ["Outside typical range", "Non-standard unit condition"]:
                    continue

                if mol_id and raw_val is not None:
                    norm_val, target_unit = normalize_activity_units(raw_val, raw_unit, activity_type)
                    if norm_val is not None:
                        data_list.append(
                            {
                                "molecule_chembl_id": mol_id,
                                "activity_type": activity_type,
                                "standard_value": norm_val
                            }
                        )
                        count += 1
            print(f"  Retrieved and normalized {count} rows for '{activity_type}'")
            break
        except Exception as e:
            print(f"  Attempt {attempt+1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                wait = 5 * (attempt + 1)
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  Skipping '{activity_type}' after {max_retries} attempts.")

df_raw = pd.DataFrame(data_list)

print(f"\nDebug: Total rows collected: {len(data_list)}")
if len(data_list) > 0:
    print(f"Debug: Sample row: {data_list[0]}")

if df_raw.empty:
    print("\nError: No valid metrics collected under configuration filters. Exiting.")
    exit()

# --- STEP 2: Pivot & Deduplicate using Median Aggregation ---
print("\nPivoting data structure to cross-compare identical molecules...")
df_pivoted = df_raw.pivot_table(
    index="molecule_chembl_id",
    columns="activity_type",
    values="standard_value",
    aggfunc="median"
).reset_index()

# --- STEP 3: Apply Logarithmic Scaling Target Column Pipeline ---
print("Calculating logarithmic scales for affinity features...")

log_targets = ["IC50", "EC50", "Ki", "Kd"]

for target in log_targets:
    if target in df_pivoted.columns:
        log_header = f"p{target} (-log10(M))"
        df_pivoted[log_header] = df_pivoted[target].apply(
            lambda x: 9.0 - np.log10(x) if (pd.notna(x) and x > 0) else np.nan
        )

# Set clean uniform headers for raw baseline value columns
header_units = {
    "EC50": "nM", "IC50": "nM", "Ki": "nM", "Kd": "nM",
    "MIC50": "nM", "MIC90": "nM", "MIC": "nM",
    "Kon": "M-1.s-1", "Koff": "s-1"
}

# Rename only the raw columns that do not already have the p-prefix formula label
rename_dict = {}
for col in df_pivoted.columns:
    if col != "molecule_chembl_id" and not col.startswith("p"):
        rename_dict[col] = f"{col} ({header_units.get(col, 'unknown')})"

df_pivoted = df_pivoted.rename(columns=rename_dict)

# --- STEP 4: Fetch Structural SMILES Keys in Batches ---
unique_molecule_ids = list(df_pivoted["molecule_chembl_id"].unique())
print(f"Fetching chemical structures (SMILES) for {len(unique_molecule_ids)} unique compounds...")

molecule_api = new_client.molecule
smiles_map = {}
chunk_size = 500

for i in range(0, len(unique_molecule_ids), chunk_size):
    chunk = unique_molecule_ids[i : i + chunk_size]
    try:
        molecule_structures = molecule_api.filter(
            molecule_chembl_id__in=chunk
        ).only("molecule_chembl_id", "molecule_structures")

        for mol in molecule_structures:
            m_id = mol.get("molecule_chembl_id")
            structs = mol.get("molecule_structures")
            if structs and structs.get("canonical_smiles"):
                smiles_map[m_id] = structs.get("canonical_smiles")
    except Exception as e:
        print(f"  Warning: Error on structural collection batch: {e}")

# --- STEP 5: Merge Structures and Finalize Clean Database Order ---
df_pivoted["canonical_smiles"] = df_pivoted["molecule_chembl_id"].map(smiles_map)
df_pivoted = df_pivoted.dropna(subset=["canonical_smiles"])

# Enforce clean sorted order: Identifier, Structure, Raw values, then Log Scaling Values
front_cols = ["molecule_chembl_id", "canonical_smiles"]
raw_cols = [c for c in df_pivoted.columns if "(" in c and not c.startswith("p")]
p_cols = [c for c in df_pivoted.columns if c.startswith("p")]
other_cols = [c for c in df_pivoted.columns if c not in front_cols + raw_cols + p_cols]

final_column_order = front_cols + sorted(raw_cols) + sorted(p_cols) + other_cols
df_pivoted = df_pivoted[final_column_order]

# --- STEP 6: Enforce Explicit "NaN" String Filling ---
print("Filling all empty cells with explicit 'NaN' strings for strict ML parsing consistency...")
df_pivoted = df_pivoted.fillna("NaN")

print(f"\nFinal Generated Database Summary:")
print(f"  - Total Matrix Shape: {df_pivoted.shape}")
print(f"  - Generated Columns: {list(df_pivoted.columns)}")

# --- STEP 7: Export clean database ---
df_pivoted.to_csv(OUTPUT_FILENAME, index=False)
print(f"\nYour clean database .csv file was successfully built: '{OUTPUT_FILENAME}'")