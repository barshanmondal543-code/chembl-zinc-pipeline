# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import io
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Draw
from rdkit.Chem import Descriptors3D
from openpyxl.drawing.image import Image as OpenpyxlImage

# ==============================================================================
# CONFIGURATION BLOCK - Customize your parameters here
# ==============================================================================
INPUT_FILENAME = "chembl_database_for_IC50.csv"
OUTPUT_EXCEL_FILE = "generated/molecular_camble_database_for_ic50.xlsx"

# OPTIONAL FLAGS
COMPUTE_3D = False  # Change to True if you want to calculate 3D descriptors
# Render + embed the 100x100 2D structure images into the workbook. Keep True
# for the Task-1 deliverable; set False for large auxiliary files where the
# images are not needed (structure drawing dominates runtime + file size).
EMBED_IMAGES = True

# Fingerprint definitions
MORGAN_RADIUS = 2 
MORGAN_BITS = 2048
# ==============================================================================

print(f"Loading database file: '{INPUT_FILENAME}'...")
df_base = pd.read_csv(INPUT_FILENAME, keep_default_na=False)

# Isolate mapping descriptors/activity data for merging later
activity_cols = [c for c in df_base.columns if c not in ["molecule_chembl_id", "canonical_smiles"]]
df_meta = df_base[["molecule_chembl_id", "canonical_smiles"] + activity_cols]

# Initialize storage lists
valid_ids = []
valid_smiles = []
morgan_list = []
maccs_list = []
desc_2d_list = []
desc_3d_list = []
image_paths = {}

# Set up descriptor catalogs
descriptor_names_2d = [desc_name for desc_name, _ in Descriptors._descList]
descriptor_names_3d = [
    "Asphericity", "Eccentricity", "InertialShapeFactor", 
    "NPR1", "NPR2", "PMI1", "PMI2", "PMI3", 
    "RadiusOfGyration", "SpherocityIndex"
]

# Ensure dynamic image temp folder exists
os.makedirs("temp_img", exist_ok=True)
os.makedirs("generated", exist_ok=True)

print(f"\nProcessing {len(df_base)} molecules and rendering visual charts...")
invalid_count = 0

for idx, row in df_base.iterrows():
    mol_id = row["molecule_chembl_id"]
    smiles = row["canonical_smiles"]
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid_count += 1
        continue
        
    valid_ids.append(mol_id)
    valid_smiles.append(smiles)
    
    # --- Step 1: Render 2D Molecular Image ---
    if EMBED_IMAGES:
        img = Draw.MolToImage(mol, size=(100, 100))
        img_path = f"temp_img/{mol_id}.png"
        img.save(img_path)
        image_paths[mol_id] = img_path
    
    # --- Step 2: Compute Morgan Fingerprints ---
    morgan_bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    morgan_arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(morgan_bv, morgan_arr)
    morgan_list.append(morgan_arr)
    
    # --- Step 3: Compute MACCS Keys ---
    maccs_bv = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(maccs_bv, maccs_arr)
    maccs_list.append(maccs_arr[1:]) # Drop 0-index offset dummy bit
    
    # --- Step 4: Compute 2D Descriptors ---
    current_mol_2d = {}
    for desc_name in descriptor_names_2d:
        try:
            current_mol_2d[desc_name] = getattr(Descriptors, desc_name)(mol)
        except Exception:
            current_mol_2d[desc_name] = np.nan
    desc_2d_list.append(current_mol_2d)
    
    # --- Step 5: Compute Optional 3D Descriptors ---
    current_mol_3d = {}
    if COMPUTE_3D:
        mol_with_h = Chem.AddHs(mol)
        embed_status = AllChem.EmbedMolecule(mol_with_h, randomSeed=42, maxAttempts=100)
        if embed_status == 0:
            AllChem.MMFFOptimizeMolecule(mol_with_h)
            for desc_name in descriptor_names_3d:
                try:
                    current_mol_3d[desc_name] = getattr(Descriptors3D, desc_name)(mol_with_h)
                except Exception:
                    current_mol_3d[desc_name] = np.nan
        else:
            for desc_name in descriptor_names_3d:
                current_mol_3d[desc_name] = np.nan
    else:
        # Populate with NaN if user config choice has the parameter disabled
        for desc_name in descriptor_names_3d:
            current_mol_3d[desc_name] = np.nan
    desc_3d_list.append(current_mol_3d)

print(f"  Successfully calculated features for {len(valid_ids)} molecules.")

# Convert base configurations to DataFrame blocks
df_core = pd.DataFrame({"molecule_chembl_id": valid_ids, "canonical_smiles": valid_smiles})
df_morgan = pd.DataFrame(morgan_list, columns=[f"morgan_bit_{i}" for i in range(MORGAN_BITS)])
df_maccs = pd.DataFrame(maccs_list, columns=[f"maccs_bit_{i}" for i in range(1, 167)])

df_2d = pd.DataFrame(desc_2d_list)
df_3d = pd.DataFrame(desc_3d_list)
df_descriptors = pd.concat([df_2d, df_3d], axis=1) if COMPUTE_3D else df_2d

# --- Step 6: Layout Architecture Organizer ---
def merge_and_arrange(feature_df):
    merged = pd.concat([df_core, feature_df], axis=1)
    final_df = pd.merge(merged, df_meta, on=["molecule_chembl_id", "canonical_smiles"], how="left")
    
    # Place visual placeholder column directly before SMILES data
    final_df.insert(1, "2D_Structure", "")
    
    # Re-order array index perfectly
    act_cols = [c for c in df_meta.columns if c not in ["molecule_chembl_id", "canonical_smiles"]]
    feat_cols = list(feature_df.columns)
    ordered_layout = ["molecule_chembl_id", "2D_Structure", "canonical_smiles"] + act_cols + feat_cols
    return final_df[ordered_layout]

# Construct independent structures for the tabs
tab1_data = merge_and_arrange(df_morgan)
tab2_data = merge_and_arrange(df_maccs)
tab3_data = merge_and_arrange(df_descriptors)
df_all_features = pd.concat([df_descriptors, df_maccs, df_morgan], axis=1)
tab4_data = merge_and_arrange(df_all_features)

# --- Step 7: Export and Embed Chemical Structure Drawings ---
print(f"\nWriting matrices out to workbook at '{OUTPUT_EXCEL_FILE}'...")
with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine="openpyxl") as writer:
    tab1_data.to_excel(writer, sheet_name="Morgan Fingerprints", index=False)
    tab2_data.to_excel(writer, sheet_name="MACCS Keys", index=False)
    tab3_data.to_excel(writer, sheet_name="Physicochemical Descriptors", index=False)
    tab4_data.to_excel(writer, sheet_name="Master Combined Matrix", index=False)
    
    # Re-open raw openpyxl worksheet engines to position graphics structures
    workbook = writer.book
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        
        # Adjust dimensions to frame 100x100 images without text clipping
        worksheet.column_dimensions['B'].width = 16

        if not EMBED_IMAGES:
            continue

        # Row 1 is text headers, insert structures sequentially starting from Row 2
        for row_idx, mol_id in enumerate(valid_ids, start=2):
            worksheet.row_dimensions[row_idx].height = 80
            img_path = image_paths.get(mol_id)
            if img_path and os.path.exists(img_path):
                # FIXED: openpyxl uses .add_image() and anchors instead of .add_drawing()
                excel_img = OpenpyxlImage(img_path)
                excel_img.anchor = f"B{row_idx}"
                worksheet.add_image(excel_img)

print("Excel Database generation completed successfully!")
