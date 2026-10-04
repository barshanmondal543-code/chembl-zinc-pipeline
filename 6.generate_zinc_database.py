# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, Draw
from rdkit.Chem import Descriptors3D
from openpyxl.drawing.image import Image as OpenpyxlImage

RDLogger.DisableLog('rdApp.*')

# ==============================================================================
# CONFIGURATION BLOCK
# ==============================================================================
INPUT_FILENAME = "ZINC20_canonical_dataset.csv"
OUTPUT_EXCEL_FILE = "generated/zinc_features_database.xlsx"

COMPUTE_3D = False

MORGAN_RADIUS = 2
MORGAN_BITS = 2048
# ==============================================================================

print(f"Loading database file: '{INPUT_FILENAME}'...")
df_base = pd.read_csv(INPUT_FILENAME)
df_base.rename(columns={"SMILES": "canonical_smiles", "ZINC_ID": "zinc_id"}, inplace=True)
df_base["zinc_id"] = df_base["zinc_id"].astype(str)
print(f"Loaded {len(df_base)} molecules.")

# Initialize storage lists
valid_ids = []
valid_smiles = []
morgan_list = []
maccs_list = []
desc_2d_list = []
desc_3d_list = []
image_paths = {}

descriptor_names_2d = [desc_name for desc_name, _ in Descriptors._descList]
descriptor_names_3d = [
    "Asphericity", "Eccentricity", "InertialShapeFactor",
    "NPR1", "NPR2", "PMI1", "PMI2", "PMI3",
    "RadiusOfGyration", "SpherocityIndex"
]

os.makedirs("temp_img", exist_ok=True)
os.makedirs("generated", exist_ok=True)

print(f"\nProcessing {len(df_base)} molecules...")
invalid_count = 0

for idx, row in df_base.iterrows():
    if idx % 500 == 0:
        print(f"  Progress: {idx}/{len(df_base)}")
    mol_id = row["zinc_id"]
    smiles = row["canonical_smiles"]

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        invalid_count += 1
        continue

    valid_ids.append(mol_id)
    valid_smiles.append(smiles)

    img = Draw.MolToImage(mol, size=(100, 100))
    img_path = f"temp_img/{mol_id}.png"
    img.save(img_path)
    image_paths[mol_id] = img_path

    morgan_bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=MORGAN_RADIUS, nBits=MORGAN_BITS)
    morgan_arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(morgan_bv, morgan_arr)
    morgan_list.append(morgan_arr)

    maccs_bv = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros((1,), dtype=int)
    Chem.DataStructs.ConvertToNumpyArray(maccs_bv, maccs_arr)
    maccs_list.append(maccs_arr[1:])

    current_mol_2d = {}
    for desc_name in descriptor_names_2d:
        try:
            current_mol_2d[desc_name] = getattr(Descriptors, desc_name)(mol)
        except Exception:
            current_mol_2d[desc_name] = np.nan
    desc_2d_list.append(current_mol_2d)

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
        for desc_name in descriptor_names_3d:
            current_mol_3d[desc_name] = np.nan
    desc_3d_list.append(current_mol_3d)

print(f"  Successfully calculated features for {len(valid_ids)} molecules. Skipped {invalid_count} invalid.")

df_core = pd.DataFrame({"zinc_id": valid_ids, "canonical_smiles": valid_smiles})
df_morgan = pd.DataFrame(morgan_list, columns=[f"morgan_bit_{i}" for i in range(MORGAN_BITS)])
df_maccs = pd.DataFrame(maccs_list, columns=[f"maccs_bit_{i}" for i in range(1, 167)])

df_2d = pd.DataFrame(desc_2d_list)
df_3d = pd.DataFrame(desc_3d_list)
df_descriptors = pd.concat([df_2d, df_3d], axis=1) if COMPUTE_3D else df_2d

def merge_and_arrange(feature_df):
    merged = pd.concat([df_core, feature_df], axis=1)
    merged.insert(1, "2D_Structure", "")
    return merged

tab1_data = merge_and_arrange(df_morgan)
tab2_data = merge_and_arrange(df_maccs)
tab3_data = merge_and_arrange(df_descriptors)
df_all_features = pd.concat([df_descriptors, df_maccs, df_morgan], axis=1)
tab4_data = merge_and_arrange(df_all_features)

print(f"\nWriting to '{OUTPUT_EXCEL_FILE}'...")
with pd.ExcelWriter(OUTPUT_EXCEL_FILE, engine="openpyxl") as writer:
    tab1_data.to_excel(writer, sheet_name="Morgan Fingerprints", index=False)
    tab2_data.to_excel(writer, sheet_name="MACCS Keys", index=False)
    tab3_data.to_excel(writer, sheet_name="Physicochemical Descriptors", index=False)
    tab4_data.to_excel(writer, sheet_name="Master Combined Matrix", index=False)

    workbook = writer.book
    for sheet_name in workbook.sheetnames:
        worksheet = workbook[sheet_name]
        worksheet.column_dimensions['B'].width = 16
        for row_idx, mol_id in enumerate(valid_ids, start=2):
            worksheet.row_dimensions[row_idx].height = 80
            img_path = image_paths.get(mol_id)
            if img_path and os.path.exists(img_path):
                excel_img = OpenpyxlImage(img_path)
                excel_img.anchor = f"B{row_idx}"
                worksheet.add_image(excel_img)

print("Done!")
