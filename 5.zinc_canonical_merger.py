# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import glob
import csv
from rdkit import Chem
from rdkit import RDLogger

# 1. Silence RDKit parse error logs in the console
RDLogger.DisableLog('rdApp.*')

# 2. Find all downloaded .smi files
smi_files = glob.glob("zinc_smi_files/*.smi")

master_csv_filename = "ZINC20_canonical_dataset.csv"
unique_canonical_smiles = set()
total_lines_processed = 0
invalid_count = 0
duplicate_count = 0  # Counter to track redundancy

print(f"Found {len(smi_files)} files to process.")

# 3. Open the CSV file and write headers
with open(master_csv_filename, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["SMILES", "ZINC_ID"])  # Column headers
    
    for file_path in smi_files:
        if "canonical_dataset" in file_path:
            continue
            
        with open(file_path, "r") as infile:
            for line in infile:
                total_lines_processed += 1
                parts = line.strip().split()
                
                if len(parts) < 2:
                    continue
                    
                raw_smiles = parts[0]
                zinc_id = parts[1]
                
                # Skip header rows if they exist in the files
                if raw_smiles.lower() == "smiles":
                    continue
                
                # 4. Convert to molecule object to generate Canonical SMILES
                mol = Chem.MolFromSmiles(raw_smiles)
                
                if mol is not None:
                    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
                    
                    # 5. Check for duplicacy
                    if canonical_smiles not in unique_canonical_smiles:
                        unique_canonical_smiles.add(canonical_smiles)
                        writer.writerow([canonical_smiles, zinc_id])
                    else:
                        duplicate_count += 1  # Found a redundant molecule
                else:
                    invalid_count += 1

# 6. Final report showing exact duplicacy metrics
print("\n✨ Processing Complete!")
print(f"Total entries scanned: {total_lines_processed}")
print(f"Redundant/Duplicate molecules removed: {duplicate_count} 🛑")
print(f"Skipped/Invalid SMILES strings: {invalid_count}")
print("-" * 40)
print(f"Final unique canonical molecules saved: {len(unique_canonical_smiles)} ✅")
print(f"Dataset successfully saved as: {master_csv_filename}")
