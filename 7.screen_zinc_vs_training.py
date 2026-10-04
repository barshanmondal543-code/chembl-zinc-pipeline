# Original code provided by Prof. Suvamay Jana, CAD course, IIT Dharwad.
# Shared with permission. Modified by Barshan.

import os
import pandas as pd

# RDKit for canonical SMILES comparison
from rdkit import Chem
from rdkit import RDLogger

# ==============================================================================
# SCRIPT 7 — REMOVE TRAINING MOLECULES FROM THE ZINC LIBRARY
#
# Screens ZINC20_canonical_dataset.csv against the labelled training database
# (generated/molecular_camble_database_for_ic50.xlsx) and removes every
# ZINC molecule that is already present in the training set, so Script 8's
# "blind" screen can no longer rediscover its own training actives.
#
# SMILES are compared in RDKit canonical form (NOT as raw strings), so the same
# molecule written in a different order still matches.
#
# SCREEN_MODE:
#   "exact"     remove rows whose canonical SMILES equals a training molecule.
#   "no_stereo" same comparison done WITHOUT stereochemistry, so a training
#               compound written with different / missing stereo in ZINC is
#               caught too. (this mode also catches every exact match)
#
# Outputs (written to generated/):
#   ZINC20_screened_for_ic50.csv  -> rows to KEEP (original rows + columns)
#   ZINC20_removed_vs_training.csv -> rows removed, with the reason (audit trail)
#   zinc_screening_report.txt      -> counts + examples
#
# Usage: point Script 8 at the cleaned file:
#   ZINC_INPUT = "generated/ZINC20_screened_for_ic50.csv"
# ==============================================================================

# --- Inputs ---
ZINC_INPUT = "ZINC20_canonical_dataset.csv"
ZINC_SMILES_COL = "SMILES"
ZINC_ID_COL = "ZINC_ID"

TRAINING_EXCEL = "generated/molecular_camble_database_for_ic50.xlsx"

# --- Screening settings ---
SCREEN_MODE = "no_stereo"   # "exact" | "no_stereo"
CHUNKSIZE = 20000           # rows per streamed chunk (memory stays flat)

# --- Outputs ---
OUTPUT_DIR = "generated"
SCREENED_CSV = f"{OUTPUT_DIR}/ZINC20_screened_for_ic50.csv"
REMOVED_CSV = f"{OUTPUT_DIR}/ZINC20_removed_vs_training.csv"
REPORT_TXT = f"{OUTPUT_DIR}/zinc_screening_report.txt"
REPORT_EXAMPLES = 25
# ==============================================================================

RDLogger.DisableLog("rdApp.*")


def canonical_form(smiles, include_stereo=True):
    """RDKit canonical SMILES for one molecule; None if it cannot be parsed."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=include_stereo)


def load_training_smiles(mode):
    """Collect every training SMILES (all sheets) as a set of comparable forms."""
    if not os.path.exists(TRAINING_EXCEL):
        raise FileNotFoundError(f"Training Excel not found: {TRAINING_EXCEL}")

    xl = pd.ExcelFile(TRAINING_EXCEL)
    raw_strings, sheets_used = set(), []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if "canonical_smiles" not in df.columns:
            continue
        sheets_used.append(sheet)
        vals = df["canonical_smiles"].dropna().astype(str).str.strip()
        raw_strings.update(v for v in vals if v and v.lower() != "nan")

    if not raw_strings:
        raise ValueError(
            f"No 'canonical_smiles' column found in any sheet of {TRAINING_EXCEL}."
        )

    include_stereo = mode == "exact"
    forms, n_invalid = set(), 0
    for s in raw_strings:
        forms.add(s)  # raw string is always a valid (exact) match
        f = canonical_form(s, include_stereo=include_stereo)
        if f is None:
            n_invalid += 1
        else:
            forms.add(f)
    return forms, sheets_used, len(raw_strings), n_invalid


def main():
    if SCREEN_MODE not in ("exact", "no_stereo"):
        raise ValueError(f'SCREEN_MODE must be "exact" or "no_stereo"; got {SCREEN_MODE!r}')
    if not os.path.exists(ZINC_INPUT):
        raise FileNotFoundError(f"ZINC file not found: {ZINC_INPUT}")

    # ---- 1. Training (ChEMBL) SMILES -----------------------------------------
    print(f"Loading training SMILES from '{TRAINING_EXCEL}'...")
    train_forms, sheets_used, n_raw, n_train_invalid = load_training_smiles(SCREEN_MODE)
    print(f"  Sheets scanned      : {', '.join(sheets_used)}")
    print(f"  Training molecules  : {n_raw} rows ({n_train_invalid} unparseable)")
    print(f"  Screen mode         : {SCREEN_MODE}")

    # ---- 2. Stream the ZINC file, split keep / remove ------------------------
    print(f"\nScreening '{ZINC_INPUT}'...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    include_stereo = SCREEN_MODE == "exact"

    total = kept = removed = invalid = 0
    removed_rows = []
    first_chunk = True

    for chunk in pd.read_csv(ZINC_INPUT, chunksize=CHUNKSIZE):
        if ZINC_SMILES_COL not in chunk.columns:
            raise ValueError(
                f"Column '{ZINC_SMILES_COL}' not in {ZINC_INPUT}. "
                f"Found: {list(chunk.columns)}"
            )
        keep_idx = []
        for i, smi in enumerate(chunk[ZINC_SMILES_COL].values):
            total += 1
            s = str(smi)
            form = s if s in train_forms else canonical_form(s, include_stereo=include_stereo)
            if form is None:
                invalid += 1          # unparseable -> cannot be a match, keep it
                keep_idx.append(i)
                continue
            if form in train_forms:
                reason = "exact SMILES match" if form == s or include_stereo \
                    else "same molecule, stereo-insensitive match"
                removed += 1
                removed_rows.append({
                    ZINC_ID_COL: chunk[ZINC_ID_COL].values[i] if ZINC_ID_COL in chunk.columns else i,
                    ZINC_SMILES_COL: s,
                    "screen_reason": reason,
                })
            else:
                keep_idx.append(i)

        chunk.iloc[keep_idx].to_csv(
            SCREENED_CSV, mode="w" if first_chunk else "a", header=first_chunk, index=False
        )
        kept += len(keep_idx)
        first_chunk = False
        print(f"  ...scanned {total} | kept {kept} | removed {removed}", end="\r")
    print()

    if total == 0:
        print("No data rows found in the ZINC file.")
        return

    # ---- 3. Audit file of everything that was screened out -------------------
    pd.DataFrame(
        removed_rows, columns=[ZINC_ID_COL, ZINC_SMILES_COL, "screen_reason"]
    ).to_csv(REMOVED_CSV, index=False)

    # ---- 4. Report -----------------------------------------------------------
    lines = [
        "ZINC vs TRAINING SCREENING REPORT",
        "=" * 78,
        f"ZINC input          : {ZINC_INPUT}",
        f"Training Excel      : {TRAINING_EXCEL}",
        f"  sheets used       : {', '.join(sheets_used)}",
        f"  training molecules: {n_raw} rows ({n_train_invalid} unparseable)",
        f"Screen mode         : {SCREEN_MODE}"
        + (" (canonical SMILES, stereo kept)" if SCREEN_MODE == "exact"
           else " (canonical SMILES, stereochemistry ignored)"),
        "-" * 78,
        f"ZINC rows scanned   : {total}",
        f"Removed (training)  : {removed}",
        f"Invalid SMILES kept : {invalid} (cannot be compared; Script 8 skips them)",
        f"Rows kept           : {kept}",
        "-" * 78,
        f"Screened file       : {SCREENED_CSV}",
        f"Removed file        : {REMOVED_CSV}",
    ]
    if removed_rows:
        lines.append(f"\nFirst {min(REPORT_EXAMPLES, len(removed_rows))} "
                     f"of {removed_rows.__len__()} removed molecules:")
        for r in removed_rows[:REPORT_EXAMPLES]:
            lines.append(f"  {r[ZINC_ID_COL]} | {r['screen_reason']} | {r[ZINC_SMILES_COL]}")
    else:
        lines.append("\nNo training molecules were found in the ZINC file.")

    report = "\n".join(lines) + "\n"
    with open(REPORT_TXT, "w", encoding="utf-8") as fh:
        fh.write(report)

    print("=" * 78)
    print("SCREENING COMPLETE")
    print("=" * 78)
    print(f"  ZINC rows scanned : {total}")
    print(f"  Removed (training): {removed}")
    print(f"  Invalid SMILES    : {invalid} (kept)")
    print(f"  Rows kept         : {kept}")
    print(f"  Screened CSV      : {SCREENED_CSV}")
    print(f"  Removed CSV       : {REMOVED_CSV}")
    print(f"  Report            : {REPORT_TXT}")


if __name__ == "__main__":
    main()
