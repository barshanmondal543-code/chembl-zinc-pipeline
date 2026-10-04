# Blind IC50 Prediction for PLK1 (CHEMBL3024)

**ML model training and blind prediction of molecular features** — CAD course
project (BB611T Assignment 2): ChEMBL → RDKit features → model comparison →
blind ZINC20 prediction.

## What it does

Script 1 pulls every IC50 measurement for the target from the ChEMBL web API and
writes a clean CSV. Script 2 turns those SMILES into a feature workbook with RDKit
(Morgan fingerprints, MACCS keys and physicochemical descriptors, plus embedded 2D
structure images). Script 3 trains and compares five regularized regressors
(Linear, Ridge, Lasso, Random Forest, XGBoost) with a leakage-free feature
selection chain and writes the evaluation report and plots. Scripts 4 and 5
download a ZINC20 tranche and merge it into one canonical SMILES dataset. Script 7
removes every ZINC molecule that already appears in the training set, and Script 8
re-runs the exact v2 feature selection, trains the chosen model on all labelled
compounds, and writes a ranked blind prediction CSV with an applicability-domain
flag for each molecule.

## Target

| Field       | Value                                                     |
| ----------- | --------------------------------------------------------- |
| Target name | Serine/threonine-protein kinase PLK1 (Polo-like kinase 1) |
| ChEMBL ID   | CHEMBL3024                                                |
| Organism    | *Homo sapiens*                                          |
| Target type | Single protein                                            |
| Endpoint    | IC50 (nM), modelled as pIC50 = 9 − log10(IC50 in nM)     |

Looked up on the ChEMBL target page: [https://www.ebi.ac.uk/chembl/explore/target/CHEMBL3024](https://www.ebi.ac.uk/chembl/explore/target/CHEMBL3024)

## Run order

Run the scripts from the repository root (all paths are relative):

1. `1.fetch_camble_data.py` — ChEMBL IC50 fetch → `chembl_database_for_IC50.csv`
2. `2.generate_camble_database.py` — RDKit features → `generated/molecular_camble_database_for_ic50.xlsx`
3. `3.molecular_pipeline_code-v2.py` — model comparison → `top_feature_of_ic50/`
4. `4.download_zinc.py` — download ZINC20 `.smi` files → `zinc_smi_files/` (needs `ZINC-downloader-2D-smi.uri`, see below)
5. `5.zinc_canonical_merger.py` — merge/deduplicate → `ZINC20_canonical_dataset.csv`
6. `6.generate_zinc_database.py` — **optional.** Featurizes the full ZINC set into `generated/zinc_features_database.xlsx`. Nothing downstream reads it (Script 8 featurizes ZINC itself), so you can skip it.
7. `7.screen_zinc_vs_training.py` — drop training molecules from ZINC → `generated/ZINC20_screened_for_ic50.csv`
8. `8.predict_zinc_blind.py` — blind prediction → `generated/blind_prediction_final.csv`

So the chain is **1 → 2 → 3 → 4 → 5 → 7 → 8**; script 6 is optional.

### Setup

Requires **Python 3.12 or newer** — xgboost 3.4.1 is the strictest of the pins
(`Requires-Python >=3.12`), pandas / numpy / scikit-learn / matplotlib all want
3.11+. Everything here was run on **Python 3.14.6** (Homebrew).

```bash
pip install -r requirements.txt
```

Pillow is pulled in for the 2D structure images that Scripts 2 and 6 embed into
their Excel workbooks.

`4.download_zinc.py` reads a URI list called `ZINC-downloader-2D-smi.uri`. That
file is **not** committed here; get it from the ZINC20 download page
([https://zinc20.docking.org/](https://zinc20.docking.org/)) by selecting a tranche subset and downloading the
selection as a URI list. The selection used here is the 16 `CA**.smi` files under
`http://files.docking.org/2D/CA/` (`CAAA`–`CAAD`, `CABA`–`CABD`, `CACA`–`CACD`,
`CAEA`–`CAED`). In the ZINC property grid, `CA` is the cell for molecular weight
250–300 Da and XlogP in the lowest column (A, i.e. very polar compounds); the
third and fourth letters of a tranche name are ZINC's reactivity and
purchasability axes. The downloaded data confirms the selection — every sampled
molecule sits between 250 and 300 Da, with XlogP from -4.2 to -0.5. Save the URI
list next to the scripts as `ZINC-downloader-2D-smi.uri`.

## Results

Configured run: sheet **"Physicochemical Descriptors"** of
`generated/molecular_camble_database_for_ic50.xlsx`, target `pIC50 (-log10(M))`,
1672 labelled compounds (75/25 split, `RANDOM_SEED = 42` → 1254 train / 418
hold-out), 148 features after correlation pruning, top 30 features selected,
5-fold CV on the training split.

| Model               | 5-fold CV R² (mean ± sd) | Hold-out test R² | MAE   | RMSE  |
| ------------------- | -------------------------- | ----------------- | ----- | ----- |
| Linear Regression   | 0.488 ± 0.059             | 0.542             | 0.873 | 1.100 |
| Ridge Regression    | 0.489 ± 0.056             | 0.539             | 0.876 | 1.103 |
| Lasso Regression    | 0.485 ± 0.049             | 0.519             | 0.904 | 1.126 |
| Random Forest (Reg) | 0.731 ± 0.036             | 0.763             | 0.612 | 0.791 |
| XGBoost (Reg)       | 0.719 ± 0.035             | 0.768             | 0.602 | 0.783 |

Metrics are in pIC50 units (log molar): the two tree ensembles clearly outperform
the three linear models — XGBoost has the best hold-out R², MAE and RMSE, while
Random Forest has the higher cross-validation score.

This table and every file in `top_feature_of_ic50/` were produced by
`3.molecular_pipeline_code-v2.py` on 3 Oct 2026 from the Script 1 fetch of the
same day (1672 molecules with an IC50 for CHEMBL3024). The full report — feature
ranking and the structural interpretation of the top features — is in
`top_feature_of_ic50/top_feature_explanations_Physicochemical Descriptors.txt`,
with the matching `top_feature_importances_Physicochemical Descriptors.png`
alongside it. The Morgan-fingerprint and MACCS-key reports are in the same
folder — the Morgan sheet also writes
`top_fingerprint_fragments_Morgan Fingerprints.png`, a grid of the four most
active molecules in that sheet — so the numbers above can be reproduced for
other feature representations by changing `TARGET_SHEET_NAME` in Script 3.

## Limitations

- **Random split, not scaffold-based.** Compounds are split at random, so close
  analogues can land on both sides of the train/test boundary; with this scaffold
  redundancy the hold-out scores are optimistic compared with a
  (scaffold/group) split.
- **No external validation.** Everything is scored on an internal hold-out of the
  same ChEMBL dump. There is no test set from another assay, lab or publication.
- **The ZINC predictions are computational only.** The blind IC50 values in
  `generated/blind_prediction_final.csv` have never been tested experimentally;
  they are ranking hypotheses for follow-up, not measurements. The
  applicability-domain column says how far each molecule sits from the training
  set, but out-of-domain predictions should be treated with particular caution.
- **Single target, single endpoint.** Only IC50 values for CHEMBL3024 are used,
  and duplicate measurements are collapsed by median.

## Data and licences

The data is **not included** in this repository (see `.gitignore`); the scripts
re-download it:

- **ChEMBL** — activity data for CHEMBL3024, fetched live through the
  `chembl_webresource_client` API. ChEMBL data are released under the CC BY-SA 3.0
  licence; see [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/) for terms and the preferred
  citation.
- **ZINC20** — the 2D SMILES library, downloaded from the ZINC20 tranche browser.
  ZINC is made available freely by the Shoichet and Irwin laboratories at UCSF.
  [https://zinc20.docking.org/](https://zinc20.docking.org/)

Third-party code and model libraries keep their own licences (RDKit BSD,
scikit-learn BSD, XGBoost Apache-2.0, pandas BSD, matplotlib PSF-based — see
`requirements.txt`).

## Acknowledgements

The base code in this repository was provided by Prof. Suvamay Jana as part of
the Computer-Aided Drug Design (CAD) course at the Indian Institute of
Technology Dharwad, 2026. It is shared here with his permission. Modifications,
analysis, and results are my own.
