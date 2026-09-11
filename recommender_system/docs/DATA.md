<!-- Codex-added 2026-08-14: document the external assets and the reproducibility boundary. -->
# EasyRL4Rec data and pretrained user-model assets

The large assets required by the EasyRL4Rec experiments are intentionally not included in this code release. The minimum checked set is approximately **4.1 GB** and consists of experiment-specific MovieLens-1M splits, processed evaluation matrices, and six pretrained DeepFM user-model ensembles.

## Important reproducibility boundary

MovieLens-1M is distributed by GroupLens under its own terms. It must not be redistributed without permission. Obtain the dataset from the [official MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/) and review the terms included with the download before using it.

This release does **not** currently contain a complete script that starts from the official MovieLens-1M download and reconstructs the exact `hetero5` and `homo3` splits and all derived files below. Consequently, this package supports rerunning the reported policy experiments when the prepared assets are supplied, but it does not claim end-to-end reproduction from the original download alone. The prepared assets should be deposited separately only where their redistribution is permitted.

## Required directory schema

Pass a staging directory with this layout to `scripts/validate_assets.py`. For the bundled training scripts, place the same directories at the repository root.

```text
<asset-root>/
├── data/MovieLens/
│   ├── hetero5_data_raw_sub1/
│   ├── hetero5_data_raw_sub2/
│   ├── hetero5_data_raw_sub3/
│   ├── hetero5_data_processed_sub1/
│   ├── hetero5_data_processed_sub2/
│   ├── hetero5_data_processed_sub3/
│   ├── homo3_data_raw_sub1/
│   ├── homo3_data_raw_sub2/
│   ├── homo3_data_raw_sub3/
│   ├── homo3_data_processed_sub1/
│   ├── homo3_data_processed_sub2/
│   └── homo3_data_processed_sub3/
└── saved_models/MovieLensEnv-v0/DeepFM/
    ├── params/
    ├── matsPre/
    ├── models/
    └── embeddings/
```

Each `*_data_raw_sub{1,2,3}` directory must contain:

```text
movielens-1m-train.csv
movielens-1m-test.csv
movies.dat
users.dat
rating_matrix.csv
```

Each corresponding `*_data_processed_sub{1,2,3}` directory must contain:

```text
distance_mat.pickle
feature_domination.pickle
item_popularity_add1.pickle
item_similarity_add1.pickle
```

The six user-model identifiers are:

```text
hetero5_pointnegA  hetero5_pointnegB  hetero5_pointnegC
homo3_pointnegA    homo3_pointnegB    homo3_pointnegC
```

For every identifier `<message>`, the minimum DeepFM files are:

```text
params/[<message>]_params.pickle
matsPre/[<message>]_matPre.pickle
models/[<message>]_model_M0.pt
models/[<message>]_model_M1.pt
models/[<message>]_model_M2.pt
models/[<message>]_model_M3.pt
models/[<message>]_model_M4.pt
embeddings/[<message>]_emb_user_val_M0.pt
embeddings/[<message>]_emb_item_val_M0.pt
```

The code creates a `matsVar` file while fitting a user-model ensemble, but the released policy-training path does not read `matsVar`; it is therefore not part of the minimum runtime asset set.

## Validate a staged asset set

The default check only verifies that every required path is a non-empty regular file, so it is fast:

```bash
python scripts/validate_assets.py --asset-root .
```

For an archival inventory, request a CSV containing byte sizes and SHA-256 digests. Hashing approximately 4.1 GB takes longer:

```bash
python scripts/validate_assets.py \
  --asset-root . \
  --write-manifest asset_manifest.csv
```

All paths stored in the manifest are relative to `<asset-root>`; the validator does not print or record the asset root itself.

## Dataset citation

Please cite the official dataset paper when using MovieLens:

> F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* **5**, Article 19 (2015). https://doi.org/10.1145/2827872

Dataset page: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
