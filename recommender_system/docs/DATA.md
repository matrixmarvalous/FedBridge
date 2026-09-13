<!-- Codex-added 2026-08-14: document the external assets and the reproducibility boundary. -->
# EasyRL4Rec data and pretrained user-model assets

The experiment-specific MovieLens splits, processed data and pretrained DeepFM user models for one representative **hetero5 FedBridge experiment** are provided as attachments to the [hetero5 asset release](https://github.com/matrixmarvalous/FedBridge/releases/tag/recommender-hetero5-v1). Large binary assets are not stored in Git history. The repository and its release are currently private.

The bundle contains **54 runtime files, 1,455,231,928 bytes (1.46 GB) uncompressed**, plus an inventory CSV. It includes 15 raw/split files, 12 processed files and 27 model-related files. All three A/B/C user groups and their five-model ensembles are retained for the original eight-client configuration. The exact byte sizes and SHA-256 digests are recorded in [`hetero5_asset_manifest.csv`](hetero5_asset_manifest.csv).

The `homo3` assets are not included. The complete two-scene inventory would contain 108 runtime files (approximately 4.1 GB). See the [component README](../README.md) for download, training, evaluation and plotting commands for the provided experiment.

The download `fedbridge-hetero5-assets-v1.tar.gz` is 974,526,696 bytes (approximately 975 MB). Its SHA-256 digest is:

```text
4b4e5cf960bd5a3eb3a357ccf3a47e57cc352ab3e57f73fc5fd1ce936caa7972
```

## Important reproducibility boundary

MovieLens-1M is distributed by GroupLens under its own terms. It must not be redistributed without permission. Obtain the dataset from the [official MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/) and review the terms included with the download before using it.

This release does **not** contain a complete script that starts from the official MovieLens-1M download and reconstructs the exact client splits and pretrained user models. The provided experiment starts from the packaged splits, processed files and pretrained models. It does not cover user-model retraining, every experimental setting or the paper's multi-seed comparisons. Before making the repository or its asset release public, confirm permission to redistribute the derived MovieLens data and model assets. Private staging does not grant redistribution rights.

## Required directory schema

Pass the extracted `fedbridge-hetero5-assets` directory as `--asset-root` to the launcher and validator. The bundle preserves the following layout. If supplying `homo3` separately, use corresponding directories with the `homo3` prefix under the same asset root.

```text
<asset-root>/
├── data/MovieLens/
│   ├── hetero5_data_raw_sub1/
│   ├── hetero5_data_raw_sub2/
│   ├── hetero5_data_raw_sub3/
│   ├── hetero5_data_processed_sub1/
│   ├── hetero5_data_processed_sub2/
│   └── hetero5_data_processed_sub3/
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

The three provided user-model identifiers are:

```text
hetero5_pointnegA  hetero5_pointnegB  hetero5_pointnegC
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
python scripts/validate_assets.py --asset-root assets/fedbridge-hetero5-assets --scene hetero5
```

The launcher checks only the selected scene. Running the validator without `--scene` retains the full 108-file check for both scenes. To check every supplied asset against the release inventory:

```bash
python scripts/validate_assets.py \
  --asset-root assets/fedbridge-hetero5-assets \
  --scene hetero5 \
  --verify-manifest docs/hetero5_asset_manifest.csv
```

For a new archival inventory, request a CSV containing byte sizes and SHA-256 digests:

```bash
python scripts/validate_assets.py \
  --asset-root assets/fedbridge-hetero5-assets \
  --scene hetero5 \
  --write-manifest asset_manifest.csv
```

All paths stored in the manifest are relative to `<asset-root>`; the validator does not print or record the asset root itself.

## Dataset citation

Please cite the official dataset paper when using MovieLens:

> F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* **5**, Article 19 (2015). https://doi.org/10.1145/2827872

Dataset page: [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/).
