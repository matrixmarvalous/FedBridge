# Third-party notices

<!-- Codex-added 2026-08-14: identify vendored projects and the excluded dataset without replacing their license texts. -->

This EasyRL4Rec experiment package contains or derives from the components below. This notice is informational and does not replace their complete license texts, which are distributed at the top level under `licenses/`.

## EasyRL4Rec

- Project: EasyRL4Rec
- Source: <https://github.com/chongminggao/EasyRL4Rec>
- Recorded baseline: `8cecd007d4da610a31647406cf079e5d9143df53`
- License: MIT
- Copyright notice: Copyright (c) 2023 Chongming GAO
- Included scope: adapted `src/core` Python source, shared policy utilities and experiment runners

The release contains research and portability modifications.
`SOURCE_MANIFEST.csv` records the copied source files and their checksums.

## Tianshou

- Project: Tianshou
- Source: <https://github.com/thu-ml/tianshou>
- Recorded local-fork baseline: `d5337368998a93ff2c65f4bb4e36f19e5d624707`
- Vendored version: 0.5.1
- License: MIT
- Copyright notice: Copyright (c) 2022 Tianshou contributors
- Included scope: vendored Python package plus the experiment-specific federated policy and trainer extensions

## DeepCTR-Torch

- Project: DeepCTR-Torch
- Source: <https://github.com/shenweichen/DeepCTR-Torch>
- Vendored version: 0.2.9
- License: Apache License 2.0
- Included scope: vendored Python package used by the frozen DeepFM user-model ensembles

Modified DeepCTR-Torch files, if any, must retain prominent change notices as required by Apache-2.0. The audited upstream tree did not contain a separate `NOTICE` file.

## MovieLens-1M data

The prepared `hetero5` data and user-model assets are staged as attachments to the private repository's [asset release](https://github.com/matrixmarvalous/FedBridge/releases/tag/recommender-hetero5-v1), not in Git history. The `homo3` assets are not provided. MovieLens-1M is distributed by GroupLens under dataset-specific terms, not under this code package's licenses. The original dataset is available from the [official MovieLens 1M page](https://grouplens.org/datasets/movielens/1m/). Review its terms and confirm permission before publicly redistributing the prepared assets. Private staging does not grant redistribution rights.

Dataset citation:

> F. Maxwell Harper and Joseph A. Konstan. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems* **5**, Article 19 (2015). <https://doi.org/10.1145/2827872>.

## Release authors' code

New orchestration, validation, aggregation, plotting and documentation files carry dated `Codex-added` comments for auditability. Their repository-level licensing should be declared by the project owner before public deposit.
