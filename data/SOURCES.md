# Data provenance

The photo-bay, MSA and facilities modules use **only generated data**. `fabstack.sim_photo`, `fabstack.msa`, and `fabstack.utilities` contain their assumptions and seeds; generated CSV/JSON/DuckDB outputs live under ignored `runs/`. These inputs are not empirical fab or facilities measurements.

## UCI SECOM yield baseline, added 18 October 2026

- **Authoritative source:** [UCI Machine Learning Repository, SECOM dataset 179](https://archive.ics.uci.edu/dataset/179/secom), donated by Michael McCann and Adrian Johnston. Citation: McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54305.
- **Archive URL:** `https://archive.ics.uci.edu/static/public/179/secom.zip`; downloaded on 2026-10-18. SHA-256: `eea568baf3c2229096d7d294cf0b096b5502bd96d92c0b80a65b84714059be8e`. The downloader verifies this hash before accepting the archive. Raw files stay in ignored `data/raw/` and are not published in the project repo.
- **License:** CC BY 4.0, as stated on UCI's dataset page; credit appears here and in the result record.
- **Grain and labels:** 1,567 records, 590 numeric sensor columns in `secom.data`, and one `secom_labels.data` line per record. UCI defines `-1` as pass and `1` as in-house test fail. The label file also has a timestamp. The code sorts by that timestamp and excludes it from predictors.
- **Split and limitation:** First 80% by timestamp for development, last 20% for a single held-out check. Five stratified folds within development fit missingness selection, imputation and variance screening afresh. The timestamp is the test point's recorded time, not proof of prospective deployment. Sensor IDs have no physical meanings here; model attribution is not causal yield diagnosis.

The original plan also mentions WM-811K wafer maps, NASA C-MAPSS remaining-useful-life data, and building energy data such as BDG2. Those have not been downloaded, licensed, cleaned, split or benchmarked in this repository. Before adding another source, document its authoritative landing page, version/checksum, license, download date, label definition, entity/time split, leakage controls and intended use. No real-data metric is claimed for those modules.
