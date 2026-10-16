# Data provenance

This implementation currently uses **only generated data**. `fabstack.sim_photo`, `fabstack.msa`, and `fabstack.utilities` contain the assumptions and seeds; generated CSV/JSON/DuckDB outputs live under ignored `runs/`. The synthetic input is not an empirical fab or facilities dataset.

The original plan mentions SECOM process/yield data, wafer map data, NASA C-MAPSS remaining-useful-life data, and building energy data such as BDG2. Those have not been downloaded, licensed, cleaned, split or benchmarked in this repository. Before adding any source, document the authoritative landing page, version/checksum, license, download date, label definition, entity/time split, leakage controls and intended use. No public-data metric is claimed here.
