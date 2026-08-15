# Intra-Store Stock Consolidation Pro

High-speed Streamlit application for moving barcode-level stock from slow-selling stores to high-opportunity stores while respecting existing store clusters, Weeks of Cover thresholds, and donor size-curve protection.

## Required files

Only two CSV files are required:

1. `Inventory.csv`: store, cluster, barcode, inventory quantity. Style, colour, size, hierarchy, store name and age days are recommended.
2. `30days sales.csv`: store, cluster, barcode, sales quantity. A date column is recommended; otherwise the file must already contain the chosen sales window.

Column names are auto-detected and can be manually mapped in the interface.

## Windows installation

```bat
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

## Calculation rules

- Velocity = sales quantity / selected days × 7.
- Recipient need = target-WOC units minus current inventory.
- Donor availability = inventory above donor maximum WOC; zero-selling inventory follows its own keep-units rule.
- Transfers first match the same barcode inside the same supplied cluster.
- Recipients are ranked by missing-size priority, velocity and shortage.
- Donors are ranked from lowest velocity upward.
- Optional cross-cluster fallback is applied only after same-cluster supply is exhausted.
- Donor size curves can retain one unit per active size to prevent breaking a productive run.

## Performance design

Polars performs multi-threaded CSV parsing and aggregation. Only the aggregated store-barcode position is converted to Pandas for the allocation loop and Streamlit display. Bad CSV rows are never silently discarded. Uploaded files and aggregations are cached, while stale results are blocked when files or rules change.

For the best speed, use UTF-8 CSV files with simple comma delimiters and avoid opening the same files in Excel while uploading them.
