# Intra-Store Stock Consolidation Pro

High-speed Streamlit application for moving barcode-level stock from slow-selling stores to high-opportunity stores while respecting existing store clusters, Weeks of Cover thresholds, and donor size-curve protection.

## Required files

Only two CSV files are required:

1. `Category Wise Stock - For Consolidation.csv`: `Location Code`, `Country`, `Item Barcode`, and `Available Qty` are the required stock fields.
2. `Sale Data- with OFP Fields -Consol.csv`: `Code`, `Country`, `Item Barcode`, and `Net Sales Qty` are the required sales fields. Inventory quantity is never requested from this file.

These exact headings are auto-detected. Advanced mapping stays hidden unless a required heading is missing. Since the supplied files have no cluster field, `Country` is used as the default transfer cluster to prevent cross-country transfers.

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

Identifier columns—including barcode, store, cluster, style, colour and size—are forced to text. Mixed numeric/alphanumeric barcodes such as `12345678` and `JY11114015628` can therefore coexist safely.

For the best speed, use UTF-8 CSV files with simple comma delimiters and avoid opening the same files in Excel while uploading them.
