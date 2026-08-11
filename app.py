"""
Intra-Store Stock Consolidation Engine — OFP/BBZ Edition (Performance-tuned)
Deploy on Streamlit Community Cloud: set this file (app.py) as the main file.
Run locally:  streamlit run app.py
"""

import io
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans

st.set_page_config(page_title="Intra-Store Stock Consolidation", layout="wide")

# ----------------------------------------------------------------------------------
# Generic helpers
# ----------------------------------------------------------------------------------

def normalize_header(name):
    return str(name).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def find_col(columns, target_names):
    """Exact (normalized) match first, then startswith fallback. target_names is a priority list."""
    norm_map = {normalize_header(c): c for c in columns}
    for t in target_names:
        nt = normalize_header(t)
        if nt in norm_map:
            return norm_map[nt]
    for t in target_names:
        nt = normalize_header(t)
        for nc, orig in norm_map.items():
            if nc.startswith(nt):
                return orig
    return None


def resolve_mapping(columns, candidates):
    resolved = {}
    for field, names in candidates.items():
        resolved[field] = find_col(columns, names)
    return resolved


def normalize_code(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def to_numeric_clean(series):
    return pd.to_numeric(series.astype(str).str.strip(), errors="coerce")


class Timer:
    """Lightweight stage-by-stage profiler shown in the UI."""
    def __init__(self):
        self.timings = {}

    def run(self, label, func, *args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        self.timings[label] = time.perf_counter() - t0
        return result


# ----------------------------------------------------------------------------------
# Fast, column-limited CSV / Excel reading
# ----------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_header(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes), encoding="utf-8-sig", nrows=0).columns.tolist()


@st.cache_data(show_spinner=False)
def read_csv_full(file_bytes, usecols, dtype, parse_dates):
    return pd.read_csv(
        io.BytesIO(file_bytes), encoding="utf-8-sig",
        usecols=usecols if usecols else None,
        dtype=dtype if dtype else None,
        parse_dates=parse_dates if parse_dates else None,
        low_memory=False,
    )


@st.cache_data(show_spinner=False)
def read_excel_cached(file_bytes, name, sheet_name=0):
    return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name)


# ----------------------------------------------------------------------------------
# Location Master consolidation (Column C = old/BBZ code, Column I = new/OFP code)
# ----------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_location_map(loc_df):
    old_col = find_col(loc_df.columns, ["Code"])
    new_col = find_col(loc_df.columns, ["Location Code"])
    name_col = find_col(loc_df.columns, ["Location Name"])
    country_col = find_col(loc_df.columns, ["Country"])
    brand_col = find_col(loc_df.columns, ["Store Brand"])
    status_col = find_col(loc_df.columns, ["Status"])

    df = loc_df.copy()
    df["_old"] = df[old_col].apply(normalize_code)
    df["_new"] = df[new_col].apply(normalize_code)

    code_map = {}
    for old, new in zip(df["_old"], df["_new"]):
        if old is not None and new is not None:
            code_map[old] = new

    name_map = {}
    if name_col:
        for new, nm in zip(df["_new"], df[name_col]):
            if new is not None and pd.notna(nm) and new not in name_map:
                name_map[new] = str(nm).strip()

    country_map, brand_map, status_map = {}, {}, {}
    for _, row in df.iterrows():
        new = row["_new"]
        if new is None:
            continue
        if country_col and new not in country_map and pd.notna(row[country_col]):
            country_map[new] = row[country_col]
        if brand_col and new not in brand_map and pd.notna(row[brand_col]):
            brand_map[new] = row[brand_col]
        if status_col and new not in status_map and pd.notna(row[status_col]):
            status_map[new] = row[status_col]

    return {
        "code_map": code_map, "name_map": name_map,
        "country_map": country_map, "brand_map": brand_map, "status_map": status_map,
    }


def apply_consolidation(df, store_code_col, loc_maps):
    df = df.copy()
    df["raw_store_code"] = df[store_code_col].apply(normalize_code)
    df["store_id"] = df["raw_store_code"].apply(lambda c: loc_maps["code_map"].get(c, c))
    return df


# ----------------------------------------------------------------------------------
# Column mapping for Sales & Stock files
# ----------------------------------------------------------------------------------

SALES_FIELD_CANDIDATES = {
    "store_code": ["Code"],
    "store_name": ["Location"],
    "country": ["Country"],
    "date": ["Date"],
    "barcode": ["Item Barcode"],
    "style": ["Item Style Code"],
    "colour": ["Item Color", "Item Colour"],
    "size": ["Ofp Size", "Size"],
    "dept": ["Ofp Dept Name"],
    "subdepartment"],
    "cls": ["Ofp Class Name"],
    "subclass": ["Ofp Sub Class Name"],
    "subbrand": ["Item Sub Brand"],
    "sales_qty": ["Net Sales Qty"],
}

STOCK_FIELD_CANDIDATES = {
    "store_code": ["Location Code"],
    "store_name": ["Location"],
    "country": ["Country"],
    "store_brand": ["Store Brand"],
    "group": ["Ofp Group Name"],
    "division": ["Item Division"],
    "barcode": ["Item Barcode"],
    "style": ["Item Style Code"],
    "size": ["Ofp Size", "Size"],
    "dept": ["Ofp Dept Name"],
    "subdept": ["Ofp Sub Department"],
    "cls": ["Ofp Class Name"],
    "subclass": ["Ofp Sub Class Name"],
    "subbrand": ["Item Sub Brand"],
    "inv_qty": ["Available Qty"],
    "last_received": ["Last recieved date store", "Last received date store"],
    "max_ageing": ["Max (Ageing Days)", "Max Ageing Days"],
    "age_days_reported": ["Age Days", "Ageing Days", "Aging Days", "Days Since Last Sale",
                           "Stock Age Days", "Age In Days"],
}


def build_read_plan(columns, candidates, force_str_fields):
    """Resolve mapping, then build usecols/dtype for a fast, column-limited read."""
    mapping = resolve_mapping(columns, candidates)
    usecols = [c for c in mapping.values() if c]
    dtype = {}
    for field in force_str_fields:
        col = mapping.get(field)
        if col:
            dtype[col] = str
    return mapping, usecols, dtype


@st.cache_data(show_spinner=False)
def parse_sales(sales_bytes, mapping):
    m = mapping
    usecols = [c for c in m.values() if c]
    dtype = {}
    for field in ["store_code", "barcode"]:
        if m.get(field):
            dtype[m[field]] = str
    raw = read_csv_full(sales_bytes, usecols, dtype, None)

    out = pd.DataFrame()
    for field in ["store_code", "country", "date", "barcode", "style", "colour", "size",
                  "dept", "subdept", "cls", "subclass", "subbrand", "sales_qty"]:
        col = m.get(field)
        out[field] = raw[col] if col and col in raw.columns else np.nan

    out["sales_qty"] = to_numeric_clean(out["sales_qty"]).fillna(0)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["barcode"] = out["barcode"].astype(str).str.strip()
    out["raw_store_code"] = out["store_code"].apply(normalize_code)
    return out


@st.cache_data(show_spinner=False)
def parse_stock(stock_bytes, mapping):
    m = mapping
    usecols = [c for c in m.values() if c]
    dtype = {}
    for field in ["store_code", "barcode"]:
        if m.get(field):
            dtype[m[field]] = str
    raw = read_csv_full(stock_bytes, usecols, dtype, None)

    out = pd.DataFrame()
    for field in ["store_code", "country", "store_brand", "group", "division", "barcode", "style",
                  "size", "dept", "subdept", "cls", "subclass", "subbrand", "inv_qty",
                  "last_received", "max_ageing", "age_days_reported"]:
        col = m.get(field)
        out[field] = raw[col] if col and col in raw.columns else np.nan

    out["inv_qty"] = to_numeric_clean(out["inv_qty"]).fillna(0)
    out["max_ageing"] = to_numeric_clean(out["max_ageing"])
    out["age_days_reported"] = to_numeric_clean(out["age_days_reported"])
    out["last_received"] = pd.to_datetime(out["last_received"], errors="coerce", dayfirst=True)
    out["barcode"] = out["barcode"].astype(str).str.strip()
    out["colour"] = np.nan  # stock feed has no colour — backfilled later from sales
    out["raw_store_code"] = out["store_code"].apply(normalize_code)
    return out


@st.cache_data(show_spinner=False)
def apply_consolidation_fast(df, loc_maps):
    df = df.copy()
    code_map = loc_maps["code_map"]
    df["store_id"] = df["raw_store_code"].map(code_map).fillna(df["raw_store_code"])
    return df


@st.cache_data(show_spinner=False)
def build_barcode_attrs(sales_parsed):
    """barcode -> best-known style/colour/size/dept hierarchy, learned from the sales feed.
    Vectorized via groupby.first() on pre-sorted non-null values — no row-wise apply."""
    attrs_cols = ["style", "colour", "size", "dept", "subdept", "cls", "subclass", "subbrand"]
    tmp = sales_parsed[["barcode"] + attrs_cols].copy()
    for c in attrs_cols:
        tmp[c] = tmp[c].replace("", np.nan)
    attrs = tmp.sort_values("barcode").groupby("barcode", as_index=False)[attrs_cols].first()
    return attrs


@st.cache_data(show_spinner=False)
def backfill_attrs(stock_parsed, barcode_attrs):
    df = stock_parsed.merge(barcode_attrs, on="barcode", how="left", suffixes=("", "_from_sales"))
    for col in ["style", "colour", "size", "dept", "subdept", "cls", "subclass", "subbrand"]:
        src = f"{col}_from_sales"
        if src in df.columns:
            df[col] = df[col].where(df[col].notna() & (df[col] != ""), df[src])
    drop_cols = [c for c in df.columns if c.endswith("_from_sales")]
    return df.drop(columns=drop_cols)


# ----------------------------------------------------------------------------------
# Core metrics engine
# ----------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def build_master(stock_parsed, sales_parsed, as_of_date):
    stock_agg = stock_parsed.groupby(["store_id", "barcode"], as_index=False).agg(
        inv_qty=("inv_qty", "sum"),
        style=("style", "first"), colour=("colour", "first"), size=("size", "first"),
        dept=("dept", "first"), subdept=("subdept", "first"),
        cls=("cls", "first"), subclass=("subclass", "first"), subbrand=("subbrand", "first"),
        country=("country", "first"), store_brand=("store_brand", "first"),
        last_received=("last_received", "max"), max_ageing=("max_ageing", "max"),
        age_days_reported=("age_days_reported", "max"),
    )

    date_min, date_max = sales_parsed["date"].min(), sales_parsed["date"].max()
    if pd.notna(date_min) and pd.notna(date_max) and date_max > date_min:
        period_days = max(1, (date_max - date_min).days + 1)
    else:
        period_days = 30

    sales_agg = sales_parsed.groupby(["store_id", "barcode"], as_index=False).agg(
        sales_qty=("sales_qty", "sum"),
        style=("style", "first"), colour=("colour", "first"), size=("size", "first"),
        dept=("dept", "first"), subdept=("subdept", "first"),
        cls=("cls", "first"), subclass=("subclass", "first"), subbrand=("subbrand", "first"),
        country=("country", "first"),
    )

    master = pd.merge(stock_agg, sales_agg, on=["store_id", "barcode"], how="outer",
                       suffixes=("", "_sales"))
    for col in ["style", "colour", "size", "dept", "subdept", "cls", "subclass", "subbrand", "country"]:
        alt = f"{col}_sales"
        if alt in master.columns:
            master[col] = master[col].where(master[col].notna(), master[alt])
            master.drop(columns=[alt], inplace=True)
        master[col] = master[col].fillna("UNKNOWN")

    master["inv_qty"] = master["inv_qty"].fillna(0)
    master["sales_qty"] = master["sales_qty"].fillna(0)
    master["sales_period_days"] = period_days

    computed_age = (pd.Timestamp(as_of_date) - master["last_received"]).dt.days
    master["ageing_days"] = master["age_days_reported"].where(
        master["age_days_reported"].notna(), computed_age
    )
    master["ageing_days"] = master["ageing_days"].fillna(0).clip(lower=0)

    return master


@st.cache_data(show_spinner=False)
def compute_metrics(master, safety_woc, max_woc, dead_stock_days):
    df = master.copy()
    period_days = df["sales_period_days"].iloc[0] if len(df) else 30
    df["velocity_per_day"] = df["sales_qty"] / period_days
    df["velocity_per_week"] = df["velocity_per_day"] * 7.0

    df["weeks_of_cover"] = np.where(
        df["velocity_per_week"] > 0, df["inv_qty"] / df["velocity_per_week"], np.inf
    )
    df["weeks_of_cover_display"] = df["weeks_of_cover"].replace(np.inf, 99).round(1)

    df["safety_stock_units"] = df["velocity_per_week"] * safety_woc
    df["excess_threshold_units"] = df["velocity_per_week"] * max_woc

    df["shortage_units"] = np.maximum(0, df["safety_stock_units"] - df["inv_qty"]).round(0)
    df["excess_units"] = np.maximum(0, df["inv_qty"] - df["excess_threshold_units"]).round(0)

    df["velocity_rank_pct"] = df.groupby("barcode")["velocity_per_week"].rank(pct=True)

    df["aged_flag"] = (df["max_ageing"].notna()) & (df["ageing_days"] > df["max_ageing"])

    # Hard rule: aged beyond threshold + zero sales at that location = force-release
    df["dead_stock_flag"] = (
        (df["ageing_days"] > dead_stock_days) &
        (df["velocity_per_week"] == 0) &
        (df["inv_qty"] > 0)
    )

    df["excess_units"] = np.where(df["velocity_per_week"] == 0, df["inv_qty"], df["excess_units"])
    df["safety_stock_units"] = np.where(df["velocity_per_week"] == 0, 0, df["safety_stock_units"])

    df["is_donor"] = (df["excess_units"] > 0) | df["dead_stock_flag"]
    df["is_recipient"] = df["shortage_units"] > 0
    return df


@st.cache_data(show_spinner=False)
def cluster_stores(df, n_clusters, top_n_styles=60):
    # Cap the pivot to the top-N styles by total velocity; bucket the rest as OTHER
    # to keep the KMeans feature matrix small even with thousands of SKUs.
    style_rank = df.groupby("style")["velocity_per_week"].sum().sort_values(ascending=False)
    top_styles = set(style_rank.head(top_n_styles).index)
    df = df.copy()
    df["_style_bucket"] = np.where(df["style"].isin(top_styles), df["style"], "OTHER")

    pivot = df.pivot_table(index="store_id", columns="_style_bucket", values="velocity_per_week",
                            aggfunc="sum", fill_value=0)
    beh = df.groupby("store_id").agg(
        _total_sales=("sales_qty", "sum"), _total_inv=("inv_qty", "sum")
    )
    demo = df.groupby("store_id").agg(
        country=("country", "first"), store_brand=("store_brand", "first")
    )
    demo_dummies = pd.get_dummies(demo[["country", "store_brand"]], prefix=["ctry", "brand"])

    feat = pivot.join(beh, how="left").join(demo_dummies, how="left").fillna(0)

    n_stores = feat.shape[0]
    k = max(1, min(n_clusters, n_stores))
    if n_stores <= 1:
        cluster_map = pd.DataFrame({"store_id": feat.index, "cluster_id": 0})
    else:
        scaled = StandardScaler().fit_transform(feat.values)
        if n_stores > 300:
            km = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=256)
        else:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled)
        cluster_map = pd.DataFrame({"store_id": feat.index, "cluster_id": labels})

    out = df.merge(cluster_map, on="store_id", how="left")
    out["cluster_id"] = out["cluster_id"].fillna(0).astype(int)
    out.drop(columns=["_style_bucket"], inplace=True)
    return out


@st.cache_data(show_spinner=False)
def compute_size_curve(df, qty_field="inv_qty"):
    present = df[df[qty_field] > 0]

    full_counts = present.groupby(["style", "colour"])["size"].nunique().rename("full_size_count")
    store_counts = present.groupby(["store_id", "style", "colour"])["size"].nunique().rename("present_size_count")
    store_counts = store_counts.reset_index()

    curve = store_counts.merge(full_counts, on=["style", "colour"], how="left")
    curve["completeness_pct"] = np.where(
        curve["full_size_count"] > 0,
        (curve["present_size_count"] / curve["full_size_count"] * 100).round(1), np.nan,
    )

    # Only compute the expensive "which sizes are missing" text for incomplete runs
    curve["missing_sizes"] = ""
    incomplete = curve[curve["present_size_count"] < curve["full_size_count"]]
    if not incomplete.empty:
        full_sets = present.groupby(["style", "colour"])["size"].apply(lambda s: frozenset(s.unique()))
        store_sets = present.groupby(["store_id", "style", "colour"])["size"].apply(lambda s: frozenset(s.unique()))
        missing_map = {}
        for key in incomplete[["store_id", "style", "colour"]].itertuples(index=False):
            full_s = full_sets.get((key.style, key.colour), frozenset())
            store_s = store_sets.get((key.store_id, key.style, key.colour), frozenset())
            missing_map[(key.store_id, key.style, key.colour)] = ", ".join(sorted(full_s - store_s))
        curve.loc[incomplete.index, "missing_sizes"] = incomplete.apply(
            lambda r: missing_map.get((r["store_id"], r["style"], r["colour"]), ""), axis=1
        )

    return curve[["store_id", "style", "colour", "full_size_count", "present_size_count",
                  "completeness_pct", "missing_sizes"]]


def flag_size_curve_priority(df, curve_before):
    """Vectorized version — no per-row Python function calls."""
    lookup = curve_before[["store_id", "style", "colour", "present_size_count", "full_size_count"]]
    merged = df.merge(lookup, on=["store_id", "style", "colour"], how="left", suffixes=("", "_curve"))

    is_zero_inv = merged["inv_qty"] <= 0
    completes_run = is_zero_inv & merged["full_size_count"].notna() & \
        (merged["present_size_count"] == merged["full_size_count"] - 1)
    improves_run = is_zero_inv & (merged["present_size_count"].fillna(0) > 0) & ~completes_run

    priority = np.select(
        [completes_run, improves_run],
        [3, 2],
        default=1,
    )
    df = df.copy()
    df["size_curve_priority"] = priority
    return df


@st.cache_data(show_spinner=False)
def run_transfer_engine(df, curve_before, horizon_weeks, allow_cross_cluster, min_transfer_qty):
    df = flag_size_curve_priority(df, curve_before)

    # Integer-encode store/barcode once for fast dict keys instead of raw strings
    stores = pd.Index(df["store_id"].unique())
    barcodes = pd.Index(df["barcode"].unique())
    store_to_int = {s: i for i, s in enumerate(stores)}
    barcode_to_int = {b: i for i, b in enumerate(barcodes)}
    df = df.copy()
    df["store_int"] = df["store_id"].map(store_to_int)
    df["barcode_int"] = df["barcode"].map(barcode_to_int)

    recipients = df[df["is_recipient"] & (df["shortage_units"] >= min_transfer_qty)].copy()
    recipients = recipients.sort_values(
        by=["size_curve_priority", "velocity_per_week", "shortage_units"],
        ascending=[False, False, False],
    )

    donors = df[df["is_donor"]].copy()
    donor_remaining = {
        (r.store_int, r.barcode_int): r.excess_units for r in donors.itertuples()
    }

    # Pre-group donors by (barcode, cluster) so each recipient only scans a tiny,
    # relevant subset instead of every donor that ever carried that barcode.
    donors_sorted = donors.sort_values(
        ["dead_stock_flag", "aged_flag", "velocity_per_week", "ageing_days"],
        ascending=[False, False, True, False],
    )
    donor_groups = {}
    for (barcode_int, cluster_id), g in donors_sorted.groupby(["barcode_int", "cluster_id"]):
        donor_groups[(barcode_int, cluster_id)] = list(
            g[["store_id", "store_int", "cluster_id", "velocity_per_week", "ageing_days",
               "aged_flag", "dead_stock_flag"]].itertuples(index=False)
        )

    donor_all_clusters = {}
    if allow_cross_cluster:
        for barcode_int, g in donors_sorted.groupby("barcode_int"):
            donor_all_clusters[barcode_int] = list(
                g[["store_id", "store_int", "cluster_id", "velocity_per_week", "ageing_days",
                   "aged_flag", "dead_stock_flag"]].itertuples(index=False)
            )

    transfer_records, unfulfilled_records = [], []

    for rec in recipients.itertuples():
        remaining_gap = rec.shortage_units
        candidates = donor_groups.get((rec.barcode_int, rec.cluster_id), [])

        for cand in candidates:
            if remaining_gap <= 0:
                break
            if cand.store_int == rec.store_int:
                continue
            key = (cand.store_int, rec.barcode_int)
            avail = donor_remaining.get(key, 0)
            if avail <= 0:
                continue
            qty = min(avail, remaining_gap)
            if qty <= 0:
                continue

            expected_opportunity = min(qty, rec.velocity_per_week * horizon_weeks)
            transfer_records.append({
                "donor_store": cand.store_id, "recipient_store": rec.store_id,
                "barcode": rec.barcode, "style": rec.style, "colour": rec.colour, "size": rec.size,
                "cluster_match": True,
                "transfer_qty": qty,
                "donor_velocity_per_week": round(cand.velocity_per_week, 2),
                "recipient_velocity_per_week": round(rec.velocity_per_week, 2),
                "donor_dead_stock": bool(cand.dead_stock_flag),
                "donor_aged_stock": bool(cand.aged_flag),
                "donor_ageing_days": cand.ageing_days,
                "size_curve_priority": rec.size_curve_priority,
                "expected_sales_opportunity_units": round(expected_opportunity, 1),
            })
            donor_remaining[key] -= qty
            remaining_gap -= qty

        if remaining_gap > 0 and allow_cross_cluster:
            for cand in donor_all_clusters.get(rec.barcode_int, []):
                if remaining_gap <= 0:
                    break
                if cand.store_int == rec.store_int or cand.cluster_id == rec.cluster_id:
                    continue
                key = (cand.store_int, rec.barcode_int)
                avail = donor_remaining.get(key, 0)
                if avail <= 0:
                    continue
                qty = min(avail, remaining_gap)
                if qty <= 0:
                    continue

                expected_opportunity = min(qty, rec.velocity_per_week * horizon_weeks)
                transfer_records.append({
                    "donor_store": cand.store_id, "recipient_store": rec.store_id,
                    "barcode": rec.barcode, "style": rec.style, "colour": rec.colour, "size": rec.size,
                    "cluster_match": False,
                    "transfer_qty": qty,
                    "donor_velocity_per_week": round(cand.velocity_per_week, 2),
                    "recipient_velocity_per_week": round(rec.velocity_per_week, 2),
                    "donor_dead_stock": bool(cand.dead_stock_flag),
                    "donor_aged_stock": bool(cand.aged_flag),
                    "donor_ageing_days": cand.ageing_days,
                    "size_curve_priority": rec.size_curve_priority,
                    "expected_sales_opportunity_units": round(expected_opportunity, 1),
                })
                donor_remaining[key] -= qty
                remaining_gap -= qty

        if remaining_gap > 0:
            unfulfilled_records.append({
                "recipient_store": rec.store_id, "barcode": rec.barcode,
                "style": rec.style, "colour": rec.colour, "size": rec.size,
                "cluster_id": rec.cluster_id, "unfulfilled_gap_units": round(remaining_gap, 1),
                "recipient_velocity_per_week": round(rec.velocity_per_week, 2),
                "reason": "No cluster-compatible donor stock available" if not allow_cross_cluster
                          else "No donor stock available company-wide",
            })

    return pd.DataFrame(transfer_records), pd.DataFrame(unfulfilled_records)


def build_closing_position(df, transfers_df, horizon_weeks):
    df = df.copy()
    if not transfers_df.empty:
        out_agg = transfers_df.groupby(["donor_store", "barcode"])["transfer_qty"].sum().rename("transferred_out")
        in_agg = transfers_df.groupby(["recipient_store", "barcode"])["transfer_qty"].sum().rename("transferred_in")
        df = df.merge(out_agg, left_on=["store_id", "barcode"], right_index=True, how="left")
        df = df.merge(in_agg, left_on=["store_id", "barcode"], right_index=True, how="left")
    else:
        df["transferred_out"], df["transferred_in"] = np.nan, np.nan
    df["transferred_out"] = df["transferred_out"].fillna(0)
    df["transferred_in"] = df["transferred_in"].fillna(0)
    df["inv_after"] = df["inv_qty"] - df["transferred_out"] + df["transferred_in"]
    df["woc_after"] = np.where(df["velocity_per_week"] > 0, df["inv_after"] / df["velocity_per_week"], np.inf)
    df["woc_after_display"] = df["woc_after"].replace(np.inf, 99).round(1)
    df["projected_sales_horizon"] = (df["velocity_per_week"] * horizon_weeks).round(1)
    return df


# ----------------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------------

st.title("📦 Intra-Store Stock Consolidation Engine")
st.caption("OFP/BBZ store-consolidated · Sell-velocity → size-curve → clustering → threshold-triggered transfers")

with st.sidebar:
    st.header("1. Data")
    sales_file = st.file_uploader("Sales file (Sale Data- with OFP Fields -Consol.csv)", type="csv")
    stock_file = st.file_uploader("Stock file (Category Wise Stock - For Consolidation.csv)", type="csv")
    loc_file = st.file_uploader("Location Master (xlsx) — Col C old code → Col I new code", type=["xlsx", "xls"])

    st.divider()
    st.header("2. Business Rules")
    safety_woc = st.slider("Safety-stock: minimum Weeks of Cover", 0.5, 8.0, 2.0, 0.5)
    max_woc = st.slider("Excess threshold: maximum Weeks of Cover", safety_woc + 0.5, 20.0, 6.0, 0.5)
    n_clusters = st.slider("Number of store clusters", 2, 10, 5)
    horizon_weeks = st.slider("Sales-opportunity horizon (weeks)", 1, 12, 4)
    min_transfer_qty = st.number_input("Minimum gap to trigger a transfer (units)", 1, 50, 1)
    allow_cross_cluster = st.checkbox("Allow cross-cluster fallback for unresolved gaps", value=False)
    as_of_date = st.date_input("As-of date (for ageing calc fallback)", value=pd.Timestamp.today())

    st.divider()
    st.header("3. Dead-Stock Force-Consolidation")
    dead_stock_weeks = st.slider(
        "Dead-stock aging threshold (weeks)", 1, 12, 3,
        help="Stock sitting longer than this at a location, with ZERO sales at that "
             "location, is force-flagged as donor stock regardless of the WOC-excess rule."
    )
    dead_stock_days = dead_stock_weeks * 7

    st.divider()
    st.header("4. Performance Settings")
    top_n_styles = st.slider(
        "Max styles used for clustering features", 20, 200, 60, step=10,
        help="Caps the clustering feature matrix width for very large assortments. "
             "Lower = faster clustering, higher = more style-level nuance in clusters."
    )

    st.divider()
    run_clicked = st.button("🚀 Run Consolidation Engine", type="primary", use_container_width=True)

if not (sales_file and stock_file and loc_file):
    st.info("Upload the **Sales file**, **Stock file**, and **Location Master (xlsx)** in the sidebar to begin.")
    st.stop()

sales_bytes = sales_file.getvalue()
stock_bytes = stock_file.getvalue()
loc_raw = read_excel_cached(loc_file.getvalue(), loc_file.name)

sales_columns = read_csv_header(sales_bytes)
stock_columns = read_csv_header(stock_bytes)
sales_mapping = resolve_mapping(sales_columns, SALES_FIELD_CANDIDATES)
stock_mapping = resolve_mapping(stock_columns, STOCK_FIELD_CANDIDATES)

with st.expander("🔍 Detected column mapping (auto-resolved — expand to verify)"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sales file**")
        st.json(sales_mapping)
    with c2:
        st.markdown("**Stock file**")
        st.json(stock_mapping)
    if not stock_mapping.get("age_days_reported"):
        st.warning(
            "No direct 'Age Days' column detected in the stock file — ageing will be "
            "computed from Last Received Date instead."
        )

if not run_clicked and "results" not in st.session_state:
    st.warning("Review the rules in the sidebar, then click **Run Consolidation Engine**.")
    st.stop()

if run_clicked:
    timer = Timer()
    with st.spinner("Consolidating BBZ/OFP codes, computing velocity, dead-stock, clusters, size curves and transfer plan..."):
        loc_maps = timer.run("Build location map", build_location_map, loc_raw)

        sales_parsed_raw = timer.run("Parse sales file", parse_sales, sales_bytes, sales_mapping)
        stock_parsed_raw0 = timer.run("Parse stock file", parse_stock, stock_bytes, stock_mapping)

        sales_parsed = timer.run("Consolidate BBZ→OFP (sales)", apply_consolidation_fast, sales_parsed_raw, loc_maps)
        stock_parsed_raw = timer.run("Consolidate BBZ→OFP (stock)", apply_consolidation_fast, stock_parsed_raw0, loc_maps)

        barcode_attrs = timer.run("Build barcode attribute lookup", build_barcode_attrs, sales_parsed)
        stock_parsed = timer.run("Backfill stock attributes", backfill_attrs, stock_parsed_raw, barcode_attrs)

        master = timer.run("Build master (stock+sales merge)", build_master, stock_parsed, sales_parsed, as_of_date)
        metrics = timer.run("Compute velocity/WOC/dead-stock metrics", compute_metrics,
                             master, safety_woc, max_woc, dead_stock_days)
        clustered = timer.run("Cluster stores", cluster_stores, metrics, n_clusters, top_n_styles)
        curve_before = timer.run("Compute size curve (before)", compute_size_curve, clustered, "inv_qty")
        transfers_df, unfulfilled_df = timer.run(
            "Run transfer engine", run_transfer_engine,
            clustered, curve_before, horizon_weeks, allow_cross_cluster, min_transfer_qty
        )
        closing = timer.run("Build closing position", build_closing_position, clustered, transfers_df, horizon_weeks)
        after_view = closing.rename(columns={"inv_after": "inv_qty_after"}).assign(inv_qty=lambda d: d["inv_qty_after"])
        curve_after = timer.run("Compute size curve (after)", compute_size_curve, after_view)

        consolidation_report = pd.DataFrame([
            {"old_code": old, "new_code": new, "consolidated_name": loc_maps["name_map"].get(new, "")}
            for old, new in loc_maps["code_map"].items() if old != new
        ])

        st.session_state["results"] = dict(
            clustered=clustered, curve_before=curve_before, curve_after=curve_after,
            transfers_df=transfers_df, unfulfilled_df=unfulfilled_df, closing=closing,
            consolidation_report=consolidation_report, sales_parsed=sales_parsed,
            dead_stock_days=dead_stock_days, timings=timer.timings,
        )

res = st.session_state["results"]
clustered = res["clustered"]; curve_before = res["curve_before"]; curve_after = res["curve_after"]
transfers_df = res["transfers_df"]; unfulfilled_df = res["unfulfilled_df"]
closing = res["closing"]; consolidation_report = res["consolidation_report"]
dead_stock_days_used = res["dead_stock_days"]
timings = res.get("timings", {})

tabs = st.tabs([
    "📊 Overview", "⏱️ Performance", "🏬 Location Consolidation", "💀 Dead Stock (Aged, No Sales)",
    "⚡ Sell Velocity", "📅 WOC & Safety Stock", "🧭 Store Clusters", "📐 Size Curve",
    "🏆 Donor/Recipient Rankings", "🔁 Transfer Plan", "🚫 Unfulfilled Gaps",
])
(tab_overview, tab_perf, tab_consol, tab_dead, tab_velocity, tab_woc, tab_cluster,
 tab_curve, tab_rank, tab_transfer, tab_gaps) = tabs

with tab_overview:
    dead_units_total = int(clustered.loc[clustered["dead_stock_flag"], "inv_qty"].sum())
    dead_units_placed = int(
        transfers_df.loc[transfers_df["donor_dead_stock"] == True, "transfer_qty"].sum()
    ) if not transfers_df.empty and "donor_dead_stock" in transfers_df.columns else 0

    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("Consolidated Stores", clustered["store_id"].nunique())
    k2.metric("SKUs (barcodes)", clustered["barcode"].nunique())
    k3.metric("Total Shortage (units)", int(clustered["shortage_units"].sum()))
    k4.metric("Total Excess (units)", int(clustered["excess_units"].sum()))
    k5.metric("Dead Stock Units (Aged & No Sale)", dead_units_total)
    k6.metric("Recommended Transfer Qty", int(transfers_df["transfer_qty"].sum()) if not transfers_df.empty else 0)
    k7.metric("Unfulfilled Gap Lines", len(unfulfilled_df))

    if dead_units_total > 0:
        pct_placed = (dead_units_placed / dead_units_total * 100) if dead_units_total else 0
        st.info(
            f"**{dead_units_total:,} units** are sitting idle beyond {dead_stock_days_used} days with zero "
            f"sales at their current location. **{dead_units_placed:,} units ({pct_placed:.0f}%)** were "
            f"matched to a needy store in this run — see the 💀 Dead Stock tab for the rest."
        )

    c1, c2 = st.columns(2)
    with c1:
        top_donor = clustered.groupby("store_id")["excess_units"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_donor, x="store_id", y="excess_units", title="Top 10 Donor Stores (Excess Units)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        top_rec = clustered.groupby("store_id")["shortage_units"].sum().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(top_rec, x="store_id", y="shortage_units", title="Top 10 Recipient Stores (Shortage Units)",
                     color_discrete_sequence=["#d62728"])
        st.plotly_chart(fig, use_container_width=True)

    if not transfers_df.empty:
        st.success(
            f"Expected sales opportunity from recommended transfers: "
            f"**{transfers_df['expected_sales_opportunity_units'].sum():,.0f} units** "
            f"over the next {horizon_weeks} weeks."
        )

with tab_perf:
    st.markdown("#### ⏱️ Stage-by-Stage Runtime")
    st.caption("Use this to see exactly where time is going on your real dataset. If one stage dominates, tell me which one and I'll optimize it further.")
    if timings:
        perf_df = pd.DataFrame(
            [{"stage": k, "seconds": round(v, 2)} for k, v in timings.items()]
        ).sort_values("seconds", ascending=False)
        total = perf_df["seconds"].sum()
        st.metric("Total run time", f"{total:.1f} sec")
        st.dataframe(perf_df, use_container_width=True, height=350)
        fig = px.bar(perf_df, x="stage", y="seconds", title="Time per Stage (seconds)")
        fig.update_layout(xaxis_tickangle=-40)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run the engine to see a performance breakdown here.")

with tab_consol:
    st.markdown("#### BBZ ↔ OFP Code Pairs Consolidated")
    st.caption("Every row below is a store whose old (BBZ) code and new (OFP) code were merged into a single stock/sales record before any calculation ran.")
    if consolidation_report.empty:
        st.info("No differing old→new code pairs found in the Location Master (all codes matched their own new code).")
    else:
        st.dataframe(consolidation_report, use_container_width=True, height=350)
    st.download_button("⬇️ Download Consolidation Report (CSV)",
                        consolidation_report.to_csv(index=False).encode(),
                        "location_consolidation_report.csv", "text/csv")

with tab_dead:
    st.markdown(f"#### Dead Stock — Aged > {dead_stock_days_used} days ({dead_stock_days_used // 7} weeks) with Zero Sales at Location")
    st.caption("This stock is force-flagged as donor-eligible regardless of the WOC-excess threshold, and given top priority in the transfer engine.")

    dead_df = clustered[clustered["dead_stock_flag"]][
        ["store_id", "barcode", "style", "colour", "size", "inv_qty", "ageing_days",
         "cluster_id", "dept", "subdept"]
    ].sort_values("ageing_days", ascending=False)

    if dead_df.empty:
        st.success("No stock currently meets the dead-stock threshold.")
    else:
        st.dataframe(dead_df, use_container_width=True, height=300)
        st.download_button("⬇️ Download Dead Stock Report (CSV)", dead_df.to_csv(index=False).encode(),
                            "dead_stock_report.csv", "text/csv")

        if not transfers_df.empty and "donor_dead_stock" in transfers_df.columns:
            st.markdown("#### Where This Dead Stock Was Routed")
            routed = transfers_df[transfers_df["donor_dead_stock"] == True]
            if routed.empty:
                st.warning("None of this dead stock found a matching recipient in this run — check the reasons below.")
            else:
                st.dataframe(routed, use_container_width=True, height=300)

        stranded_barcodes = set(dead_df["barcode"]) - (
            set(transfers_df.loc[transfers_df["donor_dead_stock"] == True, "barcode"])
            if not transfers_df.empty and "donor_dead_stock" in transfers_df.columns else set()
        )
        if stranded_barcodes:
            st.markdown("#### Still Stranded (no matching recipient found)")
            stranded = dead_df[dead_df["barcode"].isin(stranded_barcodes)]
            st.dataframe(stranded, use_container_width=True, height=250)
            st.caption(
                "These items had no store with a shortage for the same barcode within the allowed "
                "cluster scope. Consider enabling 'Allow cross-cluster fallback' or reviewing for markdown."
            )

with tab_velocity:
    st.markdown("#### Sales Velocity by Barcode / Style / Colour / Size")
    view = st.selectbox("Roll up by", ["barcode", "style", "colour", "size"])
    roll = clustered.groupby(view).agg(
        total_sales=("sales_qty", "sum"), avg_velocity_per_week=("velocity_per_week", "mean"),
        total_inv=("inv_qty", "sum"),
    ).reset_index().sort_values("total_sales", ascending=False)
    st.dataframe(roll, use_container_width=True, height=350)
    fig = px.bar(roll.head(20), x=view, y="avg_velocity_per_week", title=f"Avg Velocity/Week by {view} (Top 20)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Store-SKU Level Detail")
    st.dataframe(
        clustered[["store_id", "barcode", "style", "colour", "size", "sales_qty",
                   "velocity_per_day", "velocity_per_week", "velocity_rank_pct"]]
        .sort_values("velocity_per_week", ascending=False),
        use_container_width=True, height=350,
    )

with tab_woc:
    st.markdown("#### Weeks of Cover, Safety Stock & Excess")
    store_overall = clustered.groupby("store_id").agg(
        total_inv=("inv_qty", "sum"), total_velocity_week=("velocity_per_week", "sum")
    ).reset_index()
    store_overall["store_overall_woc"] = np.where(
        store_overall["total_velocity_week"] > 0,
        (store_overall["total_inv"] / store_overall["total_velocity_week"]).round(1), np.inf,
    )
    st.markdown("**Store-Level Overall WOC**")
    st.dataframe(store_overall, use_container_width=True, height=250)

    st.markdown("**SKU-Level Detail** (aged_flag = past per-line Max Ageing Days · dead_stock_flag = aged + zero local sales)")
    st.dataframe(
        clustered[["store_id", "barcode", "style", "colour", "size", "inv_qty", "velocity_per_week",
                   "weeks_of_cover_display", "safety_stock_units", "shortage_units", "excess_units",
                   "ageing_days", "aged_flag", "dead_stock_flag"]].sort_values("shortage_units", ascending=False),
        use_container_width=True, height=350,
    )
    fig = px.histogram(clustered, x="weeks_of_cover_display", nbins=30, title="Distribution of Weeks of Cover")
    st.plotly_chart(fig, use_container_width=True)

with tab_cluster:
    st.markdown("#### Store Clusters (sales-behavior + country/store-brand demographics)")
    cluster_summary = clustered.groupby("cluster_id").agg(
        stores=("store_id", "nunique"), total_sales=("sales_qty", "sum"),
        avg_velocity_week=("velocity_per_week", "mean"),
    ).reset_index()
    st.dataframe(cluster_summary, use_container_width=True)
    store_cluster_map = clustered[["store_id", "cluster_id", "country", "store_brand"]].drop_duplicates().sort_values("cluster_id")
    st.dataframe(store_cluster_map, use_container_width=True, height=300)

with tab_curve:
    st.markdown("#### Size-Curve Completeness — Before Transfer")
    st.dataframe(curve_before.sort_values("completeness_pct"), use_container_width=True, height=300)
    st.markdown("#### Size-Curve Completeness — After Transfer")
    st.dataframe(curve_after.sort_values("completeness_pct"), use_container_width=True, height=300)
    fig = px.histogram(curve_before, x="completeness_pct", nbins=20, title="Size-Curve Completeness Before (%)")
    st.plotly_chart(fig, use_container_width=True)

with tab_rank:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Donor Store Ranking** (dead-stock first, then slow-moving / excess)")
        donor_rank = (clustered[clustered["is_donor"]]
                      .sort_values(["dead_stock_flag", "velocity_per_week", "ageing_days"],
                                   ascending=[False, True, False])
                      [["store_id", "barcode", "style", "colour", "size", "cluster_id",
                        "velocity_per_week", "excess_units", "ageing_days", "aged_flag", "dead_stock_flag"]])
        st.dataframe(donor_rank, use_container_width=True, height=400)
    with c2:
        st.markdown("**Recipient Store Ranking** (fast-moving / shortage first)")
        rec_rank = (clustered[clustered["is_recipient"]]
                    .sort_values(["velocity_per_week", "shortage_units"], ascending=[False, False])
                    [["store_id", "barcode", "style", "colour", "size", "cluster_id",
                      "velocity_per_week", "shortage_units"]])
        st.dataframe(rec_rank, use_container_width=True, height=400)

with tab_transfer:
    st.markdown("#### Recommended Transfers")
    if transfers_df.empty:
        st.info("No transfers were recommended with the current rules/data.")
    else:
        st.dataframe(transfers_df.sort_values("transfer_qty", ascending=False), use_container_width=True, height=350)
        st.download_button("⬇️ Download Transfer Plan (CSV)", transfers_df.to_csv(index=False).encode(),
                            "transfer_recommendations.csv", "text/csv")

        st.markdown("#### Recipient Impact Summary (Closing Position)")
        impact_cols = ["store_id", "barcode", "style", "colour", "size", "inv_qty", "transferred_in",
                       "transferred_out", "inv_after", "weeks_of_cover_display", "woc_after_display",
                       "projected_sales_horizon"]
        impacted = closing[(closing["transferred_in"] > 0) | (closing["transferred_out"] > 0)][impact_cols]
        st.dataframe(impacted.sort_values("transferred_in", ascending=False), use_container_width=True, height=350)
        st.download_button("⬇️ Download Closing Inventory Position (CSV)", closing.to_csv(index=False).encode(),
                            "closing_inventory_position.csv", "text/csv")

with tab_gaps:
    st.markdown("#### Unfulfilled Gaps — No Donor Stock Available")
    if unfulfilled_df.empty:
        st.success("All identified stock gaps were fully resolved.")
    else:
        st.dataframe(unfulfilled_df.sort_values("unfulfilled_gap_units", ascending=False),
                     use_container_width=True, height=350)
        st.download_button("⬇️ Download Unfulfilled Gaps (CSV)", unfulfilled_df.to_csv(index=False).encode(),
                            "unfulfilled_gaps.csv", "text/csv")