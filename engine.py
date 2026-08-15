from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from io import BytesIO
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl


ALIASES = {
    "store": ["store", "store code", "location code", "code", "store_code"],
    "store_name": ["store name", "location", "location name", "store_name"],
    "cluster": ["cluster", "store cluster", "store_cluster", "cluster name"],
    "barcode": ["barcode", "item barcode", "sku", "sku code", "item code"],
    "style": ["style", "item style code", "style code"],
    "colour": ["colour", "color", "item colour", "item color"],
    "size": ["size", "ofp size", "item size"],
    "department": ["department", "ofp dept name", "item department"],
    "subdepartment": ["sub department", "ofp sub department", "item sub department"],
    "class": ["class", "ofp class name", "item class"],
    "subclass": ["subclass", "sub class", "ofp sub class name", "item subclass"],
    "subbrand": ["sub brand", "item sub brand", "subbrand"],
    "inventory_qty": ["inventory qty", "available qty", "stock qty", "soh", "on hand qty"],
    "sales_qty": ["sales qty", "net sales qty", "quantity", "sold qty"],
    "date": ["date", "sales date", "transaction date", "invoice date"],
    "age_days": ["age days", "ageing days", "aging days", "stock age days"],
}

REQUIRED_INVENTORY = ("store", "cluster", "barcode", "inventory_qty")
REQUIRED_SALES = ("store", "cluster", "barcode", "sales_qty")
ATTRS = ("store_name", "style", "colour", "size", "department", "subdepartment", "class", "subclass", "subbrand")


def _norm(value: str) -> str:
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def detect_columns(columns: Iterable[str]) -> dict[str, str | None]:
    normalized = {_norm(c): c for c in columns}
    result: dict[str, str | None] = {}
    for field, candidates in ALIASES.items():
        result[field] = next((normalized[_norm(c)] for c in candidates if _norm(c) in normalized), None)
    return result


def csv_columns(payload: bytes) -> list[str]:
    # Read only the physical header. Polars may otherwise infer an identifier as
    # Int64 from early rows and fail before the mapping screen when a later
    # barcode/store code contains letters.
    first_line = payload.splitlines()[0].decode("utf-8-sig", errors="replace")
    return next(csv.reader([first_line]))


def validate_mapping(mapping: dict[str, str | None], required: Iterable[str], label: str) -> None:
    missing = [f for f in required if not mapping.get(f)]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _read_selected(payload: bytes, mapping: dict[str, str | None]) -> pl.DataFrame:
    selected = list(dict.fromkeys(v for v in mapping.values() if v))
    identifier_fields = {
        "store", "store_name", "cluster", "barcode", "style", "colour", "size",
        "department", "subdepartment", "class", "subclass", "subbrand", "date",
    }
    schema_overrides = {
        source: pl.Utf8
        for field, source in mapping.items()
        if field in identifier_fields and source in selected
    }
    return pl.read_csv(
        BytesIO(payload), columns=selected, infer_schema_length=10_000,
        schema_overrides=schema_overrides,
        ignore_errors=False, truncate_ragged_lines=False, encoding="utf8-lossy",
        null_values=["", "NULL", "null", "None", "nan"], low_memory=False,
    )


def _canonicalize(df: pl.DataFrame, mapping: dict[str, str | None], fields: Iterable[str]) -> pl.DataFrame:
    expressions = []
    for field in fields:
        source = mapping.get(field)
        if source and source in df.columns:
            expressions.append(pl.col(source).alias(field))
        else:
            expressions.append(pl.lit(None).alias(field))
    return df.select(expressions)


def _clean_keys(df: pl.DataFrame) -> pl.DataFrame:
    keys = [c for c in ["store", "cluster", "barcode", "style", "colour", "size"] if c in df.columns]
    return df.with_columns([
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars().replace("", None).alias(c) for c in keys
    ])


def prepare_inventory(payload: bytes, mapping: dict[str, str | None]) -> tuple[pl.DataFrame, dict]:
    validate_mapping(mapping, REQUIRED_INVENTORY, "Inventory.csv")
    raw = _read_selected(payload, mapping)
    fields = (*REQUIRED_INVENTORY, *ATTRS, "age_days")
    df = _clean_keys(_canonicalize(raw, mapping, fields)).with_columns(
        pl.col("inventory_qty").cast(pl.Float64, strict=False).fill_null(0),
        pl.col("age_days").cast(pl.Float64, strict=False),
    )
    quality = {
        "rows": df.height,
        "missing_store": df["store"].null_count(),
        "missing_cluster": df["cluster"].null_count(),
        "missing_barcode": df["barcode"].null_count(),
        "negative_qty": df.filter(pl.col("inventory_qty") < 0).height,
    }
    df = df.filter(
        pl.col("store").is_not_null() & pl.col("cluster").is_not_null() &
        pl.col("barcode").is_not_null() & (pl.col("inventory_qty") >= 0)
    )
    group_keys = ["store", "cluster", "barcode"]
    aggregations = [pl.col("inventory_qty").sum(), pl.col("age_days").max()]
    aggregations += [pl.col(c).drop_nulls().first() for c in ATTRS]
    return df.group_by(group_keys).agg(aggregations), quality


def prepare_sales(
    payload: bytes, mapping: dict[str, str | None], as_of: date, window_days: int
) -> tuple[pl.DataFrame, dict]:
    validate_mapping(mapping, REQUIRED_SALES, "30days sales.csv")
    raw = _read_selected(payload, mapping)
    fields = (*REQUIRED_SALES, *ATTRS, "date")
    df = _clean_keys(_canonicalize(raw, mapping, fields)).with_columns(
        pl.col("sales_qty").cast(pl.Float64, strict=False).fill_null(0)
    )
    has_date = mapping.get("date") is not None
    if has_date:
        parsed_date = pl.coalesce([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, format=fmt, strict=False)
            for fmt in ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%m/%d/%Y"]
        ])
        df = df.with_columns(parsed_date.alias("date"))
        start = as_of - timedelta(days=window_days - 1)
        df = df.filter(pl.col("date").is_between(start, as_of, closed="both"))
    quality = {
        "rows_in_window": df.height,
        "missing_store": df["store"].null_count(),
        "missing_cluster": df["cluster"].null_count(),
        "missing_barcode": df["barcode"].null_count(),
        "negative_sales": df.filter(pl.col("sales_qty") < 0).height,
        "date_filter_applied": has_date,
    }
    df = df.filter(
        pl.col("store").is_not_null() & pl.col("cluster").is_not_null() & pl.col("barcode").is_not_null()
    )
    group_keys = ["store", "cluster", "barcode"]
    aggregations = [pl.col("sales_qty").sum()]
    aggregations += [pl.col(c).drop_nulls().first() for c in ATTRS]
    return df.group_by(group_keys).agg(aggregations), quality


@dataclass(frozen=True)
class Rules:
    window_days: int = 30
    safety_woc: float = 2.0
    target_woc: float = 4.0
    donor_max_woc: float = 8.0
    zero_sale_keep_units: int = 0
    minimum_transfer: int = 1
    max_sources_per_recipient: int = 3
    protect_donor_size_curve: bool = True
    donor_curve_min_sizes: int = 3
    allow_cross_cluster: bool = False


def build_position(inventory: pl.DataFrame, sales: pl.DataFrame, rules: Rules) -> pd.DataFrame:
    master = inventory.join(sales, on=["store", "cluster", "barcode"], how="full", suffix="_sales", coalesce=True)
    for field in ATTRS:
        other = f"{field}_sales"
        if other in master.columns:
            master = master.with_columns(pl.coalesce([field, other]).fill_null("UNKNOWN").alias(field)).drop(other)
        elif field in master.columns:
            master = master.with_columns(pl.col(field).fill_null("UNKNOWN"))
    master = master.with_columns(
        pl.col("inventory_qty").fill_null(0), pl.col("sales_qty").fill_null(0),
        (pl.col("sales_qty").fill_null(0) * (7.0 / rules.window_days)).alias("velocity_week"),
    ).with_columns(
        pl.when(pl.col("velocity_week") > 0).then(pl.col("inventory_qty") / pl.col("velocity_week")).otherwise(None).alias("woc"),
        (pl.col("velocity_week") * rules.safety_woc).ceil().alias("safety_units"),
        (pl.col("velocity_week") * rules.target_woc).ceil().alias("target_units"),
        (pl.col("velocity_week") * rules.donor_max_woc).ceil().alias("donor_keep_units"),
    ).with_columns(
        (pl.col("target_units") - pl.col("inventory_qty")).clip(lower_bound=0).alias("need_units"),
        pl.when(pl.col("velocity_week") > 0)
          .then((pl.col("inventory_qty") - pl.col("donor_keep_units")).clip(lower_bound=0))
          .otherwise((pl.col("inventory_qty") - rules.zero_sale_keep_units).clip(lower_bound=0))
          .alias("raw_donor_units"),
    )
    pdf = master.to_pandas(use_pyarrow_extension_array=False)
    for c in ["inventory_qty", "sales_qty", "need_units", "raw_donor_units"]:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce").fillna(0).round().astype("int64")
    return pdf


def _add_curve_signals(position: pd.DataFrame, rules: Rules) -> pd.DataFrame:
    df = position.copy()
    active = df[df["inventory_qty"] > 0]
    counts = active.groupby(["store", "style", "colour"], dropna=False)["size"].nunique().rename("sizes_present")
    df = df.merge(counts, left_on=["store", "style", "colour"], right_index=True, how="left")
    df["sizes_present"] = df["sizes_present"].fillna(0).astype(int)
    df["recipient_missing_size"] = df["inventory_qty"].le(0) & df["sizes_present"].gt(0)
    reserve = np.where(
        rules.protect_donor_size_curve & df["sizes_present"].ge(rules.donor_curve_min_sizes) & df["sales_qty"].gt(0),
        1, 0,
    )
    df["donor_units"] = np.minimum(df["raw_donor_units"], (df["inventory_qty"] - reserve).clip(lower=0)).astype(int)
    return df


def recommend_transfers(position: pd.DataFrame, rules: Rules) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = _add_curve_signals(position, rules)
    recipients = df[df["need_units"] >= rules.minimum_transfer].copy()
    recipients = recipients.sort_values(
        ["recipient_missing_size", "velocity_week", "need_units"], ascending=[False, False, False]
    )
    donors = df[df["donor_units"] >= rules.minimum_transfer].copy()
    donors = donors.sort_values(["velocity_week", "age_days", "donor_units"], ascending=[True, False, False])

    pools: dict[tuple[str, str], list[dict]] = {}
    global_pools: dict[str, list[dict]] = {}
    remaining: dict[tuple[str, str], int] = {}
    for row in donors.to_dict("records"):
        pools.setdefault((str(row["cluster"]), str(row["barcode"])), []).append(row)
        global_pools.setdefault(str(row["barcode"]), []).append(row)
        remaining[(str(row["store"]), str(row["barcode"]))] = int(row["donor_units"])

    transfers: list[dict] = []
    gaps: list[dict] = []
    for rec in recipients.to_dict("records"):
        gap = int(rec["need_units"])
        sources = 0
        primary = pools.get((str(rec["cluster"]), str(rec["barcode"])), [])
        candidates = list(primary)
        if rules.allow_cross_cluster:
            candidates += [x for x in global_pools.get(str(rec["barcode"]), []) if x["cluster"] != rec["cluster"]]
        for donor in candidates:
            if gap <= 0 or sources >= rules.max_sources_per_recipient:
                break
            if donor["store"] == rec["store"]:
                continue
            key = (str(donor["store"]), str(rec["barcode"]))
            available = remaining.get(key, 0)
            if available < rules.minimum_transfer:
                continue
            qty = min(available, gap)
            if qty < rules.minimum_transfer:
                continue
            transfers.append({
                "from_store": donor["store"], "from_store_name": donor.get("store_name", ""),
                "to_store": rec["store"], "to_store_name": rec.get("store_name", ""),
                "cluster": rec["cluster"], "donor_cluster": donor["cluster"],
                "barcode": rec["barcode"], "style": rec.get("style"), "colour": rec.get("colour"),
                "size": rec.get("size"), "department": rec.get("department"),
                "transfer_qty": qty, "recipient_velocity_week": round(float(rec["velocity_week"]), 2),
                "donor_velocity_week": round(float(donor["velocity_week"]), 2),
                "recipient_woc_before": round(float(rec["woc"]), 2) if pd.notna(rec["woc"]) else 0,
                "same_cluster": donor["cluster"] == rec["cluster"],
                "size_curve_priority": bool(rec["recipient_missing_size"]),
                "reason": "Completes/improves size curve" if rec["recipient_missing_size"] else "Fast seller below target WOC",
            })
            remaining[key] -= qty
            gap -= qty
            sources += 1
        if gap > 0:
            gaps.append({
                "store": rec["store"], "store_name": rec.get("store_name", ""), "cluster": rec["cluster"],
                "barcode": rec["barcode"], "style": rec.get("style"), "colour": rec.get("colour"),
                "size": rec.get("size"), "unfulfilled_qty": gap,
                "reason": "No eligible donor or source-store limit reached",
            })

    transfer_df = pd.DataFrame(transfers)
    gaps_df = pd.DataFrame(gaps)
    closing = df.copy()
    if not transfer_df.empty:
        sent = transfer_df.groupby(["from_store", "barcode"])["transfer_qty"].sum()
        received = transfer_df.groupby(["to_store", "barcode"])["transfer_qty"].sum()
        closing["transfer_out"] = [sent.get((s, b), 0) for s, b in zip(closing.store, closing.barcode)]
        closing["transfer_in"] = [received.get((s, b), 0) for s, b in zip(closing.store, closing.barcode)]
    else:
        closing["transfer_out"] = 0
        closing["transfer_in"] = 0
    closing["closing_inventory"] = closing["inventory_qty"] - closing["transfer_out"] + closing["transfer_in"]
    closing["closing_woc"] = np.where(closing["velocity_week"] > 0, closing["closing_inventory"] / closing["velocity_week"], np.nan)
    return transfer_df, gaps_df, closing


def dataframe_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")
