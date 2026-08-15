from __future__ import annotations

import time
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from engine import (
    ALIASES, REQUIRED_INVENTORY, REQUIRED_SALES, Rules, build_position, csv_columns,
    dataframe_csv, detect_columns, prepare_inventory, prepare_sales, recommend_transfers,
)

st.set_page_config(page_title="Stock Consolidation Pro", page_icon="🔁", layout="wide")
st.markdown("""
<style>
:root { --red:#b5121b; --ink:#202124; }
.stApp { background:#fafafa; }
[data-testid="stMetric"] { background:white; border:1px solid #e5e7eb; border-left:4px solid var(--red); padding:12px; border-radius:8px; }
.block-container { padding-top:1.4rem; max-width:1800px; }
h1,h2,h3 { color:var(--ink); }
.stButton button { border-radius:7px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_columns(payload: bytes):
    return csv_columns(payload)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_inventory(payload: bytes, mapping: dict):
    return prepare_inventory(payload, mapping)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_sales(payload: bytes, mapping: dict, as_of: date, days: int):
    return prepare_sales(payload, mapping, as_of, days)


def mapping_editor(label: str, columns: list[str], detected: dict, required: tuple[str, ...], prefix: str):
    with st.expander(f"Column mapping — {label}", expanded=False):
        result = dict(detected)
        fields = list(dict.fromkeys([*required, *[x for x in ALIASES if x not in required]]))
        left, right = st.columns(2)
        for i, field in enumerate(fields):
            target = left if i % 2 == 0 else right
            options = ["— Not available —", *columns]
            current = detected.get(field)
            index = options.index(current) if current in options else 0
            chosen = target.selectbox(
                f"{field}{' *' if field in required else ''}", options, index=index, key=f"{prefix}_{field}"
            )
            result[field] = None if chosen.startswith("—") else chosen
        return result


st.title("🔁 Intra-Store Stock Consolidation Pro")
st.caption("Fast barcode-level transfers from slow stores to high-opportunity stores · existing clusters · WOC · size-curve protection")

with st.sidebar:
    st.header("Files")
    inventory_file = st.file_uploader("Inventory.csv", type=["csv"])
    sales_file = st.file_uploader("30days sales.csv", type=["csv"])
    st.header("Rules")
    as_of = st.date_input("Report date", value=date.today())
    window_days = st.number_input("Sales window (days)", 7, 90, 30)
    safety_woc = st.slider("Safety WOC", 0.5, 8.0, 2.0, 0.5)
    target_woc = st.slider("Recipient target WOC", safety_woc, 12.0, max(4.0, safety_woc), 0.5)
    donor_max_woc = st.slider("Donor excess WOC", target_woc, 24.0, max(8.0, target_woc), 0.5)
    zero_sale_keep = st.number_input("Units retained for zero-sale SKU", 0, 20, 0)
    min_transfer = st.number_input("Minimum transfer quantity", 1, 50, 1)
    max_sources = st.number_input("Maximum donor stores per recipient SKU", 1, 10, 3)
    protect_curve = st.checkbox("Protect donor size curves", True)
    curve_min_sizes = st.number_input("Protect curve when donor has at least this many sizes", 2, 12, 3)
    cross_cluster = st.checkbox("Allow cross-cluster fallback", False)

if not inventory_file or not sales_file:
    st.info("Upload Inventory.csv and 30days sales.csv to begin. No Location Master is required.")
    st.stop()

inv_bytes, sales_bytes = inventory_file.getvalue(), sales_file.getvalue()
try:
    inv_columns, sales_columns = cached_columns(inv_bytes), cached_columns(sales_bytes)
except Exception as exc:
    st.error(f"CSV header could not be read: {exc}")
    st.stop()

inv_mapping = mapping_editor("Inventory", inv_columns, detect_columns(inv_columns), REQUIRED_INVENTORY, "inv")
sales_mapping = mapping_editor("Sales", sales_columns, detect_columns(sales_columns), REQUIRED_SALES, "sales")

missing_inv = [x for x in REQUIRED_INVENTORY if not inv_mapping.get(x)]
missing_sales = [x for x in REQUIRED_SALES if not sales_mapping.get(x)]
if missing_inv or missing_sales:
    st.error(f"Map all required fields. Inventory missing: {missing_inv or 'none'}; Sales missing: {missing_sales or 'none'}.")
    st.stop()

run = st.button("🚀 Calculate Consolidation", type="primary", use_container_width=True)
signature = (inventory_file.file_id, sales_file.file_id, tuple(inv_mapping.items()), tuple(sales_mapping.items()),
             as_of, window_days, safety_woc, target_woc, donor_max_woc, zero_sale_keep,
             min_transfer, max_sources, protect_curve, curve_min_sizes, cross_cluster)

if run:
    started = time.perf_counter()
    rules = Rules(int(window_days), safety_woc, target_woc, donor_max_woc, int(zero_sale_keep),
                  int(min_transfer), int(max_sources), protect_curve, int(curve_min_sizes), cross_cluster)
    try:
        with st.status("Running optimized consolidation…", expanded=True) as status:
            t0 = time.perf_counter(); inv, inv_quality = cached_inventory(inv_bytes, inv_mapping); t_inv = time.perf_counter()-t0
            st.write(f"Inventory aggregated: {inv.height:,} store–barcode records")
            t0 = time.perf_counter(); sales, sales_quality = cached_sales(sales_bytes, sales_mapping, as_of, int(window_days)); t_sales = time.perf_counter()-t0
            st.write(f"Sales aggregated: {sales.height:,} store–barcode records")
            t0 = time.perf_counter(); position = build_position(inv, sales, rules); t_metrics = time.perf_counter()-t0
            t0 = time.perf_counter(); transfers, gaps, closing = recommend_transfers(position, rules); t_engine = time.perf_counter()-t0
            runtime = time.perf_counter()-started
            status.update(label=f"Completed in {runtime:.1f} seconds", state="complete", expanded=False)
        st.session_state["consolidation_result"] = {
            "signature": signature, "position": position, "transfers": transfers, "gaps": gaps,
            "closing": closing, "inv_quality": inv_quality, "sales_quality": sales_quality,
            "runtime": runtime, "timings": {"Inventory load + aggregation":t_inv, "Sales load + aggregation":t_sales,
                                               "Metrics":t_metrics, "Transfer engine":t_engine}, "rules": rules,
        }
    except Exception as exc:
        st.exception(exc)
        st.stop()

result = st.session_state.get("consolidation_result")
if not result:
    st.warning("Review the detected column mapping and click Calculate Consolidation.")
    st.stop()
if result["signature"] != signature:
    st.warning("Files or rules have changed. Click Calculate Consolidation to refresh the results.")
    st.stop()

position, transfers, gaps, closing = result["position"], result["transfers"], result["gaps"], result["closing"]
qty = int(transfers["transfer_qty"].sum()) if not transfers.empty else 0
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Stores", position.store.nunique()); k2.metric("Barcodes", position.barcode.nunique())
k3.metric("Recommended Qty", f"{qty:,}"); k4.metric("Transfer Lines", f"{len(transfers):,}")
k5.metric("Unfulfilled Qty", f"{int(gaps.unfulfilled_qty.sum()) if not gaps.empty else 0:,}")
k6.metric("Runtime", f"{result['runtime']:.1f}s")

tabs = st.tabs(["Executive View", "Transfer Plan", "Store Summary", "Gaps", "SKU Position", "Data Quality & Speed"])
with tabs[0]:
    c1,c2 = st.columns(2)
    if not transfers.empty:
        donor = transfers.groupby("from_store", as_index=False).transfer_qty.sum().nlargest(15,"transfer_qty")
        recipient = transfers.groupby("to_store", as_index=False).transfer_qty.sum().nlargest(15,"transfer_qty")
        c1.plotly_chart(px.bar(donor,x="from_store",y="transfer_qty",title="Top Donor Stores"),use_container_width=True)
        c2.plotly_chart(px.bar(recipient,x="to_store",y="transfer_qty",title="Top Recipient Stores",color_discrete_sequence=["#b5121b"]),use_container_width=True)
        cluster = transfers.groupby("cluster",as_index=False).transfer_qty.sum()
        st.plotly_chart(px.bar(cluster,x="cluster",y="transfer_qty",title="Transfers by Recipient Cluster"),use_container_width=True)
    else: st.info("No transfers meet the current rules.")
with tabs[1]:
    if transfers.empty: st.info("No transfer recommendations.")
    else:
        st.dataframe(transfers.sort_values("transfer_qty",ascending=False),use_container_width=True,height=520)
        st.download_button("Download transfer plan",dataframe_csv(transfers),"transfer_plan.csv","text/csv")
with tabs[2]:
    if transfers.empty: st.info("No store movements.")
    else:
        outgoing=transfers.groupby(["from_store","from_store_name"],dropna=False).transfer_qty.sum().rename("transfer_out")
        incoming=transfers.groupby(["to_store","to_store_name"],dropna=False).transfer_qty.sum().rename("transfer_in")
        stores=pd.concat([outgoing.reset_index().rename(columns={"from_store":"store","from_store_name":"store_name"}).set_index(["store","store_name"]),
                          incoming.reset_index().rename(columns={"to_store":"store","to_store_name":"store_name"}).set_index(["store","store_name"])],axis=1).fillna(0).reset_index()
        stores["net_change"]=stores.transfer_in-stores.transfer_out
        st.dataframe(stores,use_container_width=True,height=500)
        st.download_button("Download store summary",dataframe_csv(stores),"store_summary.csv","text/csv")
with tabs[3]:
    if gaps.empty: st.success("All identified needs were fulfilled.")
    else:
        st.dataframe(gaps.sort_values("unfulfilled_qty",ascending=False),use_container_width=True,height=520)
        st.download_button("Download unfulfilled gaps",dataframe_csv(gaps),"unfulfilled_gaps.csv","text/csv")
with tabs[4]:
    show=["store","store_name","cluster","barcode","style","colour","size","inventory_qty","sales_qty","velocity_week","woc","need_units","donor_units"]
    st.dataframe(closing[show+["transfer_in","transfer_out","closing_inventory","closing_woc"]],use_container_width=True,height=560)
    st.download_button("Download closing position",dataframe_csv(closing),"closing_inventory.csv","text/csv")
with tabs[5]:
    st.subheader("Data quality")
    st.json({"inventory":result["inv_quality"],"sales":result["sales_quality"]})
    if not result["sales_quality"]["date_filter_applied"]:
        st.warning("No sales date column was mapped. The app assumes the uploaded sales file already contains exactly the selected window.")
    timing=pd.DataFrame([{"stage":k,"seconds":round(v,3)} for k,v in result["timings"].items()])
    st.dataframe(timing,use_container_width=True)
    st.caption("Invalid keys and negative inventory are excluded and counted; malformed CSV rows cause a visible error and are never silently skipped.")
