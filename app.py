@st.cache_data(show_spinner=False)
def build_master(stock_parsed, sales_parsed, as_of_date):
    # Temporarily cast the groupby keys to category for the two heaviest
    # aggregations (these run over the full, un-aggregated row counts).
    # observed=True prevents any combinatorial blow-up; both columns are cast
    # straight back to plain strings immediately after, so nothing downstream
    # ever sees category dtype.
    stock_tmp = stock_parsed.copy()
    stock_tmp["store_id"] = stock_tmp["store_id"].astype("category")
    stock_tmp["barcode"] = stock_tmp["barcode"].astype("category")
    stock_agg = stock_tmp.groupby(["store_id", "barcode"], as_index=False, observed=True).agg(
        inv_qty=("inv_qty", "sum"),
        style=("style", "first"), colour=("colour", "first"), size=("size", "first"),
        dept=("dept", "first"), subdept=("subdept", "first"),
        cls=("cls", "first"), subclass=("subclass", "first"), subbrand=("subbrand", "first"),
        country=("country", "first"), store_brand=("store_brand", "first"),
        last_received=("last_received", "max"), max_ageing=("max_ageing", "max"),
        age_days_reported=("age_days_reported", "max"),
    )
    stock_agg["store_id"] = stock_agg["store_id"].astype(str)
    stock_agg["barcode"] = stock_agg["barcode"].astype(str)

    date_min, date_max = sales_parsed["date"].min(), sales_parsed["date"].max()
    if pd.notna(date_min) and pd.notna(date_max) and date_max > date_min:
        period_days = max(1, (date_max - date_min).days + 1)
    else:
        period_days = 30

    sales_tmp = sales_parsed.copy()
    sales_tmp["store_id"] = sales_tmp["store_id"].astype("category")
    sales_tmp["barcode"] = sales_tmp["barcode"].astype("category")
    sales_agg = sales_tmp.groupby(["store_id", "barcode"], as_index=False, observed=True).agg(
        sales_qty=("sales_qty", "sum"),
        style=("style", "first"), colour=("colour", "first"), size=("size", "first"),
        dept=("dept", "first"), subdept=("subdept", "first"),
        cls=("cls", "first"), subclass=("subclass", "first"), subbrand=("subbrand", "first"),
        country=("country", "first"),
    )
    sales_agg["store_id"] = sales_agg["store_id"].astype(str)
    sales_agg["barcode"] = sales_agg["barcode"].astype(str)

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