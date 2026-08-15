from datetime import date

import polars as pl

from engine import Rules, build_position, recommend_transfers


def test_low_to_high_same_cluster_and_curve_protection():
    inv = pl.DataFrame({
        "store":["LOW","HIGH"], "cluster":["A","A"], "barcode":["100","100"],
        "inventory_qty":[20.0,0.0], "age_days":[80.0,None], "store_name":["Low","High"],
        "style":["S1","S1"], "colour":["BLACK","BLACK"], "size":["42","42"],
        "department":["FW","FW"], "subdepartment":["MEN","MEN"], "class":["SHOE","SHOE"],
        "subclass":["SPORT","SPORT"], "subbrand":["X","X"],
    })
    sales = pl.DataFrame({
        "store":["LOW","HIGH"], "cluster":["A","A"], "barcode":["100","100"],
        "sales_qty":[0.0,30.0], "store_name":["Low","High"], "style":["S1","S1"],
        "colour":["BLACK","BLACK"], "size":["42","42"], "department":["FW","FW"],
        "subdepartment":["MEN","MEN"], "class":["SHOE","SHOE"], "subclass":["SPORT","SPORT"],
        "subbrand":["X","X"],
    })
    rules = Rules(window_days=30, safety_woc=2, target_woc=4, donor_max_woc=8)
    position = build_position(inv, sales, rules)
    transfers, gaps, closing = recommend_transfers(position, rules)
    assert transfers.transfer_qty.sum() == 20
    assert transfers.iloc[0].from_store == "LOW"
    assert transfers.iloc[0].to_store == "HIGH"
    assert closing.closing_inventory.sum() == 20


if __name__ == "__main__":
    test_low_to_high_same_cluster_and_curve_protection()
    print("PASS")
