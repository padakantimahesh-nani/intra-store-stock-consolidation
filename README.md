# Intra-Store Stock Consolidation Engine

Streamlit app that consolidates BBZ→OFP renamed stores, computes sell-velocity,
weeks-of-cover, size-curve completeness, dead-stock (aged + zero local sales),
store clustering, and a full transfer recommendation engine to move stock from
low-selling/excess/dead locations to high-selling/short locations.

## Files required at runtime (upload via the app UI — not stored in the repo)
1. **Sales file** — e.g. `Sale Data- with OFP Fields -Consol.csv`
   Columns: Location, Code, Country, Date, Item Barcode, Item Style Code,
   Ofp Dept Name, Ofp Sub Department, Ofp Class Name, Ofp Sub Class Name,
   Item Color, Ofp Size, Item Sub Brand, Net Sales Qty
2. **Stock file** — e.g. `Category Wise Stock - For Consolidation.csv`
   Columns: Ofp Group Name, Ofp Dept Name, Ofp Sub Department, Ofp Class Name,
   Ofp Sub Class Name, Ofp Size, Store Brand, Country, Location Code, Location,
   Item Division, Item Sub Brand, Item Style Code, Item Barcode,
   Last recieved date store, Available Qty, Max (Ageing Days), plus your
   Age Days column (Age Days / Ageing Days / Aging Days / Days Since Last Sale /
   Stock Age Days / Age In Days — any one of these headers is auto-detected).
3. **Location Master.xlsx** — Column C = old (BBZ) code, Column I = new (OFP) code.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py