
import streamlit as st
import pandas as pd
from pandas.errors import EmptyDataError
from utils import haversine
from visualization import plot_network, summary
from optimization import optimize, ROAD_FACTOR

def _build_lane_df(res, scn):
    """Build lane table with brand + center_idx for coloring and brand filters.

    Key points:
    - Always reconstruct brand-aware TRANSFER lanes (Tier‑1 → WH) using
      res['t1_for_wh_brand'] when present; otherwise fall back to nearest Tier‑1.
    - Use provided inbound_flows only if they have meaningful brand labels;
      otherwise reconstruct brand-aware inbound from assignments and inbound_pts.
    - Never early‑return just because inbound/transfer arrays exist — ensures
      brand filtering never hides flows.
    """
    import pandas as _pd
    from utils import haversine as _hz
    lanes = []

    # ---------- Outbound (Warehouse → Customer) ----------
    if "assigned" in res and len(res["assigned"]) > 0:
        for r in res["assigned"].itertuples():
            w_idx = int(r.Warehouse)
            wlon, wlat = res["centers"][w_idx]
            lanes.append(dict(
                lane_type="outbound",
                brand=str(getattr(r, "Brand", "")),
                country=str(getattr(r, "Country", "")),
                origin_lon=float(wlon), origin_lat=float(wlat),
                dest_lon=float(r.Longitude), dest_lat=float(r.Latitude),
                distance_mi=float(r.DistMi),
                weight_lbs=float(r.DemandLbs),
                rate=float(scn.get("rate_out", 0.0)),
                cost=float(r.DemandLbs * r.DistMi * scn.get("rate_out", 0.0)),
                center_idx=w_idx,
            ))

    # ---------- Transfers (Tier‑1 → WH) ----------
    assigned = res.get("assigned")
    centers = res.get("centers") or []
    rdc_list = res.get("rdc_list") or []
    t1_coords = res.get("tier1_coords") or [t["coords"] for t in rdc_list] if rdc_list else []
    use_map = {tuple(k.split("|")): v for k, v in (res.get("t1_for_wh_brand") or {}).items()}
    # Convert wh,brand keys to ints/strs
    t1_map = {}
    for (j_str, b), v in use_map.items():
        try:
            t1_map[(int(j_str), str(b))] = int(v)
        except Exception:
            continue

    if assigned is not None and len(assigned) and t1_coords:
        if "Brand" in assigned.columns:
            g = assigned.groupby(["Warehouse", "Brand"], dropna=False)["DemandLbs"].sum().reset_index()
            wh_brand_rows = [(int(w), str(b), float(v)) for w, b, v in g.itertuples(index=False, name=None) if float(v) > 0]
        else:
            g = assigned.groupby(["Warehouse"])["DemandLbs"].sum().reset_index()
            wh_brand_rows = [(int(w), "", float(v)) for w, v in g.itertuples(index=False, name=None) if float(v) > 0]

        # Precompute nearest T1 when fallback is needed
        def _nearest_t1_for_wh(j):
            wx, wy = centers[j]
            best_t, best_d, best_tx, best_ty = None, None, None, None
            for t, (tx, ty) in enumerate(t1_coords):
                d = _hz(tx, ty, wx, wy) * ROAD_FACTOR
                if (best_d is None) or (d < best_d):
                    best_t, best_d, best_tx, best_ty = t, d, tx, ty
            return best_t, best_d, best_tx, best_ty

        for (j, b, lbs) in wh_brand_rows:
            wx, wy = centers[j]
            if (j, b) in t1_map:
                t = int(t1_map[(j, b)])
                tx, ty = t1_coords[t]
                dist = _hz(tx, ty, wx, wy) * ROAD_FACTOR
            else:
                t, dist, tx, ty = _nearest_t1_for_wh(j)
            if dist and dist > 1e-6 and lbs > 0:
                lanes.append(dict(
                    lane_type="transfer",
                    brand=b,
                    origin_lon=float(tx), origin_lat=float(ty),
                    dest_lon=float(wx), dest_lat=float(wy),
                    distance_mi=float(dist),
                    weight_lbs=float(lbs),
                    rate=float(scn.get("trans_rate", 0.0)),
                    cost=float(lbs * dist * scn.get("trans_rate", 0.0)),
                    center_idx=j,
                ))

    # ---------- Inbound (Supply → Tier‑1/WH) ----------
    provided_inbound = []
    for rec in (res.get("inbound_flows") or []):
        rr = dict(rec)
        rr.setdefault("brand", "")
        rr.setdefault("center_idx", None)
        provided_inbound.append(rr)

    have_brandful_inbound = any(str(x.get("brand", "")).strip() not in ("", "ALL", "None") for x in provided_inbound)

    if have_brandful_inbound:
        lanes.extend(provided_inbound)
    else:
        inbound_on = bool(scn.get("inbound_on"))
        inbound_pts = scn.get("inbound_pts") or []
        if assigned is not None and len(assigned) and inbound_on and inbound_pts:
            if t1_coords:
                # Aggregate (T1, brand) lbs from wh_brand via chosen T1 (same as transfers)
                if "Brand" in assigned.columns:
                    g = assigned.groupby(["Warehouse", "Brand"], dropna=False)["DemandLbs"].sum().reset_index()
                    wh_brand_rows = [(int(w), str(b), float(v)) for w, b, v in g.itertuples(index=False, name=None) if float(v) > 0]
                else:
                    g = assigned.groupby(["Warehouse"])["DemandLbs"].sum().reset_index()
                    wh_brand_rows = [(int(w), "", float(v)) for w, v in g.itertuples(index=False, name=None) if float(v) > 0]
                t1_brand_lbs = {}
                for (j, b, lbs) in wh_brand_rows:
                    if (j, b) in t1_map:
                        t = int(t1_map[(j, b)])
                        tx, ty = t1_coords[t]
                        d = _hz(tx, ty, centers[j][0], centers[j][1]) * ROAD_FACTOR
                        # d unused here, but mapping trustworthy
                    else:
                        # fallback to nearest T1
                        best_t, _, _, _ = None, None, None, None
                        best_d = None
                        for t, (tx, ty) in enumerate(t1_coords):
                            d = _hz(tx, ty, centers[j][0], centers[j][1]) * ROAD_FACTOR
                            if (best_d is None) or (d < best_d):
                                best_t, best_d = t, d
                        t = int(best_t)
                    t1_brand_lbs[(t, b)] = t1_brand_lbs.get((t, b), 0.0) + float(lbs)
                for slon, slat, pct in inbound_pts:
                    for (t, b), lbs in t1_brand_lbs.items():
                        tx, ty = t1_coords[int(t)]
                        dist = _hz(slon, slat, tx, ty) * ROAD_FACTOR
                        wt = float(lbs) * float(pct)
                        if wt > 0:
                            lanes.append(dict(
                                lane_type="inbound",
                                brand=b,
                                origin_lon=float(slon), origin_lat=float(slat),
                                dest_lon=float(tx), dest_lat=float(ty),
                                distance_mi=float(dist),
                                weight_lbs=wt,
                                rate=float(scn.get("in_rate", 0.0)),
                                cost=float(wt * dist * scn.get("in_rate", 0.0)),
                                center_idx=None,  # inbound into Tier‑1
                            ))
            else:
                # No Tier‑1: inbound goes directly → WH per (WH, brand)
                if "Brand" in assigned.columns:
                    g = assigned.groupby(["Warehouse", "Brand"], dropna=False)["DemandLbs"].sum().reset_index()
                    wh_brand_rows = [(int(w), str(b), float(v)) for w, b, v in g.itertuples(index=False, name=None) if float(v) > 0]
                else:
                    g = assigned.groupby(["Warehouse"])["DemandLbs"].sum().reset_index()
                    wh_brand_rows = [(int(w), "", float(v)) for w, v in g.itertuples(index=False, name=None) if float(v) > 0]
                for slon, slat, pct in inbound_pts:
                    for (j, b, lbs) in wh_brand_rows:
                        if lbs <= 0:
                            continue
                        wx, wy = centers[j]
                        dist = _hz(slon, slat, wx, wy) * ROAD_FACTOR
                        wt = float(lbs) * float(pct)
                        if wt > 0:
                            lanes.append(dict(
                                lane_type="inbound",
                                brand=b,
                                origin_lon=float(slon), origin_lat=float(slat),
                                dest_lon=float(wx), dest_lat=float(wy),
                                distance_mi=float(dist),
                                weight_lbs=wt,
                                rate=float(scn.get("in_rate", 0.0)),
                                cost=float(wt * dist * scn.get("in_rate", 0.0)),
                                center_idx=j,
                            ))

    # Include any transfer_flows provided by the model (non-duplicative)
    for rec in (res.get("transfer_flows") or []):
        rr = dict(rec)
        rr.setdefault("brand", "")
        rr.setdefault("center_idx", None)
        lanes.append(rr)

    return _pd.DataFrame(lanes)

    # --- Otherwise: reconstruct BRAND-AWARE transfers/inbound from assigned
    if "assigned" in res and len(res["assigned"]) > 0:
        assigned = res["assigned"]
        centers = res["centers"]
        rdc_list = res.get("rdc_list") or []
        inbound_on = bool(scn.get("inbound_on"))
        inbound_pts = scn.get("inbound_pts") or []

        # Demand per (warehouse, brand)
        if "Brand" in assigned.columns:
            grp = (assigned.groupby(["Warehouse","Brand"], dropna=False)["DemandLbs"].sum().reset_index())
            dem_wh_brand = [(int(w), str(b), float(lbs)) for w,b,lbs in grp.itertuples(index=False, name=None) if float(lbs) > 0]
        else:
            grp = (assigned.groupby(["Warehouse"])["DemandLbs"].sum().reset_index())
            dem_wh_brand = [(int(w), "", float(lbs)) for w,lbs in grp.itertuples(index=False, name=None) if float(lbs) > 0]

        # Transfers: Tier-1 -> WH per (wh,brand)
        if rdc_list:
            t1_coords = [t["coords"] for t in rdc_list]
            for (j,b,lbs) in dem_wh_brand:
                wx, wy = centers[j]
                # choose nearest T1
                best_t, best_dist, best_tx, best_ty = None, None, None, None
                for t,(tx,ty) in enumerate(t1_coords):
                    dist = _hz(tx, ty, wx, wy) * ROAD_FACTOR
                    if (best_dist is None) or (dist < best_dist):
                        best_t, best_dist, best_tx, best_ty = t, dist, tx, ty
                if best_dist and best_dist > 1e-6 and lbs > 0:
                    lanes.append(dict(
                        lane_type="transfer",
                        brand=b,
                        origin_lon=float(best_tx), origin_lat=float(best_ty),
                        dest_lon=float(wx), dest_lat=float(wy),
                        distance_mi=float(best_dist),
                        weight_lbs=float(lbs),
                        rate=float(scn.get("trans_rate", 0.0)),
                        cost=float(lbs * best_dist * scn.get("trans_rate", 0.0)),
                        center_idx=j,
                    ))

            # Inbound: supply -> Tier-1 per (T1, brand)
            if inbound_on and inbound_pts:
                # Aggregate lbs handled by each (T1, brand)
                t1_brand_lbs = {}
                for (j,b,lbs) in dem_wh_brand:
                    wx, wy = centers[j]
                    # nearest T1
                    best_t, best_dist = None, None
                    for t,(tx,ty) in enumerate(t1_coords):
                        dist = _hz(tx, ty, wx, wy) * ROAD_FACTOR
                        if (best_dist is None) or (dist < best_dist):
                            best_t, best_dist = t, dist
                    if best_t is not None and lbs > 0:
                        t1_brand_lbs[(best_t,b)] = t1_brand_lbs.get((best_t,b), 0.0) + lbs

                for slon, slat, pct in inbound_pts:
                    for (t,b), lbs in t1_brand_lbs.items():
                        tx, ty = t1_coords[t]
                        dist = _hz(slon, slat, tx, ty) * ROAD_FACTOR
                        wt = float(lbs) * float(pct)
                        if wt > 0:
                            lanes.append(dict(
                                lane_type="inbound",
                                brand=b,
                                origin_lon=float(slon), origin_lat=float(slat),
                                dest_lon=float(tx), dest_lat=float(ty),
                                distance_mi=float(dist),
                                weight_lbs=wt,
                                rate=float(scn.get("in_rate", 0.0)),
                                cost=float(wt * dist * scn.get("in_rate", 0.0)),
                                center_idx=None,  # inbound to Tier-1
                            ))
        else:
            # No Tier-1: inbound supply -> WH per (wh,brand)
            if inbound_on and inbound_pts:
                for slon, slat, pct in inbound_pts:
                    for (j,b,lbs) in dem_wh_brand:
                        if lbs <= 0: continue
                        wx, wy = centers[j]
                        dist = _hz(slon, slat, wx, wy) * ROAD_FACTOR
                        wt = float(lbs) * float(pct)
                        if wt > 0:
                            lanes.append(dict(
                                lane_type="inbound",
                                brand=b,
                                origin_lon=float(slon), origin_lat=float(slat),
                                dest_lon=float(wx), dest_lat=float(wy),
                                distance_mi=float(dist),
                                weight_lbs=wt,
                                rate=float(scn.get("in_rate", 0.0)),
                                cost=float(wt * dist * scn.get("in_rate", 0.0)),
                                center_idx=j,
                            ))

    return _pd.DataFrame(lanes)

st.set_page_config(page_title="Warehouse Network Optimizer", layout="wide")

# ---------------- Safe readers ----------------
def _read_csv_file(uploaded_file, **kw):
    if uploaded_file is None:
        raise ValueError("No file provided.")
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, **kw)
        if df.shape[1] == 0 or (df.empty and not df.columns.size):
            raise EmptyDataError("No columns to parse from file")
        return df
    except EmptyDataError:
        st.error("The selected CSV appears to be empty or unreadable. Please verify the file has a header row and data.")
        raise
    except UnicodeDecodeError:
        st.error("Encoding issue while reading the CSV. Ensure it's a UTF-8 text CSV (not Excel).")
        raise
    except Exception as e:
        st.error(f"Couldn't read CSV: {e}")
        raise

def _read_optional_csv(uploaded_file, **kw):
    if uploaded_file is None:
        return None
    try:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, **kw)
    except EmptyDataError:
        return None

# ---------------- Session ----------------
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {}
if "cache" not in st.session_state:
    st.session_state["cache"] = {}

def _num_input(scn, key, label, default, fmt="%.4f", **kw):
    scn.setdefault(key,default)
    scn[key]=st.number_input(label,value=scn[key],format=fmt,
                             key=f"{key}_{scn['_name']}",**kw)

# ---------------- Sidebar ----------------
def sidebar(scn):
    name=scn["_name"]
    with st.sidebar:
        st.header(f"Inputs — {name}")
        with st.expander("🗂️ Demand & Candidate Files",True):
            up=st.file_uploader(
                "Demand CSV (Longitude, Latitude, DemandLbs, Brand, Country, CurrWH_Lon, CurrWH_Lat) — one row per customer–product group",
                type=["csv"],
                key=f"dem_{name}"
            )
            if up:
                scn["demand_file"]=up
                try:
                    df_preview=_read_csv_file(up)
                    brands = sorted([str(x) for x in df_preview.get("Brand", pd.Series([])).dropna().unique().tolist()])
                    scn["brands"] = brands
                    st.session_state["cache"][f"{name}_demand_df"]=df_preview
                except Exception:
                    pass
            if "demand_file" not in scn:
                st.info("Upload demand file to continue.")
                return False
            if st.checkbox("Preview demand",key=f"pre_{name}"):
                dfp = st.session_state["cache"].get(f"{name}_demand_df")
                if dfp is None:
                    try:
                        dfp=_read_csv_file(scn["demand_file"])
                        st.session_state["cache"][f"{name}_demand_df"]=dfp
                    except Exception:
                        dfp=None
                if dfp is not None:
                    st.dataframe(dfp.head())

            cand_up=st.file_uploader("Candidate WH CSV (lon,lat[,cost/sqft])", type=["csv"],
                                     key=f"cand_{name}")
            if cand_up is not None:
                if cand_up:
                    scn["cand_file"]=cand_up
                else:
                    scn.pop("cand_file",None)
            scn["restrict_cand"]=st.checkbox("Restrict to candidates",
                                             value=scn.get("restrict_cand",False),
                                             key=f"rc_{name}")

        with st.expander("🇨🇦 Canada Routing Controls", False):
            scn["can_en"] = st.checkbox("Enable Canada routing (Toronto-facing rule)", value=scn.get("can_en", False), key=f"can_en_{name}")
            cols = st.columns(2)
            scn["can_lon"] = cols[0].number_input("Default longitude threshold (east faces Canada)", value=float(scn.get("can_lon", -105.0)), format="%.3f", key=f"can_lon_{name}")
            scn["can_wh_lon"] = cols[1].number_input("Canada WH Lon", value=float(scn.get("can_wh_lon", -79.3832)), format="%.6f", key=f"can_wh_lon_{name}")
            scn["can_wh_lat"] = st.number_input("Canada WH Lat", value=float(scn.get("can_wh_lat", 43.6532)), format="%.6f", key=f"can_wh_lat_{name}")
            st.caption("If enabled: CAN + (Longitude ≥ threshold) → Canada WH; other CAN and all USA rows → nearest US center.")
            # Per-brand thresholds
            brands = scn.get("brands", [])
            brand_thresh = {}
            if brands:
                st.markdown("**Per-brand longitude thresholds**")
                for b in brands:
                    key = f"can_lon_brand_{b}_{name}"
                    default_val = float(scn.get(key, scn.get("can_lon", -105.0)))
                    val = st.number_input(f"{b} threshold (east faces Canada)", value=default_val, format="%.3f", key=key)
                    scn[key] = val
                    brand_thresh[b] = float(val)
            scn["brand_can_thresholds"] = brand_thresh
            # Override toggle
            scn["brand_overrides_canada"] = st.checkbox(
                "Brand fulfillment constraints override Canada routing (when a brand has allowed sites)",
                value=scn.get("brand_overrides_canada", False),
                key=f"brand_over_can_{name}"
            )

        with st.expander("🏗️ Current-State Calibration", False):
            scn["use_current_state"] = st.checkbox("Calibrate to current state (force flows using CurrWH_Lon/CurrWH_Lat)", value=scn.get("use_current_state", False), key=f"cs_{name}")
            if scn["use_current_state"]:
                st.caption("Centers derived from current-state columns; brand & Canada rules are ignored.")

        with st.expander("🏷️ Brand Fulfillment Constraints (optional)", False):
            st.caption("Enter allowed warehouse coordinates (lon,lat per line) for each brand. Leave blank for no restriction.")
            brands = scn.get("brands", [])
            brand_allowed = {}
            if brands:
                for b in brands:
                    key = f"brand_allowed_{b}_{name}"
                    txt_default = scn.get(key, "")
                    txt = st.text_area(f"{b} — allowed warehouses (lon,lat per line)", value=txt_default, key=key, height=80)
                    scn[key] = txt
                    pairs = []
                    for ln in txt.splitlines():
                        try:
                            lon, lat = map(float, ln.split(","))
                            pairs.append([lon, lat])
                        except Exception:
                            continue
                    brand_allowed[b] = pairs
            scn["brand_allowed_sites"] = brand_allowed

        with st.expander("💰 Cost Parameters",False):
            st.subheader("Transportation $ / lb-mile")
            _num_input(scn,"rate_out","Outbound",0.35)
            _num_input(scn,"in_rate","Inbound",0.30)
            _num_input(scn,"trans_rate","Transfer",0.32)
            st.subheader("Warehouse")
            _num_input(scn,"sqft_per_lb","Sq ft per lb",0.02)
            _num_input(scn,"cost_sqft","$/sq ft / yr",6.0,"%.2f")
            _num_input(scn,"fixed_wh_cost","Fixed $",250000.0,"%.0f",step=50000.0)

        with st.expander("🔢 Warehouse Count",False):
            scn["auto_k"]=st.checkbox("Optimize k",value=scn.get("auto_k",True),
                                      key=f"ak_{name}")
            if scn["auto_k"]:
                scn["k_rng"]=st.slider("k range",1,30,scn.get("k_rng",(3,6)),
                                       key=f"kr_{name}")
            else:
                _num_input(scn,"k_fixed","k",4,"%.0f",min_value=1,max_value=30)

        with st.expander("📍 Locations",False):
            st.subheader("Fixed Warehouses (US)")
            fixed_txt=st.text_area("lon,lat per line",value=scn.get("fixed_txt",""),
                                   key=f"fx_{name}",height=80)
            scn["fixed_txt"]=fixed_txt
            fixed_centers=[]
            for ln in fixed_txt.splitlines():
                try:
                    lon,lat=map(float,ln.split(","))
                    fixed_centers.append([lon,lat])
                except Exception:
                    continue
            scn["fixed_centers"]=fixed_centers

            st.subheader("Inbound supply points")
            scn["inbound_on"]=st.checkbox("Enable inbound",
                                          value=scn.get("inbound_on",False),
                                          key=f"inb_{name}")
            inbound_pts=[]
            if scn["inbound_on"]:
                sup_txt=st.text_area("lon,lat,percent (0-100)",
                                     value=scn.get("sup_txt",""),
                                     key=f"sup_{name}",height=80)
                scn["sup_txt"]=sup_txt
                for ln in sup_txt.splitlines():
                    try:
                        lon,lat,pct=map(float,ln.split(","))
                        inbound_pts.append([lon,lat,pct/100.0])
                    except Exception:
                        continue
            scn["inbound_pts"]=inbound_pts

            # Guided inbound controls
            with st.expander("🚚 Inbound Controls (guided)", False):
                st.caption("Add inbound supply points, pick allowed Product Groups, and choose routing (Split vs Force to a Tier‑1 site).")
                scn["inbound_guided_en"] = st.checkbox("Enable guided inbound builder", value=scn.get("inbound_guided_en", False), key=f"inb_guided_{name}")
                rules = []
                if scn["inbound_guided_en"]:
                    n = st.number_input("Number of inbound points", min_value=1, max_value=20, value=int(scn.get("inb_rules_n", 1)), step=1, key=f"inb_rules_n_{name}")
                    scn["inb_rules_n"] = int(n)
                    pgroups = scn.get("brands", [])
                    t1_labels = []
                    for i in range(1,4):
                        if scn.get(f"rdc{i}_en"):
                            t1_labels.append(f"{i}: {scn.get(f'rdc{i}_typ','RDC')} @ ({scn.get(f'rdc{i}_lon',0.0):.4f},{scn.get(f'rdc{i}_lat',0.0):.4f})")
                    for i in range(int(n)):
                        st.markdown(f"**Inbound point {i+1}**")
                        c = st.columns(3)
                        lon = c[0].number_input("Longitude", key=f"inb_rule_lon_{name}_{i}", value=float(scn.get(f"inb_rule_lon_{i}", -118.25)), format="%.6f")
                        lat = c[1].number_input("Latitude", key=f"inb_rule_lat_{name}_{i}", value=float(scn.get(f"inb_rule_lat_{i}", 34.0522)), format="%.6f")
                        pct = c[2].number_input("Percent of network inbound (0–100)", key=f"inb_rule_pct_{name}_{i}", min_value=0.0, max_value=100.0, value=float(scn.get(f"inb_rule_pct_{i}", 100.0)), step=1.0)
                        scn[f"inb_rule_lon_{i}"]=lon; scn[f"inb_rule_lat_{i}"]=lat; scn[f"inb_rule_pct_{i}"]=pct
                        allowed = st.multiselect("Allowed Product Groups (blank = all)", options=pgroups, default=scn.get(f"inb_rule_allowed_{i}", []), key=f"inb_rule_allowed_{name}_{i}")
                        scn[f"inb_rule_allowed_{i}"] = allowed
                        if t1_labels:
                            mode = st.radio("Routing", ["Split to nearest Tier‑1/WH", "Force to a specific Tier‑1"], index=0 if scn.get(f"inb_rule_mode_{i}", "split")=="split" else 1, key=f"inb_rule_mode_{name}_{i}")
                            scn[f"inb_rule_mode_{i}"] = "split" if mode.startswith("Split") else "force"
                            if scn[f"inb_rule_mode_{i}"]=="force":
                                sel = st.selectbox("Send inbound into which Tier‑1?", options=list(range(len(t1_labels))), format_func=lambda j: t1_labels[j], key=f"inb_rule_force_idx_{name}_{i}")
                                scn[f"inb_rule_force_idx_{i}"] = int(sel)
                            else:
                                scn[f"inb_rule_force_idx_{i}"] = None
                        else:
                            st.info("No Tier‑1 (RDC/SDC) configured; only Split mode is available.")
                            scn[f"inb_rule_mode_{i}"] = "split"; scn[f"inb_rule_force_idx_{i}"] = None
                        rules.append(dict(
                            lon=float(lon), lat=float(lat), pct=float(pct)/100.0,
                            allowed_brands=[str(x) for x in allowed],
                            mode=scn[f"inb_rule_mode_{i}"],
                            force_t1_index=scn.get(f"inb_rule_force_idx_{i}")
                        ))
                scn["inbound_rules"] = rules
    

        
        with st.expander("🏭 Warehouse Product Group Restrictions (optional)", False):
            st.caption("Pick warehouses and choose the brands they are allowed to serve. All other brands will be disallowed at that site.")
            # Build the list of known warehouse coordinates (fixed centers + candidate sites if provided)
            options = []
            # from fixed centers
            for lon, lat in scn.get("fixed_centers", []):
                options.append(f"{round(float(lon),6)},{round(float(lat),6)}")
            # from candidate file if present
            cand_file = scn.get("cand_file")
            if cand_file is not None:
                try:
                    cf = _read_optional_csv(cand_file, header=None)
                    if cf is not None and not cf.empty:
                        cf = cf.dropna(subset=[0,1])
                        for r in cf.itertuples(index=False):
                            options.append(f"{round(float(r[0]),6)},{round(float(r[1]),6)}")
                except Exception:
                    pass
            # de-duplicate while preserving order
            seen = set(); site_options = []
            for s in options:
                if s not in seen:
                    seen.add(s); site_options.append(s)

            brands = scn.get("brands", [])
            # Load existing mapping, if any
            existing_map = scn.get("wh_brand_allowed", {})
            preselect = [k for k in site_options if k in existing_map]
            chosen_sites = st.multiselect("Warehouses to restrict (lon,lat)", site_options, default=preselect, key=f"wh_sites_{name}")
            new_map = {}
            if chosen_sites:
                for s in chosen_sites:
                    default_brands = existing_map.get(s, [])
                    sel = st.multiselect(f"{s} → allowed product groups", options=brands, default=default_brands, key=f"wh_{s}_{name}")
                    new_map[s] = sel
            scn["wh_brand_allowed"] = new_map
        with st.expander("🏬 RDC / SDC (up to 3)",False):
            for idx in range(1,4):
                cols=st.columns([1,4])
                en=cols[0].checkbox(f"{idx}",key=f"rdc_en_{name}_{idx}",
                                     value=scn.get(f"rdc{idx}_en",False))
                scn[f"rdc{idx}_en"]=en
                if en:
                    with cols[1]:
                        lon=st.number_input("Longitude",format="%.6f",
                                            value=float(scn.get(f"rdc{idx}_lon",0.0)),
                                            key=f"lon_{name}_{idx}")
                        lat=st.number_input("Latitude",format="%.6f",
                                            value=float(scn.get(f"rdc{idx}_lat",0.0)),
                                            key=f"lat_{name}_{idx}")
                        typ=st.radio("Type",["RDC","SDC"],horizontal=True,
                                     key=f"typ_{name}_{idx}",
                                     index=0 if scn.get(f"rdc{idx}_typ","RDC")=="RDC" else 1)
                        scn[f"rdc{idx}_lon"]=lon
                        scn[f"rdc{idx}_lat"]=lat
                        scn[f"rdc{idx}_typ"]=typ
            _num_input(scn,"rdc_sqft_per_lb","RDC Sq ft per lb",
                       scn.get("rdc_sqft_per_lb",scn.get("sqft_per_lb",0.02)))
            _num_input(scn,"rdc_cost_sqft","RDC $/sq ft / yr",
                       scn.get("rdc_cost_sqft",scn.get("cost_sqft",6.0)),"%.2f")

        with st.expander("📦 Service Level (optional)", False):
            scn["sl_enforce"] = st.checkbox("Enforce minimum service levels", value=scn.get("sl_enforce", False), key=f"sl_enf_{name}")
            cols = st.columns(2)
            scn["sl_0_350"] = cols[0].number_input("0–350 mi (Next day by 7AM) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_0_350", 0.0)), key=f"sl0350_{name}")
            scn["sl_351_500"] = cols[1].number_input("351–500 mi (Next day by 10AM) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_351_500", 0.0)), key=f"sl351500_{name}")
            cols2 = st.columns(2)
            scn["sl_501_700"] = cols2[0].number_input("501–700 mi (Next day EOD) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_501_700", 0.0)), key=f"sl501700_{name}")
            scn["sl_701p"] = cols2[1].number_input("701+ mi (2 day +) — min %", min_value=0.0, max_value=100.0, step=1.0, value=float(scn.get("sl_701p", 0.0)), key=f"sl701p_{name}")
            tot = scn["sl_0_350"] + scn["sl_351_500"] + scn["sl_501_700"] + scn["sl_701p"]
            if tot > 100.0:
                st.warning(f"Sum of minimums is {tot:.1f}% (> 100%). Exact feasibility may be impossible; I'll minimize total shortfall.")

        st.markdown("---")
        if st.button("🚀 Run solver",key=f"run_{name}"):
            st.session_state["run_target"]=name
    return True

# ---------------- Main ----------------
tab_names=list(st.session_state["scenarios"].keys())+["➕ New scenario"]
tabs=st.tabs(tab_names)

for i,tab in enumerate(tabs[:-1]):
    name=tab_names[i]
    scn=st.session_state["scenarios"][name]
    scn["_name"]=name
    with tab:
        if not sidebar(scn):
            continue

        k_vals=(list(range(int(scn["k_rng"][0]),int(scn["k_rng"][1])+1))
                if scn.get("auto_k",True) else [int(scn["k_fixed"])])

        if st.session_state.get("run_target")==name:
            with st.spinner("Optimizing…"):
                df = st.session_state["cache"].get(f"{name}_demand_df")
                if df is None:
                    df = _read_csv_file(scn["demand_file"])
                    st.session_state["cache"][f"{name}_demand_df"] = df
                required = {"Longitude","Latitude","DemandLbs"}
                missing = required - set(df.columns)
                if missing:
                    st.error(f"Your demand CSV is missing required columns: {sorted(missing)}")
                    st.stop()

                candidate_sites=candidate_costs=None
                if scn.get("cand_file"):
                    cf = _read_optional_csv(scn["cand_file"], header=None)
                    if cf is not None and not cf.empty:
                        cf = cf.dropna(subset=[0,1])
                        if scn.get("restrict_cand"):
                            candidate_sites=cf.iloc[:,:2].values.tolist()
                        if cf.shape[1]>=3:
                            candidate_costs={(round(r[0],6),round(r[1],6)):r[2]
                                             for r in cf.itertuples(index=False)}

                rdc_list=[{"coords":[scn[f"rdc{i}_lon"],scn[f"rdc{i}_lat"]],
                           "is_sdc":scn.get(f"rdc{i}_typ","RDC")=="SDC"}
                          for i in range(1,4) if scn.get(f"rdc{i}_en")]

                sl_targets = {
                    "by7": float(scn.get("sl_0_350", 0.0))/100.0,
                    "by10": float(scn.get("sl_351_500", 0.0))/100.0,
                    "eod": float(scn.get("sl_501_700", 0.0))/100.0,
                    "2day": float(scn.get("sl_701p", 0.0))/100.0,
                }

                res=optimize(
                    df=df,
                    k_vals=k_vals,
                    rate_out=scn["rate_out"],
                    sqft_per_lb=scn["sqft_per_lb"],
                    cost_sqft=scn["cost_sqft"],
                    fixed_cost=scn["fixed_wh_cost"],
                    consider_inbound=scn["inbound_on"],
                    inbound_rate_mile=scn["in_rate"],
                    inbound_pts=scn["inbound_pts"],
                    inbound_rules=scn.get("inbound_rules", []),
                    fixed_centers=scn["fixed_centers"],
                    rdc_list=rdc_list,
                    transfer_rate_mile=scn["trans_rate"],
                    rdc_sqft_per_lb=scn["rdc_sqft_per_lb"],
                    rdc_cost_per_sqft=scn["rdc_cost_sqft"],
                    candidate_sites=candidate_sites,
                    restrict_cand=scn.get("restrict_cand",False),
                    candidate_costs=candidate_costs,
                    service_level_targets=sl_targets,
                    enforce_service_levels=bool(scn.get("sl_enforce", False)),
                    current_state=bool(scn.get("use_current_state", False)),
                    brand_col="Brand",
                    curr_wh_lon_col="CurrWH_Lon",
                    curr_wh_lat_col="CurrWH_Lat",
                    brand_allowed_sites=scn.get("brand_allowed_sites", {}),
                    country_col="Country",
                    canada_enabled=bool(scn.get("can_en", False)),
                    canada_threshold_lon=float(scn.get("can_lon", -105.0)),
                    canada_wh=[float(scn.get("can_wh_lon", -79.3832)), float(scn.get("can_wh_lat", 43.6532))],
                    brand_can_thresholds=scn.get("brand_can_thresholds", {}),
                    brand_overrides_canada=bool(scn.get("brand_overrides_canada", False)),
                    warehouse_brand_allowed=scn.get("wh_brand_allowed", {}),
                )
            summary(res["assigned"],res["total_cost"],res["out_cost"],
                    res["in_cost"],res["trans_cost"],res["wh_cost"],
                    res["centers"],res["demand_per_wh"],
                    scn["sqft_per_lb"],bool(res.get("rdc_list")),
                    scn["inbound_on"],res["trans_cost"]>0)

            st.subheader("Service Levels Achieved")
            sl = res.get("service_levels", {})
            tgt = res.get("sl_targets", {})
            def _pct(x): return f"{(100.0*float(x)):.1f}%"
            a_cols = st.columns(4)
            a_cols[0].metric("≤ 350 mi  (Next day 7AM)", _pct(sl.get("by7", 0.0)), (f"Target {_pct(tgt.get('by7', 0.0))}" if tgt else None))
            a_cols[1].metric("351–500 mi (Next day 10AM)", _pct(sl.get("by10", 0.0)), (f"Target {_pct(tgt.get('by10', 0.0))}" if tgt else None))
            a_cols[2].metric("501–700 mi (Next day EOD)", _pct(sl.get("eod", 0.0)), (f"Target {_pct(tgt.get('eod', 0.0))}" if tgt else None))
            a_cols[3].metric("701+ mi    (2 day +)", _pct(sl.get("2day", 0.0)), (f"Target {_pct(tgt.get('2day', 0.0))}" if tgt else None))
            lanes_df = _build_lane_df(res, scn)


            from utils import haversine as _hz
            import pandas as _pd
            # Flow/brand controls
            flow_opts = []
            if not lanes_df[lanes_df["lane_type"]=="outbound"].empty: flow_opts.append("outbound")
            if not lanes_df[lanes_df["lane_type"]=="transfer"].empty: flow_opts.append("transfer")
            if not lanes_df[lanes_df["lane_type"]=="inbound"].empty: flow_opts.append("inbound")
            flow_sel = st.multiselect("Flows to display", options=["outbound","transfer","inbound"], default=flow_opts or ["outbound"])
            brands_available = (sorted([str(x) for x in res["assigned"]["Brand"].astype(str).unique().tolist()])
                                if "Brand" in res["assigned"].columns else ["ALL"])
            brand_sel = st.selectbox("Brand filter", options=["__ALL__"] + brands_available, index=0, help="Select a single brand or show all")
            from visualization import plot_flows
            plot_flows(lanes_df, res["centers"], flow_types=flow_sel, brand_filter=brand_sel)
            st.download_button("📥 Download lane-level calculations (CSV)",
                               lanes_df.to_csv(index=False).encode("utf-8"),
                               file_name=f"{name}_lanes.csv",mime="text/csv")

with tabs[-1]:
    new_name=st.text_input("Scenario name")
    if st.button("Create scenario"):
        if new_name and new_name not in st.session_state["scenarios"]:
            st.session_state["scenarios"][new_name]={}
            if hasattr(st,"rerun"):
                st.rerun()
            else:
                st.experimental_rerun()