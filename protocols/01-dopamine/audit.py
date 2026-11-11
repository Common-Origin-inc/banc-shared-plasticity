"""Protocol 01 audit: modulation-source identity.

Read-only on all inputs. Deterministic: no timestamps, no randomness, sorted outputs.

usage:
  python audit.py --meta PATH/banc_888_meta.feather \
                  --da-edges BANC_BEST_BRAIN/da_edges.npz \
                  --banc-full BANC_BEST_BRAIN/banc_full.npz \
                  [--out-json audit.json] [--out-md AUDIT.md]

The feather file is the BANC metadata table from the BANC data release (see the README). The
script stops if its SHA-256 or size differs from the values recorded in PROTOCOL.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA256 = "86ccf5df0c67419f8c5f43e93a7ed38d23a080e9f7fde26737290252f3780098"
EXPECTED_SIZE = 57503026

# The ten root_888 identifiers of the dual-classified sources, in a fixed order.
TEN_IDS = [
    720575941510925010, 720575941658225944, 720575941506180642, 720575941354201904,
    720575941595402353, 720575941515763459, 720575941601553193, 720575941591874302,
    720575941563944455, 720575941563116456,
]

# Values stated in the manuscript / protocol, kept only for side-by-side comparison.
MANUSCRIPT = {
    "modulation_edges": 33054,
    "modulation_sources": 380,
    "epg_modulation_contacts": 4573,
    "er4m_epg_modulation_contacts": 2355,
    "er4m_epg_share_percent": 51.5,
    "dual_classified_sources": 10,
}

# Columns recorded verbatim for the ten identifiers.
ROW_COLUMNS = [
    "root_888", "root_id", "status", "proofread", "side", "region", "hemilineage",
    "super_class", "cell_class", "cell_sub_class", "cell_type",
    "neurotransmitter_verified", "neurotransmitter_predicted", "neurotransmitter_score",
    "neuropeptide_verified",
    "fafb_cell_type", "fafb_alignment_cell_type", "hemibrain_cell_type", "malecns_cell_type",
    "manc_cell_type", "fanc_cell_type",
    "fafb_match", "hemibrain_match", "malecns_match", "manc_match", "fanc_match",
    "fafb_nblast_match", "hemibrain_nblast_match", "malecns_nblast_match",
]

# Substrings used to look for any column that could state where an annotation came from.
PROVENANCE_HINTS = ("source", "prov", "citation", "reference", "evidence", "paper", "doi",
                    "author", "annotator", "user", "date", "comment", "note", "match")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verbatim(v):
    """JSON-safe copy of a cell, with strings untouched."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.floating, float)):
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if v is pd.NA:
        return None
    return str(v)


def tokens(s: str) -> list[str]:
    return [t.strip().lower() for t in s.split(",")]


def position_class(s: str) -> str:
    t = tokens(s)
    if "dopamine" not in t:
        return "no_dopamine_token"
    if t == ["dopamine"]:
        return "only_token"
    if t[0] == "dopamine":
        return "first_of_several"
    return "not_first"


def table_stats(mask_edge, pre, post, nsyn, is_epg):
    return {
        "sources": int(len(np.unique(pre[mask_edge]))),
        "edges": int(mask_edge.sum()),
        "contacts": int(round(float(nsyn[mask_edge].sum()))),
        "targets": int(len(np.unique(post[mask_edge]))),
        "contacts_onto_epg": int(round(float(nsyn[mask_edge & is_epg[post]].sum()))),
        "epg_cells_with_any_contact": int(len(np.unique(post[mask_edge & is_epg[post]]))),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, type=Path)
    ap.add_argument("--da-edges", required=True, type=Path)
    ap.add_argument("--banc-full", required=True, type=Path)
    here = Path(__file__).resolve().parent
    ap.add_argument("--out-json", type=Path, default=here / "audit.json")
    ap.add_argument("--out-md", type=Path, default=here / "AUDIT.md")
    a = ap.parse_args()

    # ------------------------------------------------------------ step 1: hash
    size = a.meta.stat().st_size
    digest = sha256_of(a.meta)
    if digest != EXPECTED_SHA256 or size != EXPECTED_SIZE:
        print(f"STOP: feather hash/size mismatch: sha256 {digest}, size {size}", file=sys.stderr)
        return 2

    meta = pd.read_feather(a.meta)
    meta_rid = meta["root_888"].astype(np.int64)
    da = np.load(a.da_edges, allow_pickle=True)
    g = np.load(a.banc_full, allow_pickle=True)
    pre = da["pre_idx"].astype(np.int64)
    post = da["post_idx"].astype(np.int64)
    nsyn = da["n_syn"].astype(float)
    root = g["root_id"].astype(np.int64)
    ct = g["cell_type"].astype(str)
    nt = g["nt"].astype(str)
    nt_source = g["nt_source"].astype(str)
    n = len(root)

    ver_by_root = dict(zip(meta_rid.tolist(), meta["neurotransmitter_verified"].tolist()))
    row_by_root = {int(r): i for i, r in enumerate(meta_rid.tolist())}

    # ------------------------------------------------------------ step 2: ten rows
    prov_like = [c for c in meta.columns if any(h in c.lower() for h in PROVENANCE_HINTS)]
    ten_rows = []
    for rid in TEN_IDS:
        if rid not in row_by_root:
            ten_rows.append({"root_888": str(rid), "found_in_meta": False})
            continue
        r = meta.iloc[row_by_root[rid]]
        row = {c: verbatim(r[c]) for c in ROW_COLUMNS if c in meta.columns}
        row["found_in_meta"] = True
        ten_rows.append(row)

    # ------------------------------------------------------------ step 3: all sources
    src_idx = np.unique(pre)
    src_str = {}
    for i in src_idx:
        v = ver_by_root.get(int(root[i]))
        src_str[int(i)] = "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)
    src_class = {i: position_class(s) for i, s in src_str.items()}
    pos_counts = {k: 0 for k in ("only_token", "first_of_several", "not_first", "no_dopamine_token")}
    for c in src_class.values():
        pos_counts[c] += 1
    string_table = {}
    for i, s in src_str.items():
        e = string_table.setdefault(s, {"string": s, "position": src_class[i], "sources": 0,
                                        "edges": 0, "contacts": 0})
        e["sources"] += 1
    edge_str = np.array([src_str[int(i)] for i in pre], dtype=object)
    for s, e in string_table.items():
        m = edge_str == s
        e["edges"] = int(m.sum())
        e["contacts"] = int(round(float(nsyn[m].sum())))
    string_rows = sorted(string_table.values(), key=lambda e: (-e["sources"], e["string"]))

    # Whole-metadata tabulation of the substring selector, for context.
    ver_all = meta["neurotransmitter_verified"].astype("string")
    sub = ver_all.str.lower().str.contains("dopamine", na=False).to_numpy()
    in_sim = meta_rid.isin(set(root.tolist())).to_numpy()
    meta_strings = (ver_all[sub].value_counts().sort_index()
                    .rename_axis("s").reset_index(name="n"))
    meta_string_rows = []
    for s, cnt in zip(meta_strings["s"].tolist(), meta_strings["n"].tolist()):
        m = (ver_all == s).fillna(False).to_numpy()
        meta_string_rows.append({"string": s, "position": position_class(s),
                                 "rows_in_meta": int(cnt), "rows_in_banc_full": int((m & in_sim).sum()),
                                 "sources_in_da_edges": int(string_table.get(s, {}).get("sources", 0))})
    meta_string_rows.sort(key=lambda e: (-e["rows_in_meta"], e["string"]))

    # ------------------------------------------------------------ step 4: EPG
    is_epg = ct == "EPG"
    all_edges = np.ones(len(pre), bool)
    total = table_stats(all_edges, pre, post, nsyn, is_epg)
    epg_total = total["contacts_onto_epg"]

    is_src = np.zeros(n, bool)
    is_src[src_idx] = True
    dual_idx = np.flatnonzero(is_src & (nt_source == "verified"))
    idx_of_root = {int(r): i for i, r in enumerate(root.tolist())}

    fpre = g["pre_idx"].astype(np.int64)
    fpost = g["post_idx"].astype(np.int64)
    fn = g["n_syn"].astype(np.int64)

    def per_source(i: int) -> dict:
        m = pre == i
        me = m & is_epg[post]
        fm = (fpre == i) & is_epg[fpost]
        # identical (target, count) pairs onto EPG in both tables?
        mod_pairs = sorted(zip(post[me].tolist(), [int(round(x)) for x in nsyn[me].tolist()]))
        fast_pairs = sorted(zip(fpost[fm].tolist(), fn[fm].tolist()))
        c = int(round(float(nsyn[me].sum())))
        return {
            "root_888": str(int(root[i])), "banc_full_index": int(i),
            "cell_type_banc_full": str(ct[i]), "nt_banc_full": str(nt[i]),
            "nt_source_banc_full": str(nt_source[i]),
            "neurotransmitter_verified": src_str.get(int(i), None),
            "modulation_edges": int(m.sum()),
            "modulation_contacts": int(round(float(nsyn[m].sum()))),
            "modulation_contacts_onto_epg": c,
            "share_of_epg_modulation_contacts_percent": round(100.0 * c / epg_total, 4) if epg_total else None,
            "fast_graph_contacts_onto_epg": int(fn[fm].sum()),
            "epg_edges_identical_in_both_tables": mod_pairs == fast_pairs,
        }

    ten_contrib = []
    for rid in TEN_IDS:
        i = idx_of_root.get(rid)
        if i is None:
            ten_contrib.append({"root_888": str(rid), "in_banc_full": False})
        else:
            d = per_source(i)
            d["in_banc_full"] = True
            d["is_modulation_source"] = bool(is_src[i])
            ten_contrib.append(d)
    dual_found = sorted(str(int(root[i])) for i in dual_idx)
    ten_set = sorted(str(r) for r in TEN_IDS)
    er4m_ten = [d for d in ten_contrib if d.get("cell_type_banc_full") == "ER4m"]
    er4m_epg = sum(d["modulation_contacts_onto_epg"] for d in er4m_ten)
    ten_epg = sum(d.get("modulation_contacts_onto_epg", 0) for d in ten_contrib)

    # EPG modulation contacts by source cell type, for context.
    by_type = {}
    me = is_epg[post]
    for i, c in zip(pre[me].tolist(), nsyn[me].tolist()):
        k = (str(ct[i]), src_str[int(i)])
        by_type[k] = by_type.get(k, 0.0) + c
    epg_by_type = [{"source_cell_type": k[0], "neurotransmitter_verified": k[1],
                    "contacts_onto_epg": int(round(v)),
                    "share_percent": round(100.0 * v / epg_total, 4)}
                   for k, v in sorted(by_type.items(), key=lambda kv: (-kv[1], kv[0]))]

    # ------------------------------------------------------------ step 5: alternative selectors
    edge_class = np.array([src_class[int(i)] for i in pre], dtype=object)
    sel_a = np.isin(edge_class, ["only_token", "first_of_several"])
    sel_b = edge_class == "only_token"
    alt = {
        "current_substring": total,
        "a_first_token_is_dopamine": table_stats(sel_a, pre, post, nsyn, is_epg),
        "b_dopamine_only_token": table_stats(sel_b, pre, post, nsyn, is_epg),
    }
    # Sources kept by (a) that banc_full also treats as a verified fast source: should be none,
    # because both rules then read the same first token.
    a_src = np.unique(pre[sel_a])
    alt["a_first_token_is_dopamine"]["sources_also_verified_fast_in_banc_full"] = int(
        (nt_source[a_src] == "verified").sum())
    alt["a_first_token_is_dopamine"]["source_nt_source_in_banc_full"] = {
        k: int(v) for k, v in sorted(pd.Series(nt_source[a_src]).value_counts().items())}

    # ------------------------------------------------------------ consistency checks
    checks = {
        "is_dan_equals_unique_pre_idx": bool(np.array_equal(np.flatnonzero(da["is_dan"]), src_idx)),
        "da_in_equals_bincount": bool(np.allclose(
            np.bincount(post, weights=nsyn, minlength=n), da["da_in"])),
        "all_sources_found_in_meta": bool(all(int(root[i]) in row_by_root for i in src_idx)),
        "all_sources_contain_dopamine_substring": bool(all("dopamine" in s.lower() for s in src_str.values())),
        "root_888_unique_in_meta": bool(not meta_rid.duplicated().any()),
        "meta_rows_with_dopamine_substring": int(sub.sum()),
        "of_which_in_banc_full": int((sub & in_sim).sum()),
        "of_which_in_banc_full_without_any_da_edge": int(
            (sub & in_sim).sum() - len(src_idx)),
        "n_epg_cells": int(is_epg.sum()),
        "n_epg_cells_that_are_cx_core": int((is_epg & g["is_cx_core"].astype(bool)).sum()),
    }

    recomputed = {
        "modulation_edges": total["edges"],
        "modulation_sources": total["sources"],
        "epg_modulation_contacts": epg_total,
        "er4m_epg_modulation_contacts": er4m_epg,
        "er4m_epg_share_percent": round(100.0 * er4m_epg / epg_total, 4) if epg_total else None,
        "dual_classified_sources": int(len(dual_idx)),
    }
    comparison = {k: {"manuscript": MANUSCRIPT[k], "recomputed": recomputed[k],
                      "agree": (round(recomputed[k], 1) == MANUSCRIPT[k]
                                if isinstance(MANUSCRIPT[k], float) else recomputed[k] == MANUSCRIPT[k])}
                  for k in MANUSCRIPT}

    out = {
        "protocol": "01-dopamine",
        "inputs": {
            "meta_feather": {"name": a.meta.name, "sha256": digest, "size_bytes": size,
                             "expected_sha256": EXPECTED_SHA256, "expected_size_bytes": EXPECTED_SIZE,
                             "verified": True, "rows": int(len(meta)), "columns": list(meta.columns)},
            "da_edges_npz": {"name": a.da_edges.name, "sha256": sha256_of(a.da_edges)},
            "banc_full_npz": {"name": a.banc_full.name, "sha256": sha256_of(a.banc_full)},
        },
        "columns_matching_provenance_hints": prov_like,
        "ten_ids_verbatim_rows": ten_rows,
        "modulation_sources": {
            "n_sources": int(len(src_idx)),
            "dopamine_position_counts": pos_counts,
            "strings": string_rows,
        },
        "meta_wide_dopamine_strings": meta_string_rows,
        "epg": {
            "definition": "banc_full cell_type == 'EPG'",
            "n_epg_cells": int(is_epg.sum()),
            "modulation_contacts_onto_epg": epg_total,
            "ten_ids": ten_contrib,
            "ten_ids_contacts_onto_epg": ten_epg,
            "er4m_contacts_onto_epg": er4m_epg,
            "er4m_share_percent": recomputed["er4m_epg_share_percent"],
            "by_source_cell_type": epg_by_type,
        },
        "dual_classified": {
            "definition": "modulation source in da_edges.npz whose banc_full nt_source == 'verified'",
            "n": int(len(dual_idx)),
            "root_888": dual_found,
            "same_set_as_ten_ids": dual_found == ten_set,
        },
        "alternative_selectors": alt,
        "consistency_checks": checks,
        "manuscript_comparison": comparison,
    }
    a.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    a.out_md.write_text(render_md(out))
    print(f"wrote {a.out_json} and {a.out_md}")
    return 0


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    return "\n".join(lines)


def code(v):
    return "(null)" if v is None else f"`{v}`"


def render_md(o) -> str:
    L = []
    inp = o["inputs"]
    L.append("# Audit 01: modulation-source identity\n")
    L.append("Generated by `protocols/01-dopamine/audit.py` from the inputs below. "
             "Machine-readable copy: `audit.json`. This file reports what the annotation table "
             "contains. It does not decide between keep and rebuild.\n")
    L.append("## 1. Inputs\n")
    m = inp["meta_feather"]
    L.append(md_table(["File", "SHA-256", "Bytes"], [
        [f"`{m['name']}`", f"`{m['sha256']}`", f"{m['size_bytes']:,}"],
        [f"`{inp['da_edges_npz']['name']}`", f"`{inp['da_edges_npz']['sha256']}`", ""],
        [f"`{inp['banc_full_npz']['name']}`", f"`{inp['banc_full_npz']['sha256']}`", ""]]))
    L.append(f"\nThe feather hash and size match the expected values "
             f"(`{m['expected_sha256']}`, {m['expected_size_bytes']:,} bytes). "
             f"The table has {m['rows']:,} rows and {len(m['columns'])} columns.\n")

    L.append("## 2. The ten identifiers, verbatim\n")
    rows = o["ten_ids_verbatim_rows"]
    L.append(md_table(
        ["root_888", "cell_type", "super_class", "neurotransmitter_verified",
         "neurotransmitter_predicted", "neurotransmitter_score", "neuropeptide_verified"],
        [[r["root_888"], code(r.get("cell_type")), code(r.get("super_class")),
          code(r.get("neurotransmitter_verified")), code(r.get("neurotransmitter_predicted")),
          r.get("neurotransmitter_score"), code(r.get("neuropeptide_verified"))] for r in rows]))
    L.append("\nOther recorded columns for the same rows:\n")
    L.append(md_table(
        ["root_888", "cell_class", "cell_sub_class", "hemilineage", "side", "status", "proofread"],
        [[r["root_888"], code(r.get("cell_class")), code(r.get("cell_sub_class")),
          code(r.get("hemilineage")), code(r.get("side")), code(r.get("status")),
          code(r.get("proofread"))] for r in rows]))
    L.append("\nCross-dataset cell-type and match columns (the only columns in the table that "
             "point to another dataset):\n")
    L.append(md_table(
        ["root_888", "fafb_cell_type", "hemibrain_cell_type", "malecns_cell_type",
         "fafb_match", "hemibrain_match", "malecns_match"],
        [[r["root_888"], code(r.get("fafb_cell_type")), code(r.get("hemibrain_cell_type")),
          code(r.get("malecns_cell_type")), code(r.get("fafb_match")),
          code(r.get("hemibrain_match")), code(r.get("malecns_match"))] for r in rows]))
    L.append("\nColumns whose names match a provenance keyword search "
             "(source, prov, citation, reference, evidence, paper, doi, author, annotator, user, "
             "date, comment, note, match): "
             + ", ".join(f"`{c}`" for c in o["columns_matching_provenance_hints"]) + ". "
             "All of them are cell-matching columns. The table has no column that states the "
             "source, method, citation, annotator or date of `neurotransmitter_verified`. "
             "The full column list is in `audit.json`.\n")

    L.append("## 3. All modulation sources in `da_edges.npz`\n")
    ms = o["modulation_sources"]
    pc = ms["dopamine_position_counts"]
    L.append(f"Sources: {ms['n_sources']}. Position of the `dopamine` token in "
             "`neurotransmitter_verified` (comma-split, lower-cased):\n")
    L.append(md_table(["Position", "Sources"], [
        ["dopamine is the only token", pc["only_token"]],
        ["dopamine is the first of several tokens", pc["first_of_several"]],
        ["dopamine is present but not first", pc["not_first"]],
        ["no dopamine token", pc["no_dopamine_token"]]]))
    L.append("\nFull strings among the sources:\n")
    L.append(md_table(["neurotransmitter_verified", "Position", "Sources", "Edges", "Contacts"],
                      [[code(s["string"]), s["position"], s["sources"], f"{s['edges']:,}",
                        f"{s['contacts']:,}"] for s in ms["strings"]]))
    L.append("\nFor context, every string in the whole metadata table that contains `dopamine`:\n")
    L.append(md_table(["neurotransmitter_verified", "Position", "Rows in metadata",
                       "Rows in banc_full", "Sources in da_edges"],
                      [[code(s["string"]), s["position"], s["rows_in_meta"], s["rows_in_banc_full"],
                        s["sources_in_da_edges"]] for s in o["meta_wide_dopamine_strings"]]))
    c = o["consistency_checks"]
    L.append(f"\nOf {c['meta_rows_with_dopamine_substring']:,} metadata rows containing the substring, "
             f"{c['of_which_in_banc_full']:,} are in `banc_full.npz`, and "
             f"{c['of_which_in_banc_full_without_any_da_edge']:,} of those have no edge in `da_edges.npz`.\n")

    L.append("## 4. Modulation contacts onto EPG, recomputed\n")
    e = o["epg"]
    L.append(f"EPG is defined as `cell_type == 'EPG'` in `banc_full.npz`: {e['n_epg_cells']} cells "
             f"({c['n_epg_cells_that_are_cx_core']} flagged `is_cx_core`). "
             f"Total modulation contacts onto EPG: {e['modulation_contacts_onto_epg']:,}.\n")
    L.append(md_table(
        ["root_888", "Cell type", "Fast transmitter (banc_full)", "Modulation contacts onto EPG",
         "Share of EPG total (%)", "Fast-graph contacts onto EPG", "EPG edges identical in both tables"],
        [[d["root_888"], d.get("cell_type_banc_full"), d.get("nt_banc_full"),
          f"{d.get('modulation_contacts_onto_epg', 0):,}",
          f"{d.get('share_of_epg_modulation_contacts_percent', 0):.2f}",
          f"{d.get('fast_graph_contacts_onto_epg', 0):,}",
          d.get("epg_edges_identical_in_both_tables")] for d in e["ten_ids"]]))
    L.append(f"\nFour ER4m sources together: {e['er4m_contacts_onto_epg']:,} of "
             f"{e['modulation_contacts_onto_epg']:,} contacts = {e['er4m_share_percent']:.2f}%. "
             f"All ten together: {e['ten_ids_contacts_onto_epg']:,}.\n")
    dc = o["dual_classified"]
    L.append(f"Dual-classified sources (modulation source and `nt_source == 'verified'` in "
             f"`banc_full.npz`): {dc['n']}. Same set as the ten listed identifiers: "
             f"{dc['same_set_as_ten_ids']}.\n")
    L.append("EPG modulation contacts by source cell type and annotation string:\n")
    L.append(md_table(["Source cell type", "neurotransmitter_verified", "Contacts onto EPG", "Share (%)"],
                      [[code(r["source_cell_type"]), code(r["neurotransmitter_verified"]),
                        f"{r['contacts_onto_epg']:,}", f"{r['share_percent']:.2f}"]
                       for r in e["by_source_cell_type"]]))

    L.append("\n## 5. Alternative selectors\n")
    al = o["alternative_selectors"]
    names = [("current_substring", "Current: `dopamine` anywhere in the string"),
             ("a_first_token_is_dopamine", "(a) first token equals `dopamine`"),
             ("b_dopamine_only_token", "(b) `dopamine` is the only token")]
    L.append(md_table(["Selector", "Sources", "Edges", "Contacts", "Targets", "Contacts onto EPG",
                       "EPG cells reached"],
                      [[lab, al[k]["sources"], f"{al[k]['edges']:,}", f"{al[k]['contacts']:,}",
                        f"{al[k]['targets']:,}", f"{al[k]['contacts_onto_epg']:,}",
                        al[k]["epg_cells_with_any_contact"]] for k, lab in names]))
    aa = al["a_first_token_is_dopamine"]
    L.append("\nBoth alternatives were computed by filtering the committed `da_edges.npz` by source. "
             "This is exact because every source kept by (a) or (b) is also kept by the current "
             "selector and edges are selected per source. The raw edge list was not re-read. "
             f"Under (a), {aa['sources_also_verified_fast_in_banc_full']} sources are also verified "
             "fast sources in `banc_full.npz`; their `nt_source` values there are "
             + ", ".join(f"`{k}`: {v}" for k, v in aa["source_nt_source_in_banc_full"].items()) + ".\n")

    L.append("## 6. Comparison with the manuscript\n")
    L.append(md_table(["Quantity", "Manuscript", "Recomputed", "Agree"],
                      [[k, v["manuscript"], v["recomputed"], v["agree"]]
                       for k, v in o["manuscript_comparison"].items()]))
    L.append("\nConsistency checks on the committed arrays:\n")
    L.append(md_table(["Check", "Result"], [[k, v] for k, v in c.items()]))

    L.append("\n## 7. What the annotations do and do not establish\n")
    L.append(render_establish(o))
    L.append("\n## 8. Reproduce\n")
    L.append("```bash\npython protocols/01-dopamine/audit.py \\\n"
             "  --meta <path to the BANC metadata table>/banc_888_meta.feather \\\n"
             "  --da-edges BANC_BEST_BRAIN/da_edges.npz \\\n"
             "  --banc-full BANC_BEST_BRAIN/banc_full.npz\n```\n")
    L.append("The script reads its inputs and writes only `audit.json` and `AUDIT.md`. "
             "It exits with status 2 before reading the table if the hash or size differ from the expected values.\n")
    return "\n".join(L)


def render_establish(o) -> str:
    rows = [r for r in o["ten_ids_verbatim_rows"] if r.get("cell_type") == "ER4m"]
    er_strings = sorted({str(r.get("neurotransmitter_verified")) for r in rows})
    er_pred = sorted({str(r.get("neurotransmitter_predicted")) for r in rows})
    er_scores = sorted(r.get("neurotransmitter_score") for r in rows)
    pc = o["modulation_sources"]["dopamine_position_counts"]
    return (
        "Established by the table: the four ER4m rows carry the `neurotransmitter_verified` "
        f"string {', '.join(code(s) for s in er_strings)}, so `dopamine` is an explicit token in a "
        "column named verified, in second position. The same four rows carry "
        f"`neurotransmitter_predicted` = {', '.join(code(s) for s in er_pred)} with scores "
        f"{', '.join(str(s) for s in er_scores)}. Among all {o['modulation_sources']['n_sources']} "
        f"modulation sources, dopamine is the only token for {pc['only_token']}, the first of "
        f"several for {pc['first_of_several']}, and not first for {pc['not_first']}. "
        "The dual use of the ten sources follows from the two parsing rules and is reproduced here "
        "from the committed arrays.\n\n"
        "Not established by the table: where the label came from. The file has no column giving a "
        "source, method, citation, annotator or date for `neurotransmitter_verified`, for any row. "
        "It does not say whether the label was assigned to the individual neuron or inherited from "
        "its cell type, what the token order means, or whether a second token denotes cotransmission "
        "onto the same targets. The table therefore contains an explicit dopamine label for all four "
        "ER4m neurons and no stated source for it. Whether that satisfies the keep condition in "
        "`PROTOCOL.md` (\"an explicit dopamine label with a stated source\") is a decision "
        "made before protocol 02.\n")


if __name__ == "__main__":
    raise SystemExit(main())
