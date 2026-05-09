"""
BIM-MOO Decision Engine | TRC Project
Element-Level Disaggregated Optimization
Methods: NSGA-II (primary) + WSM + TOPSIS + VIKOR (cross-validation)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import random
from copy import deepcopy

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="BIM-MOO Decision Engine | TRC", layout="wide")

st.markdown("""
<style>
.main { background-color: #0e1117; }
[data-testid="stMetricValue"] {
    font-size: 1.0rem !important; color: #00d4ff !important; font-weight: 700;
}
[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important; color: #adb5bd !important;
    text-transform: uppercase; letter-spacing: 1px;
}
[data-testid="stMetric"] {
    background-color: #161b22; border-radius: 8px;
    border: 1px solid #30363d; border-left: 4px solid #00d4ff; padding: 12px;
}
h1 { color: #ffffff !important; font-family: 'Segoe UI', sans-serif;
     border-bottom: 2px solid #00d4ff; padding-bottom: 10px; }
h2, h3 { color: #e6edf3 !important; font-family: 'Segoe UI', sans-serif;
     border-bottom: 1px solid #30363d; padding-bottom: 8px; }
.section-desc {
    color: #8b949e; font-style: italic; font-size: 0.91rem;
    line-height: 1.7; margin-bottom: 16px;
    border-left: 3px solid #30363d; padding-left: 12px;
}
.stAlert { background-color: #161b22 !important; border: 1px solid #30363d !important; }
[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] .stMarkdown { color: #c9d1d9; }
</style>
""", unsafe_allow_html=True)

# ── Formatters ────────────────────────────────────────────────────────────────
def fmt_inr(v):
    v = float(v)
    if v >= 1_00_00_000: return f"₹{v/1_00_00_000:.2f} Cr"
    elif v >= 1_00_000:   return f"₹{v/1_00_000:.2f} L"
    else:                 return f"₹{int(round(v)):,}"

def fmt_ec(v):
    return f"{int(round(float(v))):,} kgCO₂e"

# ── BIM Quantities (Dynamo-extracted, element-level) ─────────────────────────
Q = {
    "structural_concrete": 283.62,   # Beams (142.17) + Columns (141.45) m³
    "slab_concrete":       155.08,   # Floor Slabs m³
    "wall_concrete":       122.07,   # Structural Walls m³
    "wall_screed":          73.14,   # Wall Screed m³
    "masonry":             365.69,   # Masonry Walls m³
    "flooring":           1550.09,   # Floor Finishing m²
}

BASELINE_COST   = 81_01_473
BASELINE_CARBON = 36_28_000

# ── Technical descriptions ────────────────────────────────────────────────────
material_insights = {
    # STRUCTURAL CONCRETE
    "OPC Concrete M25":
        "Ordinary Portland Cement, IS 456:2000. Minimum permissible grade for beams and columns "
        "per IS 456 Cl.6.1.2. Highest clinker content; embodied carbon and cost baseline. EC: 330 kgCO₂e/m³.",
    "PPC Concrete M25 (20% Fly Ash)":
        "Portland Pozzolana Cement per IS 1489 Part 1:1991. 20% fly ash replaces clinker; "
        "21% EC reduction over OPC M25. BIS-approved for all structural elements. EC: 260 kgCO₂e/m³.",
    "PSC Concrete M25 (GGBS 30%)":
        "GGBS blend per IS 16714:2018 at 30% replacement. 36% EC reduction. Superior sulfate "
        "resistance; recommended for TRC foundations and columns. EC: 210 kgCO₂e/m³.",
    "LC3 Concrete M25 (Calcined Clay)":
        "Limestone Calcined Clay Cement, 40% clinker substitution (Scrivener et al., 2018). "
        "41% EC reduction. Under BIS evaluation; structural engineer approval required. EC: 195 kgCO₂e/m³.",
    "Green Concrete M25 (Fly Ash 40%)":
        "High-volume fly ash per IS 10262:2019. 40% FA replacement; 47% EC reduction. "
        "Requires approved mix design beyond IS 456 Cl.5.2 limit of 35%. EC: 175 kgCO₂e/m³.",
    "High Volume GGBS M25 (50%)":
        "IS 16714:2018 GGBS at 50% replacement. 61% EC reduction. Very low heat of hydration; "
        "long-term strength exceeds OPC at 90 days. Suitable for columns. EC: 130 kgCO₂e/m³.",
    "RMC M30 OPC":
        "IS 4926:2003 ready-mixed M30. Higher EC than site-mixed due to production overhead. "
        "Superior QC for critical transfer beams and columns. EC: 360 kgCO₂e/m³.",
    "Geopolymer Concrete M25":
        "Cement-free; fly ash/GGBS with NaOH+Na₂SiO₃ activator. 76% EC reduction. "
        "No current IS code — requires project-specific mix validation. EC: 80 kgCO₂e/m³.",
    "Nano-Silica Enhanced M25":
        "Nanometric SiO₂ at 1–3% cement weight. 77% EC reduction. High cost; "
        "applicable to specialised high-density structural elements. EC: 75 kgCO₂e/m³.",
    "Recycled Aggregate M25":
        "Coarse aggregate replaced with recycled demolition concrete per draft IS 383 Pt3. "
        "Kochi Metro demolition aggregate viable local source. EC: 215 kgCO₂e/m³.",
    # SLAB CONCRETE
    "OPC Concrete M20":
        "OPC M20 per IS 456:2000. Minimum permissible grade for floor slabs in general structures. EC: 300 kgCO₂e/m³.",
    "PPC Concrete M20 (20% Fly Ash)":
        "PPC M20 per IS 1489 Pt1. 18% EC reduction over OPC M20 at slab grade. EC: 245 kgCO₂e/m³.",
    "PSC Concrete M20 (GGBS 30%)":
        "GGBS M20 per IS 16714:2018. 33% EC reduction. Reduced heat of hydration for large slab pours. EC: 200 kgCO₂e/m³.",
    "Green Concrete M20 (Fly Ash 40%)":
        "HVFA M20 per IS 10262:2019. 45% EC reduction over OPC M20. Approved mix design mandatory. EC: 165 kgCO₂e/m³.",
    "High Volume GGBS M20 (50%)":
        "GGBS at 50% for slabs per IS 16714:2018. 60% EC reduction. Well-suited for raft and ground floor slabs. EC: 120 kgCO₂e/m³.",
    "Geopolymer Concrete M20":
        "Cement-free geopolymer at slab grade. 75% EC reduction. No IS code; engineer discretion. EC: 75 kgCO₂e/m³.",
    "Recycled Aggregate M20":
        "RA concrete for slabs per draft IS 383 Pt3. Adjusted water-cement ratio required. EC: 205 kgCO₂e/m³.",
    "LC3 Concrete M20":
        "LC3 at slab grade. 38% EC reduction. Under BIS evaluation. EC: 185 kgCO₂e/m³.",
    "Carbon-Cured Concrete (Precast)":
        "CO₂ mineralised as CaCO₃ in precast curing (CarbonCure EPD). Precast slab panels only. "
        "Not commercially available in Kerala as of 2024. EC: 45 kgCO₂e/m³.",
    "Nano-Silica Enhanced M20":
        "Nano-SiO₂ at slab grade. 77% EC reduction. Research grade; limited availability. EC: 70 kgCO₂e/m³.",
    # WALL CONCRETE
    "OPC Concrete M25 (Wall)":
        "OPC M25 for load-bearing structural walls per IS 456:2000. EC: 330 kgCO₂e/m³.",
    "PPC Concrete M25 (Wall, 20% FA)":
        "PPC M25 for structural walls per IS 1489 Pt1. 21% EC reduction. EC: 260 kgCO₂e/m³.",
    "PSC Concrete M25 (Wall, GGBS 30%)":
        "GGBS M25 for walls per IS 16714:2018. Superior sulfate resistance for basement walls. EC: 210 kgCO₂e/m³.",
    "High Volume GGBS M25 Wall (50%)":
        "GGBS 50% for structural walls. 61% EC reduction. Low heat; ideal for thick wall sections. EC: 130 kgCO₂e/m³.",
    "Green Concrete M25 Wall (FA 40%)":
        "HVFA M25 for walls per IS 10262:2019. 47% EC reduction. Approved mix design required. EC: 175 kgCO₂e/m³.",
    "Geopolymer Concrete M25 (Wall)":
        "Geopolymer M25 for structural walls. 76% EC reduction. No IS code. EC: 80 kgCO₂e/m³.",
    "LC3 Concrete M25 (Wall)":
        "LC3 M25 for structural walls. 41% EC reduction. BIS evaluation ongoing. EC: 195 kgCO₂e/m³.",
    "Recycled Aggregate M25 (Wall)":
        "RA M25 for structural walls per draft IS 383 Pt3. EC: 215 kgCO₂e/m³.",
    # WALL SCREED
    "OPC Cement Screed (1:3)":
        "Standard OPC cement-sand screed per IS 2116:1980. 1:3 mix; 40mm thickness. Institutional baseline. EC: 280 kgCO₂e/m³.",
    "PPC Cement Screed (1:3)":
        "PPC-based screed per IS 1489 Pt1. 21% EC reduction over OPC screed. EC: 220 kgCO₂e/m³.",
    "Fly Ash Blended Screed":
        "30% fly ash replacement in screed mix per IS 10262:2019. 43% EC reduction. EC: 160 kgCO₂e/m³.",
    "GGBS Blended Screed":
        "GGBS partial replacement in screed per IS 16714:2018. 50% EC reduction. EC: 140 kgCO₂e/m³.",
    "Lime-Sand Screed (Non-structural)":
        "Lime-sand mix per IS 712:1984 for non-structural partition backing. 70% EC reduction. EC: 85 kgCO₂e/m³.",
    "Gypsum-Based Screed":
        "Gypsum binder per IS 2547:1976. Interior walls only. Low EC; rapid setting. EC: 95 kgCO₂e/m³.",
    "Geo-Polymer Screed":
        "Geopolymer binder screed. 80% EC reduction. No IS code; research application. EC: 55 kgCO₂e/m³.",
    "Recycled Aggregate Screed":
        "RA screed per draft IS 383 Pt3. 37% EC reduction over OPC screed. EC: 175 kgCO₂e/m³.",
    # MASONRY
    "Burnt Clay Bricks IS 1077":
        "IS 1077:1992 Grade A (≥5 N/mm²). Traditional baseline; high topsoil extraction and kiln emissions. Not recommended near thermal plants. EC: 270 kgCO₂e/m³.",
    "Wire-Cut Clay Bricks":
        "Machine-extruded per IS 1077:1992. Better dimensional accuracy than hand-moulded; consistent kiln utilisation. EC: 205 kgCO₂e/m³.",
    "Fly Ash Bricks Grade A IS 12894":
        "IS 12894:2002 Grade A (≥75 kg/cm²). Mandatory within 100km of thermal power plants (MoEFCC 2009). 52% EC reduction. EC: 130 kgCO₂e/m³.",
    "AAC Blocks IS 2185 Pt3":
        "IS 2185 Pt3:1984 Grade A (≥3.5 N/mm²). 67% EC reduction. Best thermal performance (U ≈ 0.5 W/m²K). Aerocon/Siporex available in Kerala. EC: 88 kgCO₂e/m³.",
    "Hollow Concrete Blocks IS 2185":
        "IS 2185 Pt1:2005 Grade D (≥3.5 N/mm²). Low dead load; infill/non-load-bearing walls. EC: 165 kgCO₂e/m³.",
    "Solid Concrete Blocks IS 2185":
        "IS 2185 Pt1:2005 Grade A (≥7.5 N/mm²). Load-bearing external walls. EC: 180 kgCO₂e/m³.",
    "Fal-G Blocks (FA+Lime+Gypsum)":
        "MoEFCC-promoted; no kiln firing. Lime calcination only carbon source. Non-structural partitions. EC: 110 kgCO₂e/m³.",
    "CLC Lightweight Blocks":
        "IS 6598:1977. Foam-aerated; density 400–800 kg/m³. 73% EC reduction. Non-load-bearing partitions. EC: 72 kgCO₂e/m³.",
    "Porotherm Clay Blocks":
        "Wienerberger perforated clay per IS 3952:1988. Better thermal mass than solid bricks. EC: 140 kgCO₂e/m³.",
    "Stabilized Earth Blocks CSEB":
        "IS 1725:2013; 5–8% cement stabilisation (2.5–5.0 N/mm²). 84% EC reduction. Ground floor partitions. EC: 42 kgCO₂e/m³.",
    "Sintered Fly Ash Blocks":
        "High-temp sintered per IS 12894:2002. Superior strength and durability. 78% EC reduction. Exposed external masonry. EC: 60 kgCO₂e/m³.",
    "Hempcrete (Non-load-bearing)":
        "Hemp hurds + lime binder. Carbon-negative lifecycle. NOT structurally load-bearing. No IS code. Not commercially produced in India (2024). EC: 12 kgCO₂e/m³.",
    # FLOORING
    "IPS Cement Flooring":
        "IS 2114:1984; 40mm cement-sand-grit topping. Standard institutional baseline. Low cost. EC: 36 kgCO₂e/m².",
    "Ceramic Floor Tiles IS 13755":
        "IS 13755:1993; kiln-fired at 1,200°C. Service life 25–50 years. Public areas/circulation. EC: 26 kgCO₂e/m².",
    "Vitrified Tiles IS 15622":
        "IS 15622:2006; water absorption ≤0.5%. High abrasion/chemical resistance. Labs/corridors. EC: 22 kgCO₂e/m².",
    "Polished Concrete Screed":
        "In-situ concrete, mechanically ground and sealed. No secondary finish layer. Basement/utility/plant rooms. EC: 18 kgCO₂e/m².",
    "Kota Stone (Natural, Unpolished)":
        "Fine-grained quartzite limestone, Kota Rajasthan. IS 1121. Minimal processing. Transport ~1,500km to Kerala. EC: 8 kgCO₂e/m².",
    "Terrazzo (In-Situ, Cement-Chip)":
        "IS 2571:1970; cement matrix with marble chip. Service life >50 years. Reception/public corridors. EC: 24 kgCO₂e/m².",
    "Polished Granite (Indian Black)":
        "Indian black granite (Karimnagar/Hosur) per IS 1121, IS 3316:1974. Lower transport EC than imports. EC: 14 kgCO₂e/m².",
    "Marble Flooring (Imported)":
        "Rajasthan marble per IS 1121. ~1,200km transport. Premium finish; not recommended as primary sustainable spec. EC: 28 kgCO₂e/m².",
    "Bamboo Flooring (Engineered)":
        "IS 6874:2008. Biogenic carbon; rapidly renewable (3–5yr harvest). Offices/conference rooms. EC: 7 kgCO₂e/m².",
    "Recycled Rubber Tiles":
        "Post-consumer tyre rubber; ASTM F1816. Waste stream diversion. Acoustic attenuation. Labs/workshops. EC: 12 kgCO₂e/m².",
    "Linoleum (Natural, Linseed Oil)":
        "Bio-based; linseed oil + cork + jute backing; EN 548. Biodegradable. Seminar rooms/offices. EC: 9 kgCO₂e/m².",
    "Cork Flooring (Natural)":
        "Amorim EPD 2022. Bark harvested without felling; tree continues sequestering. Lowest EC flooring. EC: 4 kgCO₂e/m².",
}

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_database():
    try:
        df = pd.read_excel("material_database.xlsx")
        df.columns = df.columns.str.strip().str.lower()
        required = ['material', 'category', 'cost', 'ec']
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns in Excel: {missing}")
            return None
        return df
    except Exception as e:
        st.error(f"Database load error: {e}")
        return None

def get_category_df(df, cat):
    return df[df['category'].str.lower().str.strip() == cat].reset_index(drop=True)

# ══════════════════════════════════════════════════════════════════════════════
# NSGA-II IMPLEMENTATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate(individual, cats, qs):
    """Compute (cost, EC) for a given individual (list of material row indices)."""
    cost, ec = 0.0, 0.0
    for i, (cat, q) in enumerate(zip(cats, qs)):
        row = cats[i].iloc[individual[i]]
        cost += row['cost'] * q
        ec   += row['ec']   * q
    return cost, ec

def dominates(a, b):
    """True if solution a dominates solution b (both objectives minimised)."""
    return (a[0] <= b[0] and a[1] <= b[1]) and (a[0] < b[0] or a[1] < b[1])

def fast_non_dominated_sort(pop_obj):
    n = len(pop_obj)
    S = [[] for _ in range(n)]
    rank = [0] * n
    n_dom = [0] * n
    fronts = [[]]
    for p in range(n):
        for q in range(n):
            if p == q: continue
            if dominates(pop_obj[p], pop_obj[q]):
                S[p].append(q)
            elif dominates(pop_obj[q], pop_obj[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            rank[p] = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    rank[q] = i + 1
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return fronts[:-1], rank

def crowding_distance(front_indices, pop_obj):
    n = len(front_indices)
    if n == 0: return []
    dist = [0.0] * n
    for m in range(2):  # 2 objectives
        vals = [pop_obj[i][m] for i in front_indices]
        order = sorted(range(n), key=lambda x: vals[x])
        dist[order[0]] = dist[order[-1]] = float('inf')
        rng = vals[order[-1]] - vals[order[0]]
        if rng == 0: continue
        for k in range(1, n - 1):
            dist[order[k]] += (vals[order[k+1]] - vals[order[k-1]]) / rng
    return dist

def tournament_select(pop, pop_obj, fronts, rank, cd):
    """Binary tournament selection."""
    i, j = random.sample(range(len(pop)), 2)
    if rank[i] < rank[j]: return deepcopy(pop[i])
    if rank[j] < rank[i]: return deepcopy(pop[j])
    return deepcopy(pop[i] if cd[i] >= cd[j] else pop[j])

def crossover(p1, p2):
    """Uniform crossover."""
    return [p1[k] if random.random() < 0.5 else p2[k] for k in range(len(p1))]

def mutate(ind, cat_sizes, rate=0.15):
    return [random.randrange(cat_sizes[k]) if random.random() < rate else ind[k]
            for k in range(len(ind))]

def run_nsga2(cat_dfs, qs, pop_size=200, generations=300, seed=42,
              progress_bar=None, status_text=None):
    random.seed(seed)
    np.random.seed(seed)
    cat_sizes = [len(df) for df in cat_dfs]
    n_cats = len(cat_dfs)

    # Initial population
    pop = [[random.randrange(s) for s in cat_sizes] for _ in range(pop_size)]
    pop_obj = [evaluate(ind, cat_dfs, qs) for ind in pop]

    for gen in range(generations):
        if progress_bar:
            progress_bar.progress((gen + 1) / generations)
        if status_text:
            status_text.text(f"NSGA-II: Generation {gen+1}/{generations} | "
                             f"Population: {pop_size}")

        fronts, rank = fast_non_dominated_sort(pop_obj)

        # Crowding distance per individual
        cd = [0.0] * len(pop)
        for front in fronts:
            dists = crowding_distance(front, pop_obj)
            for idx, fi in enumerate(front):
                cd[fi] = dists[idx]

        # Generate offspring
        offspring = []
        while len(offspring) < pop_size:
            p1 = tournament_select(pop, pop_obj, fronts, rank, cd)
            p2 = tournament_select(pop, pop_obj, fronts, rank, cd)
            child = crossover(p1, p2)
            child = mutate(child, cat_sizes)
            offspring.append(child)

        off_obj = [evaluate(ind, cat_dfs, qs) for ind in offspring]

        # Combine parent + offspring
        combined = pop + offspring
        combined_obj = pop_obj + off_obj
        c_fronts, c_rank = fast_non_dominated_sort(combined_obj)

        # Select next generation
        new_pop, new_obj = [], []
        for front in c_fronts:
            if len(new_pop) + len(front) <= pop_size:
                for fi in front:
                    new_pop.append(combined[fi])
                    new_obj.append(combined_obj[fi])
            else:
                needed = pop_size - len(new_pop)
                dists = crowding_distance(front, combined_obj)
                sorted_front = sorted(zip(dists, front), reverse=True)
                for _, fi in sorted_front[:needed]:
                    new_pop.append(combined[fi])
                    new_obj.append(combined_obj[fi])
                break

        pop, pop_obj = new_pop, new_obj

    # Extract Pareto front from final population
    final_fronts, final_rank = fast_non_dominated_sort(pop_obj)
    pareto_indices = final_fronts[0]
    pareto_pop = [pop[i] for i in pareto_indices]
    pareto_obj = [pop_obj[i] for i in pareto_indices]
    all_obj = pop_obj
    all_pop = pop

    return pareto_pop, pareto_obj, all_pop, all_obj

def individual_to_row(ind, cat_dfs, cat_names):
    row = {}
    for i, (df, name) in enumerate(zip(cat_dfs, cat_names)):
        row[name] = df.iloc[ind[i]]['material']
    return row

# ── WSM on Pareto front ───────────────────────────────────────────────────────
def wsm_best_from_pareto(pareto_pop, pareto_obj, cat_dfs, cat_names, wc, wk):
    costs = [o[0] for o in pareto_obj]
    ecs   = [o[1] for o in pareto_obj]
    c_min, c_max = min(costs), max(costs)
    e_min, e_max = min(ecs),   max(ecs)
    best_z, best_idx = float('inf'), 0
    for i, (c, e) in enumerate(pareto_obj):
        cn = (c - c_min) / (c_max - c_min + 1e-9)
        en = (e - e_min) / (e_max - e_min + 1e-9)
        z  = wc * cn + wk * en
        if z < best_z:
            best_z, best_idx = z, i
    row = individual_to_row(pareto_pop[best_idx], cat_dfs, cat_names)
    row['Cost'] = pareto_obj[best_idx][0]
    row['EC']   = pareto_obj[best_idx][1]
    row['Z']    = round(best_z, 4)
    return row

# ── TOPSIS on full population ─────────────────────────────────────────────────
def run_topsis(all_pop, all_obj, cat_dfs, cat_names, wc, wk):
    costs = np.array([o[0] for o in all_obj])
    ecs   = np.array([o[1] for o in all_obj])
    cn = costs / (np.sqrt((costs**2).sum()) + 1e-9)
    en = ecs   / (np.sqrt((ecs**2).sum())   + 1e-9)
    cw = wc * cn;  ew = wk * en
    d_pos = np.sqrt((cw - cw.min())**2 + (ew - ew.min())**2)
    d_neg = np.sqrt((cw - cw.max())**2 + (ew - ew.max())**2)
    ci    = d_neg / (d_pos + d_neg + 1e-9)
    best_idx = int(np.argmax(ci))
    row = individual_to_row(all_pop[best_idx], cat_dfs, cat_names)
    row['Cost'] = all_obj[best_idx][0]
    row['EC']   = all_obj[best_idx][1]
    row['Ci']   = round(float(ci[best_idx]), 4)
    return row

# ── VIKOR on full population ──────────────────────────────────────────────────
def run_vikor(all_pop, all_obj, cat_dfs, cat_names, wc, wk, v=0.5):
    """
    VIKOR (Opricovic, 1998).
    Q = v·(S-S*)/(S⁻-S*) + (1-v)·(R-R*)/(R⁻-R*)
    S = weighted sum of normalised gaps; R = max weighted gap.
    Lower Q = better compromise solution.
    """
    costs = np.array([o[0] for o in all_obj])
    ecs   = np.array([o[1] for o in all_obj])
    # Best and worst
    f_best = [costs.min(), ecs.min()]
    f_worst= [costs.max(), ecs.max()]
    weights = [wc, wk]
    criteria = [costs, ecs]
    # Utility (S) and Regret (R)
    S = np.zeros(len(all_obj))
    R = np.zeros(len(all_obj))
    for j, (fj, w) in enumerate(zip(criteria, weights)):
        gap = (f_best[j] - fj) / (f_best[j] - f_worst[j] + 1e-9)
        term = w * np.abs(gap)  # both criteria are cost-type (lower is better)
        # Correct formulation for cost-type: (fj - f_best) / (f_worst - f_best)
        gap2 = (fj - f_best[j]) / (f_worst[j] - f_best[j] + 1e-9)
        term2 = w * gap2
        S += term2
        R = np.maximum(R, term2)
    S_star, S_neg = S.min(), S.max()
    R_star, R_neg = R.min(), R.max()
    Q = v * (S - S_star) / (S_neg - S_star + 1e-9) + \
        (1 - v) * (R - R_star) / (R_neg - R_star + 1e-9)
    best_idx = int(np.argmin(Q))
    row = individual_to_row(all_pop[best_idx], cat_dfs, cat_names)
    row['Cost'] = all_obj[best_idx][0]
    row['EC']   = all_obj[best_idx][1]
    row['Q']    = round(float(Q[best_idx]), 4)
    return row

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
df = load_database()

if df is not None:
    CATS = ['structural_concrete', 'slab_concrete', 'wall_concrete',
            'wall_screed', 'masonry', 'flooring']
    CAT_LABELS = {
        'structural_concrete': '🏗️ Struct. Concrete (Beams+Cols)',
        'slab_concrete':       '🟦 Slab Concrete (Floors)',
        'wall_concrete':       '🧱 Wall Concrete (Struct. Walls)',
        'wall_screed':         '🪣 Wall Screed',
        'masonry':             '🧱 Masonry (Infill/Partition)',
        'flooring':            '📐 Floor Finishing',
    }
    cat_dfs = [get_category_df(df, c) for c in CATS]
    qs_list = [Q[c] for c in CATS]

    st.sidebar.title("📋 Project Specifications")
    st.sidebar.markdown("**Project:** Translational Research Centre (TRC)")
    st.sidebar.markdown("**Institution:** APJ Abdul Kalam Technological University")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🏗️ BIM-Extracted Quantities (Dynamo)")
    st.sidebar.caption("Revit model → Dynamo script → Quantity extraction")
    for c in CATS:
        unit = "m²" if c == 'flooring' else "m³"
        st.sidebar.markdown(f"- **{CAT_LABELS[c]}:** {Q[c]} {unit}")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 📊 Phase 1 Baseline")
    st.sidebar.markdown(f"- **Cost:** {fmt_inr(BASELINE_COST)}")
    st.sidebar.markdown(f"- **Carbon:** {fmt_ec(BASELINE_CARBON)}")
    st.sidebar.caption("A1–A3 per EN 15978 | Kerala PWD DSR 2023-24 + One Click LCA")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### ⚙️ NSGA-II Parameters")
    pop_size   = st.sidebar.slider("Population size", 100, 400, 200, 50)
    n_gens     = st.sidebar.slider("Generations", 100, 500, 300, 50)
    rand_seed  = st.sidebar.number_input("Random seed", 0, 9999, 42)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🔧 Custom Quantities")
    use_manual = st.sidebar.toggle("Enable custom quantities", value=False)
    if use_manual:
        for c in CATS:
            unit = "m²" if c == 'flooring' else "m³"
            Q[c] = st.sidebar.number_input(
                f"{CAT_LABELS[c]} ({unit})", min_value=1.0, value=Q[c], step=10.0)
        qs_list = [Q[c] for c in CATS]

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN PAGE
    # ══════════════════════════════════════════════════════════════════════════
    st.title("🏛️ Material Optimization Decision Support System")
    st.markdown("##### BIM-Integrated Multi-Objective Material Optimization | Translational Research Centre")
    st.markdown("*NSGA-II + WSM + TOPSIS + VIKOR*")
    st.markdown("---")

    # ── I. Weight Calibration ─────────────────────────────────────────────────
    st.subheader("I. Optimization Weight Calibration")
    st.markdown(
        '<p class="section-desc">'
        "Two decision criteria are optimized simultaneously: total project cost (₹) and total "
        "embodied carbon (kgCO₂e, A1–A3 per EN 15978). NSGA-II generates the full Pareto frontier "
        "without weight input. Post-hoc WSM, TOPSIS, and VIKOR then select the preferred solution "
        "from the Pareto front using the weight w<sub>k</sub>. Setting w<sub>k</sub> = 0 selects "
        "the minimum-cost solution; w<sub>k</sub> = 1 selects the minimum-carbon solution."
        "</p>", unsafe_allow_html=True)

    col_sl, col_info = st.columns([3, 1.2])
    with col_sl:
        wk = st.select_slider(
            "🌿 Environmental Priority Weight (wₖ):",
            options=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            value=0.6)
        wc = round(1.0 - wk, 2)
    with col_info:
        st.markdown(f"💰 **Cost weight (wc):** `{wc}`")
        st.markdown(f"🌿 **Carbon weight (wk):** `{wk}`")

    # ── II. NSGA-II Optimization ──────────────────────────────────────────────
    st.markdown("---")
    st.subheader("II. NSGA-II Multi-Objective Optimization")
    st.markdown(
        '<p class="section-desc">'
        "NSGA-II (Deb et al., 2002 — IEEE Transactions on Evolutionary Computation) is a "
        "fast elitist multi-objective genetic algorithm. It simultaneously minimises both "
        "cost and embodied carbon across all 6 element-level material categories without "
        "requiring weight pre-specification. The algorithm uses non-dominated sorting and "
        "crowding distance to maintain a diverse Pareto-optimal population. The resulting "
        "Pareto front represents all solutions for which no alternative achieves lower cost "
        "AND lower embodied carbon simultaneously."
        "</p>", unsafe_allow_html=True)

    run_col, _ = st.columns([1, 3])
    with run_col:
        run_btn = st.button("▶ Run NSGA-II Optimization", type="primary",
                            use_container_width=True)

    if 'nsga2_done' not in st.session_state:
        st.session_state.nsga2_done = False

    if run_btn:
        st.session_state.nsga2_done = False
        prog = st.progress(0)
        stat = st.empty()
        with st.spinner("Running NSGA-II..."):
            p_pop, p_obj, a_pop, a_obj = run_nsga2(
                cat_dfs, qs_list, pop_size, n_gens, int(rand_seed), prog, stat)
        prog.empty(); stat.empty()
        st.session_state.pareto_pop = p_pop
        st.session_state.pareto_obj = p_obj
        st.session_state.all_pop    = a_pop
        st.session_state.all_obj    = a_obj
        st.session_state.nsga2_done = True
        st.success(f"✅ NSGA-II complete — Pareto front: {len(p_pop)} solutions | "
                   f"Population: {len(a_pop)} | Generations: {n_gens}")

    if not st.session_state.nsga2_done:
        st.info("⬆ Press **Run NSGA-II Optimization** to generate the Pareto front.")
        st.stop()

    p_pop  = st.session_state.pareto_pop
    p_obj  = st.session_state.pareto_obj
    a_pop  = st.session_state.all_pop
    a_obj  = st.session_state.all_obj

    # ── III. WSM Best from Pareto ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("III. Optimal Solution — WSM Selection from Pareto Front")
    st.markdown(
        '<p class="section-desc">'
        "The Weighted Sum Model (Fishburn, 1967) is applied post-hoc to the NSGA-II Pareto front "
        "to select the preferred solution given the current weight setting. "
        "This two-stage approach (NSGA-II generates the front; WSM selects from it) is "
        "the standard methodology in construction optimisation research "
        "(Eleftheriadis et al., 2017 — Energy and Buildings)."
        "</p>", unsafe_allow_html=True)

    best = wsm_best_from_pareto(p_pop, p_obj, cat_dfs, CATS, wc, wk)

    m_cols = st.columns(3)
    m_cols[0].metric("🏗️ Struct. Concrete (Beams+Cols)", best['structural_concrete'])
    m_cols[1].metric("🟦 Slab Concrete (Floors)",        best['slab_concrete'])
    m_cols[2].metric("🧱 Wall Concrete (Struct. Walls)", best['wall_concrete'])
    m_cols2 = st.columns(3)
    m_cols2[0].metric("🪣 Wall Screed",                  best['wall_screed'])
    m_cols2[1].metric("🧱 Masonry (Infill/Partition)",   best['masonry'])
    m_cols2[2].metric("📐 Floor Finishing",              best['flooring'])

    r_cols = st.columns(3)
    r_cols[0].metric("📊 WSM Z-score", f"{best['Z']}")
    r_cols[1].metric("💰 Estimated Total Cost", fmt_inr(best['Cost']))
    r_cols[2].metric("🌿 Total Embodied Carbon", fmt_ec(best['EC']))

    st.markdown("##### 📉 Comparison with Phase 1 Baseline")
    cost_chg   = (best['Cost'] - BASELINE_COST)   / BASELINE_COST   * 100
    carbon_chg = (best['EC']   - BASELINE_CARBON) / BASELINE_CARBON * 100
    b_cols = st.columns(4)
    b_cols[0].metric("Baseline Cost",    fmt_inr(BASELINE_COST))
    b_cols[1].metric("Optimised Cost",   fmt_inr(best['Cost']),
                     delta=f"{cost_chg:+.1f}%",   delta_color="inverse")
    b_cols[2].metric("Baseline Carbon",  fmt_ec(BASELINE_CARBON))
    b_cols[3].metric("Optimised Carbon", fmt_ec(best['EC']),
                     delta=f"{carbon_chg:+.1f}%", delta_color="inverse")

    # ── IV. Material Technical Specifications ──────────────────────────────────
    st.markdown("---")
    st.subheader("IV. Material Technical Specifications — Optimal Solution")
    cat_display = {
        'structural_concrete': ('🏗️ Structural Concrete — Beams & Columns', f"{Q['structural_concrete']} m³"),
        'slab_concrete':       ('🟦 Slab Concrete — Floor Slabs',           f"{Q['slab_concrete']} m³"),
        'wall_concrete':       ('🧱 Wall Concrete — Structural Walls',       f"{Q['wall_concrete']} m³"),
        'wall_screed':         ('🪣 Wall Screed',                            f"{Q['wall_screed']} m³"),
        'masonry':             ('🧱 Masonry — Infill/Partition Walls',       f"{Q['masonry']} m³"),
        'flooring':            ('📐 Floor Finishing',                        f"{Q['flooring']} m²"),
    }
    for cat, (label, qty) in cat_display.items():
        mat = best[cat]
        insight = material_insights.get(mat, f"No technical data for '{mat}'.")
        st.info(f"**{label}** | Quantity: {qty}\n\n**{mat}** — {insight}")

    # ── V. Pareto Frontier ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("V. NSGA-II Pareto Efficiency Frontier")
    st.markdown(
        '<p class="section-desc">'
        f"All {len(a_obj):,} final-generation solutions are plotted in cost–embodied carbon space. "
        "The <strong>red Pareto frontier</strong> contains solutions for which no alternative "
        "simultaneously achieves lower cost and lower embodied carbon. Each point on the frontier "
        "represents a feasible material combination; movement along the frontier quantifies the "
        "marginal cost of embodied carbon reduction. The WSM-selected optimal solution is "
        "highlighted in gold."
        "</p>", unsafe_allow_html=True)

    all_costs = [o[0] for o in a_obj]
    all_ecs   = [o[1] for o in a_obj]
    par_costs = [o[0] for o in p_obj]
    par_ecs   = [o[1] for o in p_obj]
    par_sorted = sorted(zip(par_costs, par_ecs))
    par_c_s, par_e_s = zip(*par_sorted)

    fig_p = go.Figure()
    fig_p.add_trace(go.Scatter(
        x=all_costs, y=all_ecs, mode='markers', name='All solutions',
        marker=dict(color='#4dabf7', size=4, opacity=0.4)))
    fig_p.add_trace(go.Scatter(
        x=par_c_s, y=par_e_s, mode='lines+markers', name='Pareto Frontier',
        line=dict(color='#ff4444', width=3),
        marker=dict(size=7, color='#ff4444')))
    fig_p.add_trace(go.Scatter(
        x=[best['Cost']], y=[best['EC']], mode='markers', name='WSM Optimal',
        marker=dict(color='gold', size=14, symbol='star',
                    line=dict(color='white', width=1))))
    fig_p.update_layout(
        template='plotly_dark',
        xaxis_title='Total Project Cost (₹)',
        yaxis_title='Total Embodied Carbon (kgCO₂e)',
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0.4)'))
    st.plotly_chart(fig_p, use_container_width=True)

    # ── VI. Sensitivity Analysis ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("VI. Sensitivity Analysis — WSM on Pareto Front")
    st.markdown(
        '<p class="section-desc">'
        "Sensitivity analysis applies WSM post-selection across all weight values using the "
        "same NSGA-II Pareto front, identifying stable selection zones where the optimal "
        "material combination remains unchanged across consecutive weight steps."
        "</p>", unsafe_allow_html=True)

    sens_rows = []
    for w_env in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        w_cost = round(1.0 - w_env, 2)
        bs = wsm_best_from_pareto(p_pop, p_obj, cat_dfs, CATS, w_cost, w_env)
        zone = ("Cost-driven" if w_env < 0.4 else
                ("Balanced" if w_env <= 0.6 else "Carbon-driven"))
        sens_rows.append({
            "wk": w_env, "Zone": zone,
            "Struct. Concrete": bs['structural_concrete'],
            "Slab Concrete":    bs['slab_concrete'],
            "Wall Concrete":    bs['wall_concrete'],
            "Wall Screed":      bs['wall_screed'],
            "Masonry":          bs['masonry'],
            "Flooring":         bs['flooring'],
            "Z-score":          bs['Z'],
            "Total Cost":       fmt_inr(bs['Cost']),
            "Total Carbon (kgCO₂e)": f"{int(round(bs['EC'])):,}",
        })
    sens_df = pd.DataFrame(sens_rows)
    st.dataframe(sens_df, use_container_width=True)

    # Dual-axis trade-off
    wk_v = [r['wk'] for r in sens_rows]
    cost_v = []
    ec_v   = []
    for w_env in wk_v:
        w_cost = round(1.0 - w_env, 2)
        bs2 = wsm_best_from_pareto(p_pop, p_obj, cat_dfs, CATS, w_cost, w_env)
        cost_v.append(bs2['Cost']); ec_v.append(bs2['EC'])

    fig_s = go.Figure()
    fig_s.add_trace(go.Scatter(x=wk_v, y=cost_v, name="Total Cost (₹)",
        mode="lines+markers", line=dict(color="#00d4ff", width=2.5),
        marker=dict(size=8), yaxis="y1"))
    fig_s.add_trace(go.Scatter(x=wk_v, y=ec_v, name="Total Carbon (kgCO₂e)",
        mode="lines+markers", line=dict(color="#ff6b6b", width=2.5),
        marker=dict(size=8), yaxis="y2"))
    fig_s.update_layout(
        template="plotly_dark",
        title="Cost–Carbon Trade-off vs. wₖ (WSM on NSGA-II Pareto Front)",
        xaxis=dict(title="wₖ (0 = cost priority → 1 = carbon priority)", tickvals=wk_v),
        yaxis=dict(title=dict(text="Total Cost (₹)", font=dict(color="#00d4ff", size=12))),
        yaxis2=dict(title=dict(text="Total Embodied Carbon (kgCO₂e)",
                               font=dict(color="#ff6b6b", size=12)),
                    overlaying="y", side="right"),
        legend=dict(x=0.3, y=1.12, orientation="h"), margin=dict(t=80))
    st.plotly_chart(fig_s, use_container_width=True)

    # ── VII. Carbon Breakdown ──────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("VII. Embodied Carbon Contribution by Element Category")
    st.markdown(
        '<p class="section-desc">'
        "Decomposition of total embodied carbon by all six element categories in the "
        "optimal solution, quantifying the relative environmental impact of each building "
        "sub-system. This element-level resolution enables targeted intervention — "
        "categories with the highest share are priority candidates for further decarbonisation."
        "</p>", unsafe_allow_html=True)

    ec_by_cat, cost_by_cat, label_list = [], [], []
    for cat, q in zip(CATS, qs_list):
        mat_name = best[cat]
        cat_df   = get_category_df(df, cat)
        row      = cat_df[cat_df['material'] == mat_name]
        if not row.empty:
            ec_by_cat.append(float(row.iloc[0]['ec']) * q)
            cost_by_cat.append(float(row.iloc[0]['cost']) * q)
        else:
            ec_by_cat.append(0); cost_by_cat.append(0)
        label_list.append(CAT_LABELS[cat].split(' ')[1])

    total_ec_bd = sum(ec_by_cat)
    bd_df = pd.DataFrame({
        "Element Category":          label_list,
        "Embodied Carbon (kgCO₂e)":  [round(v, 0) for v in ec_by_cat],
        "Share (%)":                 [round(v / total_ec_bd * 100, 1) for v in ec_by_cat],
        "Cost (₹)":                  [round(v, 0) for v in cost_by_cat],
    })
    col_c, col_t = st.columns([2, 1])
    with col_c:
        fig_bar = px.bar(bd_df, x="Element Category", y="Embodied Carbon (kgCO₂e)",
            color="Element Category", template="plotly_dark",
            text=bd_df["Share (%)"].apply(lambda x: f"{x}%"),
            title="Embodied Carbon by Element Category — Optimal Solution")
        fig_bar.update_traces(textposition='outside')
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_t:
        st.markdown("**Element Breakdown**")
        st.dataframe(bd_df, use_container_width=True)

    # ── VIII. TOPSIS Cross-Validation ──────────────────────────────────────────
    st.markdown("---")
    st.subheader("VIII. TOPSIS Cross-Validation")
    st.markdown(
        '<p class="section-desc">'
        "TOPSIS (Hwang &amp; Yoon, 1981) is applied across the full NSGA-II final population "
        "as an independent cross-validation. Each solution is ranked by its closeness coefficient "
        "Cᵢ = d⁻ / (d⁺ + d⁻), where d⁺ and d⁻ are Euclidean distances from the positive and "
        "negative ideal solutions. Cᵢ = 1 represents the ideal. Convergence of WSM and TOPSIS "
        "provides method-level validation of the result."
        "</p>", unsafe_allow_html=True)

    best_t = run_topsis(a_pop, a_obj, cat_dfs, CATS, wc, wk)
    t_cols = st.columns(3)
    t_cols[0].metric("🏗️ Struct. Concrete (TOPSIS)", best_t['structural_concrete'])
    t_cols[1].metric("🟦 Slab Concrete (TOPSIS)",    best_t['slab_concrete'])
    t_cols[2].metric("🧱 Wall Concrete (TOPSIS)",    best_t['wall_concrete'])
    t_cols2 = st.columns(3)
    t_cols2[0].metric("🪣 Wall Screed (TOPSIS)",     best_t['wall_screed'])
    t_cols2[1].metric("🧱 Masonry (TOPSIS)",         best_t['masonry'])
    t_cols2[2].metric("📐 Flooring (TOPSIS)",        best_t['flooring'])
    tr = st.columns(3)
    tr[0].metric("📊 TOPSIS Closeness Cᵢ", f"{best_t['Ci']:.4f}")
    tr[1].metric("💰 TOPSIS Cost", fmt_inr(best_t['Cost']))
    tr[2].metric("🌿 TOPSIS Carbon", fmt_ec(best_t['EC']))

    # ── IX. VIKOR Cross-Validation ─────────────────────────────────────────────
    st.markdown("---")
    st.subheader("IX. VIKOR Cross-Validation")
    st.markdown(
        '<p class="section-desc">'
        "VIKOR (Opricovic, 1998; Opricovic &amp; Tzeng, 2004 — European Journal of Operational "
        "Research) provides a compromise ranking optimised for the concept of maximum group utility "
        "and minimum individual regret. The VIKOR index Q combines a utility measure S (weighted "
        "sum of normalised gaps from the positive ideal) and a regret measure R (maximum weighted "
        "gap). The parameter v = 0.5 represents balanced group utility and regret. VIKOR is "
        "specifically designed for situations with conflicting criteria — directly applicable "
        "to cost–carbon trade-offs in sustainable construction. Lower Q = better compromise."
        "</p>", unsafe_allow_html=True)

    best_v = run_vikor(a_pop, a_obj, cat_dfs, CATS, wc, wk, v=0.5)
    v_cols = st.columns(3)
    v_cols[0].metric("🏗️ Struct. Concrete (VIKOR)", best_v['structural_concrete'])
    v_cols[1].metric("🟦 Slab Concrete (VIKOR)",    best_v['slab_concrete'])
    v_cols[2].metric("🧱 Wall Concrete (VIKOR)",    best_v['wall_concrete'])
    v_cols2 = st.columns(3)
    v_cols2[0].metric("🪣 Wall Screed (VIKOR)",     best_v['wall_screed'])
    v_cols2[1].metric("🧱 Masonry (VIKOR)",         best_v['masonry'])
    v_cols2[2].metric("📐 Flooring (VIKOR)",        best_v['flooring'])
    vr = st.columns(3)
    vr[0].metric("📊 VIKOR Q-index", f"{best_v['Q']:.4f}")
    vr[1].metric("💰 VIKOR Cost", fmt_inr(best_v['Cost']))
    vr[2].metric("🌿 VIKOR Carbon", fmt_ec(best_v['EC']))

    # ── X. Method Convergence Summary ─────────────────────────────────────────
    st.markdown("---")
    st.subheader("X. Multi-Method Convergence Summary")
    st.markdown(
        '<p class="section-desc">'
        "Convergence across three independent MCDM methods (WSM, TOPSIS, VIKOR) applied to "
        "the NSGA-II Pareto front constitutes method-level validation. "
        "Where all three agree, the recommendation is robust to the choice of decision model. "
        "Divergence identifies the sensitivity boundary and is analytically reported per "
        "Opricovic &amp; Tzeng (2004)."
        "</p>", unsafe_allow_html=True)

    cats_check = CATS
    conv_rows = []
    for cat in cats_check:
        wsm_m = best[cat]
        top_m = best_t[cat]
        vik_m = best_v[cat]
        agree = "✅ All agree" if wsm_m == top_m == vik_m else (
                "⚠️ Partial" if wsm_m == top_m or wsm_m == vik_m or top_m == vik_m
                else "❌ All differ")
        conv_rows.append({
            "Element":    CAT_LABELS[cat],
            "WSM":        wsm_m,
            "TOPSIS":     top_m,
            "VIKOR":      vik_m,
            "Convergence": agree,
        })
    conv_df = pd.DataFrame(conv_rows)
    st.dataframe(conv_df, use_container_width=True)

    all_agree = all(r['Convergence'] == '✅ All agree' for r in conv_rows)
    any_differ = any(r['Convergence'] == '❌ All differ' for r in conv_rows)
    if all_agree:
        st.success(
            f"✅ **Full Method Convergence** — WSM, TOPSIS, and VIKOR select identical materials "
            f"across all 6 element categories at wk = {wk}. "
            f"Result is independent of the MCDM technique selected.")
    elif not any_differ:
        st.warning(
            f"⚠️ **Partial Convergence at wk = {wk}** — At least two methods agree on all categories. "
            "Divergence is analytically expected between WSM (linear aggregation) and "
            "TOPSIS/VIKOR (geometric/compromise methods) at intermediate weight settings. "
            "Adjust wk to identify the full convergence zone.")
    else:
        st.error(
            f"❌ **Methods Diverge at wk = {wk}** — This identifies the weight-sensitivity boundary. "
            "See Opricovic & Tzeng (2004) for the theoretical basis of WSM–TOPSIS–VIKOR divergence.")

    # ── XI. Data Sources & Scope ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("XI. Data Sources, IS Code References & Scope Boundary")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📗 Embodied Carbon Sources**")
        st.markdown(
            "- ICE Database v3.0 — Hammond & Jones, Univ. of Bath (2019)\n"
            "- ECOINVENT 3.8 — Swiss Centre for Life Cycle Inventories\n"
            "- Manufacturer EPDs: Wienerberger (2022), Amorim (2022), Kajaria (2022)\n"
            "- TERI India Material Carbon Profiles (2020)\n"
            "- CarbonCure Technology EPD (2023)\n")
        st.markdown("**🌍 Carbon Scope**")
        st.markdown(
            "A1–A3 per EN 15978: raw material extraction (A1), transport to factory (A2), "
            "manufacturing (A3). Stages A4–A5, B, C excluded.")
    with col_b:
        st.markdown("**📘 Cost Sources**")
        st.markdown(
            "- Kerala PWD DSR 2023-24\n"
            "- Kerala market survey: AAC, fly ash, stone (2023–24)\n")
        st.markdown("**📋 IS Codes Referenced**")
        st.markdown(
            "- IS 456:2000 — Plain & Reinforced Concrete (M20/M25/M30)\n"
            "- IS 16714:2018 — GGBS for Concrete\n"
            "- IS 1489 Pt1:1991 — Portland Pozzolana Cement\n"
            "- IS 2185 Pts 1 & 3 — Concrete Masonry Units\n"
            "- IS 12894:2002 — Fly Ash Bricks\n"
            "- IS 1077:1992 — Burnt Clay Bricks\n"
            "- IS 1725:2013 — Stabilised Soil Blocks\n"
            "- IS 15622:2006 — Vitrified Tiles\n"
            "- IS 2114:1984 — IPS Flooring\n"
            "- IS 2571:1970 — In-Situ Terrazzo\n"
            "- IS 6874:2008 — Bamboo Products\n"
            "- IS 4926:2003 — Ready-Mixed Concrete\n"
            "- IS 10262:2019 — Concrete Mix Design\n"
            "- IS 2116:1980 — Sand for Masonry Mortars\n"
            "- IS 712:1984 — Building Limes\n")

    with st.expander("⚠️ Scope Limitations and Methodological Boundaries"):
        st.markdown("""
**1. Element-Level Material Aggregation**
Structural concrete is differentiated by element type: one specification for beams+columns
(minimum M25 per IS 456:2000 Cl.6.1.2), a separate specification for floor slabs (minimum M20),
and a third for structural walls. Wall screed and masonry are independently optimised.
This represents a significant advance over building-scale category aggregation
(Röck et al., 2020; Pomponi & Moncaster, 2016).

**2. Non-IS-Compliant Materials**
Hempcrete, LC3 Concrete, Geopolymer Concrete, Bio-Concrete, and Carbon-Cured Concrete
have no current IS code equivalent. Inclusion defines the theoretical carbon reduction boundary.

**3. Intra-Element Differentiation**
Beams and columns share one concrete specification in this framework.
Further disaggregation (e.g., individual structural member optimisation) is possible
with element-tagged Dynamo output and is identified as extended scope.

**4. Transport EC**
ICE DB v3.0 values are UK-origin. Transport to TRC site not separately quantified.
Materials sourced distantly (Kota Stone ~1,500 km; marble ~1,200 km) carry
unaccounted transport contributions.

**5. Cost Basis**
Kerala PWD DSR 2023-24: material + labour. Site overhead, contractor profit,
GST (18%), and contingencies excluded.

**6. NSGA-II Convergence**
Population size 200, generations 300 provide good convergence for this problem scale.
Increasing to 400/500 may marginally improve Pareto front density.
        """)