import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="Billing Models Dashboard", layout="wide")

st.title("Billing: Outsourced vs In-House (3-Model Toggle)")
st.caption("Models: Laconia only • Wise Path only • Combined • Includes break-even points")

# =========================
# Helpers
# =========================

def tier_fee(annual_collections: float, tier_widths: list[float], tier_rates: list[float], sales_tax_rate: float) -> float:
    """Wise Path-style tiered percentage on collections (tier_widths are widths; 0 = unlimited)."""
    ac = abs(float(annual_collections))
    remaining = ac
    fee = 0.0

    for width, rate in zip(tier_widths, tier_rates):
        width = float(width)
        rate = float(rate)

        if remaining <= 0:
            break

        if width == 0:
            amt = remaining
        else:
            amt = min(remaining, width)

        amt = max(0.0, amt)
        fee += amt * rate
        remaining = max(0.0, remaining - amt)

    return fee * (1.0 + float(sales_tax_rate))


def laconia_fee(
    annual_paid_claims: float,
    aged_pct: float,
    contracting_hours: float,
    min_monthly: float,
    reg_rate: float,
    aged_rate: float,
    hourly_rate: float,
) -> float:
    """Laconia-style pricing: regular claims % + aged claims % + hourly contracting, subject to minimum monthly."""
    claims = abs(float(annual_paid_claims))
    aged_pct = min(max(float(aged_pct), 0.0), 1.0)

    reg_claims = claims * (1.0 - aged_pct)
    aged_claims = claims * aged_pct

    fee = reg_claims * float(reg_rate) + aged_claims * float(aged_rate) + float(contracting_hours) * float(hourly_rate)
    min_annual = float(min_monthly) * 12.0
    return max(fee, min_annual)


def staffing_cost(df_roles: pd.DataFrame, benefits_rate: float) -> float:
    base = float((df_roles["FTE"] * df_roles["Salary"]).sum())
    return base * (1.0 + float(benefits_rate))


def make_staff_table(title: str, key_prefix: str, defaults: list[tuple[str, float, float]]) -> pd.DataFrame:
    st.subheader(title)

    rows = []
    hdr = st.columns([2, 1, 1])
    hdr[0].write("**Role**")
    hdr[1].write("**FTE**")
    hdr[2].write("**Salary**")

    for role, fte0, sal0 in defaults:
        c1, c2, c3 = st.columns([2, 1, 1])
        c1.write(role)
        fte = c2.number_input(
            f"{role} FTE ({title})",
            min_value=0.0,
            value=float(fte0),
            step=0.5,
            key=f"{key_prefix}_fte_{role}",
        )
        sal = c3.number_input(
            f"{role} Salary ({title})",
            min_value=0.0,
            value=float(sal0),
            step=5000.0,
            key=f"{key_prefix}_sal_{role}",
        )
        rows.append({"Role": role, "FTE": fte, "Salary": sal})

    return pd.DataFrame(rows)


def find_break_even(fn, target, low, high, tol=1.0, max_iter=80):
    """Binary search for x such that fn(x) ~= target. Returns None if no solution in [low, high]."""
    f_low = fn(low) - target
    f_high = fn(high) - target

    if f_low == 0:
        return low
    if f_high == 0:
        return high

    # No sign change => no root in range
    if (f_low < 0 and f_high < 0) or (f_low > 0 and f_high > 0):
        return None

    lo, hi = low, high
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = fn(mid) - target

        if abs(f_mid) <= tol:
            return mid

        # Keep the half-interval that contains the root
        if (f_low < 0 and f_mid < 0) or (f_low > 0 and f_mid > 0):
            lo = mid
            f_low = f_mid
        else:
            hi = mid

    return (lo + hi) / 2.0


def fmt_money(x: float) -> str:
    return f"${x:,.0f}"


# =========================
# Sidebar inputs
# =========================

st.sidebar.header("Global")
benefits_rate = st.sidebar.slider("Benefits & Payroll Overhead (%)", 0.0, 40.0, 20.0, 1.0) / 100.0

st.sidebar.divider()
st.sidebar.header("Laconia (claims-based outsourced)")

lac_paid_claims = st.sidebar.number_input(
    "Laconia – Annual Paid Claims ($)",
    min_value=0.0,
    value=0.0,
    step=50_000.0,
    format="%.2f",
)

lac_aged_pct = st.sidebar.slider("Laconia – % of claims aged", 0.0, 100.0, 15.0, 1.0) / 100.0
lac_contract_hours = st.sidebar.number_input("Laconia – Annual contracting hours", min_value=0.0, value=0.0, step=10.0)

with st.sidebar.expander("Laconia rates (edit)"):
    lac_min_monthly = st.number_input("Minimum monthly invoice ($)", min_value=0.0, value=170.0, step=10.0)
    lac_reg_rate = st.number_input("Claims processing rate (regular)", min_value=0.0, value=0.075, step=0.005, format="%.3f")
    lac_aged_rate = st.number_input("Aged claims rate", min_value=0.0, value=0.10, step=0.01, format="%.3f")
    lac_hourly = st.number_input("Contracting hourly rate ($/hr)", min_value=0.0, value=40.0, step=5.0)

st.sidebar.divider()
st.sidebar.header("Wise Path (tiered % on collections)")

wise_collections = st.sidebar.number_input(
    "Wise Path – Annual Collections ($)",
    min_value=0.0,
    value=710_502.32 * 12,
    step=50_000.0,
    format="%.2f",
)

wise_sales_tax = st.sidebar.number_input("Sales tax on billing fee (%)", min_value=0.0, value=0.9, step=0.1, format="%.2f") / 100.0

st.sidebar.markdown("**Tier widths & rates** (0 width = unlimited)")
default_widths = [250000.0, 250000.0, 500000.0, 1000000.0, 0.0]
default_rates = [0.05, 0.04, 0.035, 0.03, 0.025]

tier_widths: list[float] = []
tier_rates: list[float] = []
for i in range(5):
    st.sidebar.markdown(f"Tier {i+1}")
    w = st.sidebar.number_input(
        f"Width {i+1}",
        min_value=0.0,
        value=float(default_widths[i]),
        step=50_000.0,
        key=f"w_{i}",
    )
    r = st.sidebar.number_input(
        f"Rate {i+1}",
        min_value=0.0,
        value=float(default_rates[i]),
        step=0.001,
        format="%.3f",
        key=f"r_{i}",
    )
    tier_widths.append(w)
    tier_rates.append(r)


# =========================
# Staffing inputs (main page)
# =========================

st.divider()
st.header("In-House Staffing Assumptions (Editable)")

default_roles = [
    ("Billing Manager", 1.0, 80000.0),
    ("Billing Specialist", 2.0, 50000.0),
    ("UR Coordinator", 1.0, 68000.0),
]

left, right = st.columns(2)
with left:
    lac_staff = make_staff_table("Laconia Team", "lac", defaults=default_roles)
with right:
    wise_staff = make_staff_table("Wise Path Team", "wise", defaults=default_roles)


# =========================
# Compute model costs
# =========================

lac_out = laconia_fee(
    annual_paid_claims=lac_paid_claims,
    aged_pct=lac_aged_pct,
    contracting_hours=lac_contract_hours,
    min_monthly=lac_min_monthly,
    reg_rate=lac_reg_rate,
    aged_rate=lac_aged_rate,
    hourly_rate=lac_hourly,
)

wise_out = tier_fee(
    annual_collections=wise_collections,
    tier_widths=tier_widths,
    tier_rates=tier_rates,
    sales_tax_rate=wise_sales_tax,
)

lac_in = staffing_cost(lac_staff, benefits_rate)
wise_in = staffing_cost(wise_staff, benefits_rate)

models = [
    {"Model": "Model 1: Laconia only", "Outsourced": lac_out, "In-House": lac_in},
    {"Model": "Model 2: Wise Path only", "Outsourced": wise_out, "In-House": wise_in},
    {"Model": "Model 3: Combined", "Outsourced": lac_out + wise_out, "In-House": lac_in + wise_in},
]

summary_all = pd.DataFrame(models)
summary_all["Difference (In-House − Outsourced)"] = summary_all["In-House"] - summary_all["Outsourced"]


# =========================
# Break-even calculations
# =========================

def lac_outsourced_given_claims(x: float) -> float:
    return laconia_fee(
        annual_paid_claims=x,
        aged_pct=lac_aged_pct,
        contracting_hours=lac_contract_hours,
        min_monthly=lac_min_monthly,
        reg_rate=lac_reg_rate,
        aged_rate=lac_aged_rate,
        hourly_rate=lac_hourly,
    )


def wise_outsourced_given_collections(x: float) -> float:
    return tier_fee(
        annual_collections=x,
        tier_widths=tier_widths,
        tier_rates=tier_rates,
        sales_tax_rate=wise_sales_tax,
    )


# Wide upper bounds so the search can find a crossing if it exists
lac_hi = max(1.0, lac_paid_claims * 10.0 + 1_000_000.0)
wise_hi = max(1.0, wise_collections * 10.0 + 5_000_000.0)

lac_break_even_claims = find_break_even(fn=lac_outsourced_given_claims, target=lac_in, low=0.0, high=lac_hi)
wise_break_even_collections = find_break_even(fn=wise_outsourced_given_collections, target=wise_in, low=0.0, high=wise_hi)

combined_in = lac_in + wise_in

def combined_outsourced_given_k(k: float) -> float:
    return lac_outsourced_given_claims(lac_paid_claims * k) + wise_outsourced_given_collections(wise_collections * k)

combined_break_even_k = find_break_even(fn=combined_outsourced_given_k, target=combined_in, low=0.0, high=10.0)


# =========================
# Model toggle + KPIs
# =========================

st.divider()
st.header("Model View")

model_choice = st.radio("Select model to view KPIs", summary_all["Model"].tolist(), horizontal=True)
row = summary_all.loc[summary_all["Model"] == model_choice].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("Outsourced (annual)", fmt_money(row["Outsourced"]))
k2.metric("In-House (annual)", fmt_money(row["In-House"]))
k3.metric("Difference (In-House − Outsourced)", fmt_money(row["Difference (In-House − Outsourced)"]))
k4.metric("Benefits rate", f"{benefits_rate*100:.0f}%")

st.subheader("All Models Summary")
st.dataframe(
    summary_all.style.format(
        {
            "Outsourced": "${:,.0f}",
            "In-House": "${:,.0f}",
            "Difference (In-House − Outsourced)": "${:,.0f}",
        }
    ),
    use_container_width=True,
)


# =========================
# Break-even section (management-friendly)
# =========================

st.subheader("Break-even Points (Where In-House = Outsourced)")

be_rows = [
    {
        "Model": "Model 1: Laconia only",
        "Break-even Driver": "Annual Paid Claims ($)",
        "Break-even Value": lac_break_even_claims,
    },
    {
        "Model": "Model 2: Wise Path only",
        "Break-even Driver": "Annual Collections ($)",
        "Break-even Value": wise_break_even_collections,
    },
    {
        "Model": "Model 3: Combined",
        "Break-even Driver": "Proportional Volume Multiplier (k)",
        "Break-even Value": combined_break_even_k,
    },
]

be_df = pd.DataFrame(be_rows)


def be_display(driver: str, val):
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "No break-even found (within search range)"
    if "Multiplier" in driver:
        return f"{val:.2f}x"
    return f"${val:,.0f}"


be_df["Break-even Value (Display)"] = [be_display(d, v) for d, v in zip(be_df["Break-even Driver"], be_df["Break-even Value"]) ]

st.dataframe(be_df[["Model", "Break-even Driver", "Break-even Value (Display)"]], use_container_width=True)

# Callouts
if wise_break_even_collections is not None:
    st.info(
        f"Wise Path break-even: if annual collections exceed **{fmt_money(wise_break_even_collections)}**, "
        f"in-house becomes cheaper than outsourced (given current staffing inputs)."
    )
else:
    st.warning(
        "Wise Path: No break-even found in the tested range. This usually means either "
        "in-house is always cheaper (even at low volume) or always more expensive (even at high volume), "
        "given your current tier rates and staffing."
    )

if lac_break_even_claims is not None:
    st.info(
        f"Laconia break-even: if annual paid claims exceed **{fmt_money(lac_break_even_claims)}**, "
        f"in-house becomes cheaper than outsourced (given current staffing inputs)."
    )
else:
    st.warning(
        "Laconia: No break-even found in the tested range. This can happen if the minimum monthly fee "
        "dominates at low volume, or if staffing cost is far above/below outsourced across the entire range."
    )

if combined_break_even_k is not None:
    st.info(
        f"Combined break-even: if both locations scale proportionally by about **{combined_break_even_k:.2f}x**, "
        f"in-house becomes cheaper than outsourced."
    )
else:
    st.warning(
        "Combined: No proportional break-even found between 0x and 10x. Adjust staffing assumptions or check drivers."
    )


# =========================
# Visual comparison (grouped bars)
# =========================

st.subheader("Visual Comparison (Grouped Bars)")

plot_df = summary_all.melt(
    id_vars=["Model"],
    value_vars=["Outsourced", "In-House"],
    var_name="Cost Type",
    value_name="Annual Cost",
)

chart = (
    alt.Chart(plot_df)
    .mark_bar()
    .encode(
        x=alt.X("Model:N", sort=summary_all["Model"].tolist(), title="Model"),
        xOffset=alt.XOffset("Cost Type:N"),
        y=alt.Y("Annual Cost:Q", title="Annual Cost ($)"),
        color=alt.Color("Cost Type:N", title="Cost Type"),
        tooltip=["Model", "Cost Type", alt.Tooltip("Annual Cost:Q", format=",.0f")],
    )
    .properties(height=420)
)

st.altair_chart(chart, use_container_width=True)
st.caption("Grouped bars compare outsourced vs in-house annual costs across the three models.")


# =========================
# Optional sanity checks
# =========================

with st.expander("Sanity checks (optional)"):
    st.write("These help verify inputs are behaving as expected.")
    st.write(f"- Laconia outsourced (annual): {fmt_money(lac_out)}")
    st.write(f"- Wise Path outsourced (annual): {fmt_money(wise_out)}")
    st.write(f"- Laconia in-house (annual): {fmt_money(lac_in)}")
    st.write(f"- Wise Path in-house (annual): {fmt_money(wise_in)}")