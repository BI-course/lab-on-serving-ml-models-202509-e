import ast
import re

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Grocery Product Recommender",
    page_icon="🛒",
    layout="centered"
)

# ---------------------------------------------------------------------------
# Load association rules from CSV
# The file must be present in the GitHub repository at:
#   rule/top_rules_7a.csv
# Columns: antecedents, consequents, support, confidence, lift
#
# Antecedents and consequents are stored as frozenset strings, e.g.:
#   frozenset({'matoke', 'sukuma wiki'})
# The helper below parses them back to Python frozensets.
# ---------------------------------------------------------------------------
def _parse_frozenset(s: str) -> frozenset:
    """Parse a frozenset string back to a Python frozenset."""
    match = re.match(r"frozenset\((.*)\)", str(s).strip())
    if match:
        return frozenset(ast.literal_eval(match.group(1)))
    return frozenset(ast.literal_eval(str(s).strip()))

@st.cache_resource
def load_rules() -> pd.DataFrame:
    rules = pd.read_csv("./rule/top_rules_7a.csv")
    rules["antecedents"] = rules["antecedents"].apply(_parse_frozenset)
    rules["consequents"] = rules["consequents"].apply(_parse_frozenset)
    return rules

association_rules_df = load_rules()

# Derive the full item catalogue from all rules so the multiselect is populated
# with every item that appears in at least one antecedent or consequent.
all_items = sorted(
    item
    for col in ("antecedents", "consequents")
    for fs in association_rules_df[col]
    for item in fs
)
ITEM_CATALOGUE = sorted(set(all_items))

# ---------------------------------------------------------------------------
# Recommendation logic  (mirrors Notebook 7a, Cell 49)
# ---------------------------------------------------------------------------
def get_recommendations(cart: set, rules_df: pd.DataFrame) -> tuple:
    """
    For every rule whose antecedent is a subset of the cart, collect the
    consequent items (excluding items already in the cart).

    Returns:
        recommendations : sorted list of recommended item strings
        matched_rules   : list of dicts with rule details for display
    """
    recommendations = set()
    matched_rules   = []

    for _, rule in rules_df.iterrows():
        if rule["antecedents"].issubset(cart):
            new_items = rule["consequents"] - cart
            if new_items:
                recommendations.update(new_items)
                matched_rules.append({
                    "Antecedents":  ", ".join(sorted(rule["antecedents"])),
                    "Consequents":  ", ".join(sorted(rule["consequents"])),
                    "Support":      round(float(rule["support"]),    4),
                    "Confidence":   round(float(rule["confidence"]), 4),
                    "Lift":         round(float(rule["lift"]),        4),
                })

    return sorted(recommendations), matched_rules

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("🛒 Grocery Product Recommender")
st.write(
    "Select the items already in your basket and click **Get Recommendations** "
    "to discover products that are frequently bought together, based on "
    "association rules mined with the **Apriori algorithm** on the "
    "[Hahsler et al. (2011)](https://jmlr.org/papers/v12/hahsler11a.html) "
    "*Groceries* dataset (9,835 market basket transactions)."
)
st.divider()

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("recommender_form"):
    st.subheader("🧺 Current Basket")

    selected_items = st.multiselect(
        "Select items in your basket",
        options=ITEM_CATALOGUE,
        default=["matoke", "maziwa mala"],
        help="Choose one or more items already in your basket."
    )

    # Allow free-text entry for items not in the catalogue
    custom_input = st.text_input(
        "Or type additional items (comma-separated)",
        placeholder="e.g. ndizi, mango",
        help="Items not in the dropdown can be entered here."
    )

    submitted = st.form_submit_button(
        "Get Recommendations", use_container_width=True, type="primary"
    )

# ---------------------------------------------------------------------------
# Recommendation output
# ---------------------------------------------------------------------------
if submitted:
    # Build the cart — combine multiselect and any free-text entries
    custom_items = (
        {item.strip().lower() for item in custom_input.split(",") if item.strip()}
        if custom_input else set()
    )
    cart = {item.lower() for item in selected_items} | custom_items

    if not cart:
        st.warning("⚠️ Please select or type at least one item in your basket.")
    else:
        recommendations, matched_rules = get_recommendations(cart, association_rules_df)

        st.divider()
        st.subheader("📊 Results")

        # --- Basket summary ---
        st.markdown("**Your basket:**")
        st.info("  |  ".join(f"🛒 {item}" for item in sorted(cart)))

        # --- Recommendations ---
        st.markdown("**Recommended items:**")
        if recommendations:
            cols = st.columns(min(len(recommendations), 4))
            for col, item in zip(cols, recommendations):
                col.success(f"✅ {item}")
        else:
            st.warning(
                "No recommendations found for this basket. "
                "Try adding more items or different combinations."
            )

        # --- Matched rules detail ---
        if matched_rules:
            st.divider()
            with st.expander("🔍 Association Rules That Fired", expanded=True):
                st.write(
                    "The table below shows every rule whose antecedent was "
                    "fully present in your basket and that produced a new recommendation."
                )
                st.dataframe(
                    pd.DataFrame(matched_rules),
                    use_container_width=True,
                    hide_index=True
                )

# ---------------------------------------------------------------------------
# How it works & reference (always visible)
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️ How the recommender works"):
    st.markdown(
        """
        The recommender uses **association rules** mined with the Apriori
        algorithm. Each rule has the form:

        > **If** a customer buys {antecedent items} **→ Then** they are likely
        > to also buy {consequent items}

        A rule **fires** when every item in its antecedent is present in the
        customer's basket. The consequent items (excluding anything already in
        the basket) are then surfaced as recommendations.

        Rules are ranked by **lift** then **confidence**:

        | Metric | Meaning |
        |---|---|
        | **Support** | Proportion of transactions containing both antecedent and consequent |
        | **Confidence** | Probability that the consequent is bought given the antecedent |
        | **Lift** | How much more likely the consequent is bought with the antecedent vs. independently (lift > 1 = positive association) |
        """
    )

with st.expander("📋 Item Catalogue"):
    st.write("The following items appear in the mined association rules:")
    st.write("  |  ".join(ITEM_CATALOGUE))