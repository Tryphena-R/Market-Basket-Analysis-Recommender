import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

# ---------------- UI Setup -------------------
st.set_page_config(page_title="Market Basket Recommender", layout="wide")
st.title("🛍️ Market Basket Analysis & Product Recommender")

st.markdown("""
Analyze customer buying patterns using the **Apriori algorithm** and get intelligent **product recommendations**.
""")

with st.expander("📘 What do Support, Confidence, and Lift mean?"):
    st.markdown("""
    - **Support**: Frequency of itemset in transactions.
    - **Confidence**: Likelihood of B when A is bought.
    - **Lift**: Strength of the association. > 1 means positive correlation.
    """)

# --------------- Load Excel --------------------
uploaded_file = st.sidebar.file_uploader("📂 Upload Online_Retail_Proper.xlsx", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # ---------- Preprocessing ----------
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[df['Quantity'] > 0]

    # ---------- Sidebar Parameters ----------
    st.sidebar.header("⚙️ Apriori Parameters")
    min_support = st.sidebar.slider("Min Support", 0.01, 0.1, 0.01, 0.01)
    min_confidence = st.sidebar.slider("Min Confidence", 0.1, 1.0, 0.2, 0.05)
    min_lift = st.sidebar.slider("Min Lift", 0.5, 5.0, 0.5, 0.1)

    # --------- Basket Matrix ----------
    basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().fillna(0)
    basket = basket.applymap(lambda x: 1 if x > 0 else 0)

    # --------- Apriori Frequent Sets ----------
    frequent_items = apriori(basket, min_support=min_support, use_colnames=True)
    rules = association_rules(frequent_items, metric="lift", min_threshold=min_lift)
    rules = rules[rules['confidence'] >= min_confidence]

    st.success(f"✅ Found {len(frequent_items)} frequent itemsets and {len(rules)} rules")

    # --------- Top Frequent Items Chart ----------
    if not frequent_items.empty:
        st.subheader("📊 Top 10 Frequent Items")
        top_items = frequent_items[frequent_items['itemsets'].apply(lambda x: len(x) == 1)]
        top_items = top_items.nlargest(10, 'support')
        fig, ax = plt.subplots()
        ax.barh([list(x)[0] for x in top_items['itemsets']], top_items['support'], color='skyblue')
        ax.set_xlabel("Support")
        ax.set_title("Top Frequent Items")
        st.pyplot(fig)

    # --------- Product Recommendations ----------
    if not rules.empty:
        st.subheader("🔍 Product Recommendations")
        all_products = sorted(set(item for ante in rules['antecedents'] for item in ante))
        selected_product = st.selectbox("Select a product:", all_products)

        def get_recommendations(prod):
            return rules[rules['antecedents'].apply(lambda x: prod in x)]

        recos = get_recommendations(selected_product)

        if not recos.empty:
            st.markdown(f"**Products bought with `{selected_product}`:**")
            st.dataframe(recos[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
            
            top = list(recos['consequents'].iloc[0])[0]
            st.info(f"📦 Bundle suggestion: Offer `{selected_product}` + `{top}` together to increase sales!")
        else:
            st.warning("No recommendations found for this product.")
    else:
        st.warning("No association rules generated. Try reducing support/confidence/lift.")

else:
    st.warning("👈 Upload your `Online_Retail_Proper.xlsx` file to begin.")
