# 🛍️ Market Basket Analysis & Product Recommender System

This project is a Streamlit web application that performs Market Basket Analysis on e-commerce transactional data using the Apriori algorithm and generates smart product recommendations.

It is designed with an intuitive interface for business stakeholders, allowing real-time filtering and automatic insights generation.

---

## 💡 Features

- 📁 **File Upload**: Upload your `.xlsx` transaction dataset
- 📊 **Visualization**: View top 10 frequent products with support values
- 🧠 **Apriori Algorithm**: Analyze frequent itemsets and association rules
- 🎯 **Product Recommendations**: Get intelligent suggestions based on past purchase combinations
- ⚙️ **Interactive Filters**: Adjust minimum Support, Confidence, and Lift
- 🧾 **Explainable AI Section**: Understand what Support, Confidence, and Lift mean
- 💼 **Business Insight Tips**: Recommendations for product bundling strategy

---

## 📁 Files Included

| File Name                  | Description                                        |
|---------------------------|----------------------------------------------------|
| `market_basket_analysis.py` | Main Streamlit app code                         |
| `README.md`               | This documentation file                           |

---

## ⚙️ Requirements

Install dependencies using pip:

```bash
pip install streamlit pandas mlxtend matplotlib
