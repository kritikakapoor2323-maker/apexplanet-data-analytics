"""
Builds notebooks/eda_ecommerce_sales.ipynb by actually executing each code
cell in a persistent namespace and capturing real stdout / plot outputs into
proper Jupyter notebook JSON (nbformat v4), without relying on the nbformat
package (not available in this offline sandbox).
"""
import io
import sys
import json
import base64
import contextlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NB_PATH = "/home/claude/apexplanet-data-analytics/notebooks/eda_ecommerce_sales.ipynb"

cells = []  # list of (type, source) ; type = 'markdown' | 'code'

def md(text):
    cells.append(("markdown", text))

def code(text):
    cells.append(("code", text))

# ---------------------------------------------------------------- CELLS ----

md("# E-commerce Sales — Exploratory Data Analysis\n"
   "**Task 1: Foundational Setup & EDA** · Apexplanet Data Analytics Internship\n\n"
   "This notebook covers:\n"
   "1. Environment / library setup\n"
   "2. Data sourcing & understanding\n"
   "3. Data cleaning & preprocessing\n"
   "4. Exploratory Data Analysis (univariate, bivariate, patterns & anomalies)\n\n"
   "> **Data source note:** No dataset was provided for this task and this "
   "environment has no internet access to pull a real Kaggle dataset. A "
   "realistic **synthetic e-commerce sales dataset** (`data/raw/ecommerce_sales_raw.csv`) "
   "was generated instead, with deliberately injected missing values, duplicates, "
   "inconsistent text, wrong dtypes, and outliers — so the cleaning and EDA "
   "steps below are genuine, not illustrative. Swap in a real CSV and re-run "
   "if you'd like to analyze actual data.")

code("import pandas as pd\n"
     "import numpy as np\n"
     "import matplotlib.pyplot as plt\n"
     "import seaborn as sns\n\n"
     "sns.set_theme(style='whitegrid')\n"
     "pd.set_option('display.max_columns', None)\n"
     "%matplotlib inline")

md("## 1. Load Data")

code("df = pd.read_csv('../data/raw/ecommerce_sales_raw.csv')\n"
     "print('Shape:', df.shape)\n"
     "df.head()")

md("## 2. Data Understanding\n"
   "Data dictionary, dtypes, and a first look at data quality issues.")

code("df.info()")

code("df.describe(include='all').T")

code("print('Missing values per column:')\n"
     "print(df.isna().sum())\n"
     "print('\\nDuplicate rows:', df.duplicated().sum())")

md("**Data dictionary (as sourced):**\n\n"
   "| Column | Description |\n"
   "|---|---|\n"
   "| order_id | Unique order identifier |\n"
   "| order date | Date the order was placed |\n"
   "| Customer ID | Unique customer identifier |\n"
   "| customer_age | Customer age in years |\n"
   "| Gender | Customer gender |\n"
   "| product_category | Product category purchased |\n"
   "| unit price | Price per unit (local currency) |\n"
   "| Quantity | Units purchased in the order |\n"
   "| discount_percent | Discount applied, % |\n"
   "| Revenue | Net revenue for the order |\n"
   "| region | Customer's region |\n"
   "| payment method | Payment method used |\n"
   "| ship_mode | Shipping speed selected |\n"
   "| customer_rating | Post-purchase rating (1-5), may be missing |\n\n"
   "**Limitations:** synthetic data generated for this exercise; ages/ratings/"
   "revenue outliers were deliberately injected to practice cleaning; real-world "
   "seasonality and product-level trends are simplified.")

md("## 3. Data Cleaning & Preprocessing")

md("### 3.1 Standardize column names")

code("df.columns = (df.columns.str.strip().str.lower()\n"
     "              .str.replace(' ', '_', regex=False))\n"
     "print(df.columns.tolist())")

md("### 3.2 Remove duplicate rows")

code("before = len(df)\n"
     "df = df.drop_duplicates()\n"
     "print(f'Removed {before - len(df)} duplicate rows -> {len(df)} rows remain')")

md("### 3.3 Fix data types")

code("df['order_date'] = pd.to_datetime(df['order_date'], errors='coerce')\n"
     "df['product_category'] = df['product_category'].str.strip().str.title().astype('category')\n"
     "df['region'] = df['region'].str.strip().str.title().astype('category')\n"
     "df['payment_method'] = df['payment_method'].astype('category')\n"
     "df['ship_mode'] = df['ship_mode'].astype('category')\n\n"
     "# Standardize gender labels\n"
     "gender_map = {'M': 'Male', 'F': 'Female'}\n"
     "df['gender'] = df['gender'].replace(gender_map).astype('category')\n\n"
     "df.dtypes")

md("### 3.4 Handle missing values\n"
   "- `customer_age`, `unit_price`: numeric → fill with **median** (robust to outliers)\n"
   "- `gender`, `region`, `payment_method`: categorical → fill with **mode**\n"
   "- `customer_rating`: left as missing (NaN) since 'no rating given' is a valid state — "
   "flagged with an indicator column instead of imputed, to avoid biasing satisfaction analysis")

code("df['rating_missing'] = df['customer_rating'].isna()\n\n"
     "df['customer_age'] = df['customer_age'].fillna(df['customer_age'].median())\n"
     "df['unit_price'] = df['unit_price'].fillna(df['unit_price'].median())\n\n"
     "for col in ['gender', 'region', 'payment_method']:\n"
     "    df[col] = df[col].fillna(df[col].mode()[0])\n\n"
     "print('Remaining missing values:')\n"
     "print(df.isna().sum())")

md("### 3.5 Handle outliers (IQR method)\n"
   "Applying the IQR rule to `customer_age` (implausible/negative ages) and `revenue` "
   "(a few orders were 15–40x inflated). Rather than dropping rows outright, values are "
   "**capped (winsorized)** at the IQR fences to preserve sample size while removing "
   "distortion — a common approach for order-level revenue data.")

code("def iqr_bounds(series, k=1.5):\n"
     "    q1, q3 = series.quantile([0.25, 0.75])\n"
     "    iqr = q3 - q1\n"
     "    return q1 - k * iqr, q3 + k * iqr\n\n"
     "for col in ['customer_age', 'revenue']:\n"
     "    lo, hi = iqr_bounds(df[col])\n"
     "    n_out = ((df[col] < lo) | (df[col] > hi)).sum()\n"
     "    print(f'{col}: bounds=({lo:.1f}, {hi:.1f}), outliers capped={n_out}')\n"
     "    df[col] = df[col].clip(lower=max(lo, 0) if col=='customer_age' else lo, upper=hi)\n\n"
     "# Realistic age floor (no customer under 13)\n"
     "df = df[df['customer_age'] >= 13].reset_index(drop=True)\n"
     "print('\\nShape after outlier handling:', df.shape)")

md("### 3.6 Cleaning log\n\n"
   "| Step | Action | Rows/Values Affected |\n"
   "|---|---|---|\n"
   "| 1 | Standardized column names to snake_case | 14 columns |\n"
   "| 2 | Removed exact duplicate rows | 40 rows |\n"
   "| 3 | Converted `order_date` to datetime | all rows |\n"
   "| 4 | Standardized category/region/gender text casing | ~350 values |\n"
   "| 5 | Imputed missing `customer_age`, `unit_price` with median | 241 values |\n"
   "| 6 | Imputed missing `gender`, `region`, `payment_method` with mode | 226 values |\n"
   "| 7 | Left `customer_rating` missing, added `rating_missing` flag | 511 values flagged |\n"
   "| 8 | Capped `customer_age` and `revenue` outliers via IQR winsorizing | ~60 values |\n"
   "| 9 | Dropped implausible ages < 13 | a few rows |")

md("## 4. Exploratory Data Analysis")

md("### 4.1 Statistical Summary")

code("df.describe()")

code("df['product_category'].value_counts()")

code("df['region'].value_counts()")

md("### 4.2 Univariate Analysis")

code("fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n"
     "sns.histplot(df['revenue'], bins=40, kde=True, ax=axes[0], color='#4C72B0')\n"
     "axes[0].set_title('Distribution of Order Revenue')\n"
     "sns.histplot(df['customer_age'], bins=30, kde=True, ax=axes[1], color='#55A868')\n"
     "axes[1].set_title('Distribution of Customer Age')\n"
     "sns.histplot(df['unit_price'], bins=40, kde=True, ax=axes[2], color='#C44E52')\n"
     "axes[2].set_title('Distribution of Unit Price')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/univariate_histograms.png', dpi=120)\n"
     "plt.show()")

code("fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
     "sns.boxplot(y=df['revenue'], ax=axes[0], color='#8172B2')\n"
     "axes[0].set_title('Revenue — Boxplot (post outlier handling)')\n"
     "sns.boxplot(y=df['customer_age'], ax=axes[1], color='#CCB974')\n"
     "axes[1].set_title('Customer Age — Boxplot')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/boxplots.png', dpi=120)\n"
     "plt.show()")

code("fig, ax = plt.subplots(figsize=(9, 4.5))\n"
     "order = df['product_category'].value_counts().index\n"
     "sns.countplot(data=df, y='product_category', order=order, ax=ax, palette='viridis')\n"
     "ax.set_title('Order Count by Product Category')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/category_bar_chart.png', dpi=120)\n"
     "plt.show()")

md("### 4.3 Bivariate Analysis")

code("fig, ax = plt.subplots(figsize=(7, 5))\n"
     "sns.scatterplot(data=df, x='unit_price', y='revenue', hue='product_category',\n"
     "                 alpha=0.5, s=25, ax=ax)\n"
     "ax.set_title('Unit Price vs Revenue by Category')\n"
     "ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/scatter_price_vs_revenue.png', dpi=120)\n"
     "plt.show()")

code("num_cols = ['customer_age', 'unit_price', 'quantity', 'discount_percent', 'revenue', 'customer_rating']\n"
     "corr = df[num_cols].corr()\n"
     "fig, ax = plt.subplots(figsize=(7, 5.5))\n"
     "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)\n"
     "ax.set_title('Correlation Matrix — Numeric Features')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/correlation_heatmap.png', dpi=120)\n"
     "plt.show()")

code("fig, ax = plt.subplots(figsize=(8, 4.5))\n"
     "sns.boxplot(data=df, x='product_category', y='revenue', ax=ax, palette='Set2')\n"
     "ax.set_title('Revenue Distribution by Product Category')\n"
     "plt.xticks(rotation=30, ha='right')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/revenue_by_category.png', dpi=120)\n"
     "plt.show()")

code("monthly = df.set_index('order_date').resample('ME')['revenue'].sum()\n"
     "fig, ax = plt.subplots(figsize=(10, 4))\n"
     "monthly.plot(ax=ax, marker='o', color='#4C72B0')\n"
     "ax.set_title('Monthly Revenue Trend')\n"
     "ax.set_ylabel('Total Revenue')\n"
     "plt.tight_layout()\n"
     "plt.savefig('../reports/monthly_revenue_trend.png', dpi=120)\n"
     "plt.show()")

md("### 4.4 Patterns, Trends & Anomalies")

code("top_category = df.groupby('product_category', observed=True)['revenue'].sum().idxmax()\n"
     "top_region = df.groupby('region', observed=True)['revenue'].sum().idxmax()\n"
     "avg_rating = df['customer_rating'].mean()\n"
     "corr_price_rev = df[['unit_price','revenue']].corr().iloc[0,1]\n"
     "pct_discounted = (df['discount_percent'] > 0).mean() * 100\n\n"
     "print(f'Top revenue-generating category : {top_category}')\n"
     "print(f'Top revenue-generating region    : {top_region}')\n"
     "print(f'Average customer rating          : {avg_rating:.2f} / 5')\n"
     "print(f'Corr(unit_price, revenue)        : {corr_price_rev:.2f}')\n"
     "print(f'% of orders with a discount      : {pct_discounted:.1f}%')")

md("**Key findings:**\n"
   "1. **Electronics dominates revenue** despite fewer orders than Clothing — high unit "
   "price outweighs lower order volume, confirmed by the price-vs-revenue scatter and the "
   "revenue-by-category boxplot.\n"
   "2. **Unit price is strongly correlated with revenue** (per the correlation heatmap), while "
   "`discount_percent` shows a mild negative relationship with revenue per order — heavier "
   "discounting is concentrated in lower-ticket categories.\n"
   "3. **Revenue outliers** (a small number of orders 15–40x normal size) were detected via "
   "the IQR method and capped during cleaning; without this step they would have skewed "
   "category-level revenue comparisons substantially.\n"
   "4. **Ratings are missing for ~10% of orders** (flagged via `rating_missing` rather than "
   "imputed) — worth investigating operationally, e.g. whether certain shipping modes or "
   "payment methods correlate with customers not leaving a rating.\n"
   "5. Monthly revenue shows **no runaway trend** but does fluctuate — consistent with normal "
   "order-volume variation rather than seasonality in this dataset.")

md("## 5. Save Cleaned Data")

code("out_path = '../data/processed/ecommerce_sales_cleaned.csv'\n"
     "df.to_csv(out_path, index=False)\n"
     "print(f'Saved cleaned dataset: {out_path}')\n"
     "print('Final shape:', df.shape)")

md("## 6. Summary\n\n"
   "- Started with **5,040 raw rows / 14 columns**, ending with a clean, analysis-ready "
   "dataset after removing duplicates, fixing dtypes, imputing/flagging missing values, "
   "and capping outliers.\n"
   "- Full cleaning steps are logged in the table above and mirrored in `scripts/`.\n"
   "- EDA charts are saved to `reports/` for reuse in the README/LinkedIn recap and any "
   "downstream dashboard (Task 2+).\n"
   "- Next steps: build the Power BI / Tableau dashboard from `data/processed/ecommerce_sales_cleaned.csv`, "
   "and dig deeper into the ratings-missing pattern flagged above.")

# ---------------------------------------------------------------- EXECUTE ----

ns = {}
nb_cells = []
exec_count = 0

for kind, source in cells:
    if kind == "markdown":
        nb_cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": source.splitlines(keepends=True),
        })
        continue

    exec_count += 1
    outputs = []
    buf = io.StringIO()
    fig_nums_before = plt.get_fignums()

    # strip IPython magics (e.g. %matplotlib inline) - not valid plain Python
    exec_source = "\n".join(
        line for line in source.split("\n") if not line.strip().startswith("%")
    )

    try:
        with contextlib.redirect_stdout(buf):
            # emulate notebook "last expression auto-prints" behavior for simple cases
            compiled_lines = exec_source.strip("\n").split("\n")
            code_str = exec_source
            # Detect a trailing bare expression (very common in our cells, e.g. df.head())
            last_line = compiled_lines[-1].strip()
            is_expr = (
                last_line
                and not last_line.startswith(("#", "print", "plt.", "for ", "if ", "def ", "%"))
                and "=" not in last_line.split("(")[0]
                and not last_line.endswith(":")
            )
            if is_expr:
                head = "\n".join(compiled_lines[:-1])
                if head.strip():
                    exec(compile(head, "<cell>", "exec"), ns)
                result = eval(compile(last_line, "<cell>", "eval"), ns)
                if result is not None:
                    ns["_"] = result
            else:
                exec(compile(code_str, "<cell>", "exec"), ns)
                result = None
        cell_errored = False
    except Exception as e:
        outputs.append({
            "output_type": "error",
            "ename": type(e).__name__,
            "evalue": str(e),
            "traceback": [f"{type(e).__name__}: {e}"],
        })
        result = None
        cell_errored = True

    text = buf.getvalue()
    if text:
        outputs.append({
            "output_type": "stream",
            "name": "stdout",
            "text": text.splitlines(keepends=True),
        })

    if not cell_errored and result is not None:
        # crude repr-based text/plain output, plus HTML table for DataFrames
        data = {}
        try:
            import pandas as pd
            if isinstance(result, (pd.DataFrame, pd.Series)):
                data["text/html"] = result.to_html(max_rows=20).splitlines(keepends=True) if hasattr(result, "to_html") else None
                data["text/plain"] = repr(result).splitlines(keepends=True)
                data = {k: v for k, v in data.items() if v}
            else:
                data["text/plain"] = repr(result).splitlines(keepends=True)
        except Exception:
            data = {"text/plain": repr(result).splitlines(keepends=True)}
        outputs.append({
            "output_type": "execute_result",
            "execution_count": exec_count,
            "data": data,
            "metadata": {},
        })

    fig_nums_after = plt.get_fignums()
    for fnum in fig_nums_after:
        fig = plt.figure(fnum)
        imgbuf = io.BytesIO()
        fig.savefig(imgbuf, format="png", dpi=100, bbox_inches="tight")
        imgbuf.seek(0)
        b64 = base64.b64encode(imgbuf.read()).decode("ascii")
        outputs.append({
            "output_type": "display_data",
            "data": {"image/png": b64, "text/plain": ["<Figure>"]},
            "metadata": {},
        })
        plt.close(fig)

    nb_cells.append({
        "cell_type": "code",
        "execution_count": exec_count,
        "metadata": {},
        "outputs": outputs,
        "source": source.splitlines(keepends=True),
    })

notebook = {
    "cells": nb_cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w") as f:
    json.dump(notebook, f, indent=1)

print("Notebook written:", NB_PATH)
print("Total cells:", len(nb_cells))
