"""
Generates a synthetic, intentionally-messy e-commerce sales dataset.

NOTE: No dataset was supplied for this task and this sandbox has no internet
access to pull a real Kaggle dataset, so a realistic synthetic dataset is
generated here to stand in for the "E-commerce sales data (Kaggle)" option.
Swap this out for a real CSV in data/raw/ and re-run the notebook if you'd
like to use actual company/Kaggle data instead.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

N = 5000

categories = ["Electronics", "Clothing", "Home & Kitchen", "Beauty", "Sports", "Toys", "Books"]
category_price_range = {
    "Electronics": (50, 1200),
    "Clothing": (10, 150),
    "Home & Kitchen": (15, 400),
    "Beauty": (5, 100),
    "Sports": (10, 300),
    "Toys": (5, 120),
    "Books": (5, 60),
}
regions = ["North", "South", "East", "West", "Central"]
payment_methods = ["Credit Card", "Debit Card", "UPI", "Net Banking", "Cash on Delivery"]
ship_modes = ["Standard", "Express", "Same Day"]

order_dates = pd.to_datetime("2024-01-01") + pd.to_timedelta(
    rng.integers(0, 545, size=N), unit="D"
)

rows = []
for i in range(N):
    cat = rng.choice(categories, p=[0.22, 0.20, 0.16, 0.12, 0.12, 0.1, 0.08])
    lo, hi = category_price_range[cat]
    unit_price = round(rng.uniform(lo, hi), 2)
    qty = int(rng.choice([1, 1, 1, 2, 2, 3, 4, 5], p=[0.35, 0.2, 0.15, 0.12, 0.08, 0.05, 0.03, 0.02]))
    discount_pct = round(rng.choice([0, 0, 0, 5, 10, 15, 20, 25, 30], p=[0.35,0.1,0.05,0.15,0.15,0.08,0.06,0.04,0.02]), 2)
    revenue = round(unit_price * qty * (1 - discount_pct / 100), 2)

    row = {
        "order_id": f"ORD{100000+i}",
        "order date": order_dates[i],
        "Customer ID": f"CUST{rng.integers(1, 1800):05d}",
        "customer_age": int(rng.normal(38, 13)),
        "Gender": rng.choice(["Male", "Female", "Other", "M", "F"], p=[0.44, 0.44, 0.02, 0.06, 0.04]),
        "product_category": cat,
        "unit price": unit_price,
        "Quantity": qty,
        "discount_percent": discount_pct,
        "Revenue": revenue,
        "region": rng.choice(regions),
        "payment method": rng.choice(payment_methods, p=[0.35, 0.2, 0.25, 0.1, 0.1]),
        "ship_mode": rng.choice(ship_modes, p=[0.6, 0.3, 0.1]),
        "customer_rating": rng.choice([1, 2, 3, 4, 5, np.nan], p=[0.03, 0.05, 0.15, 0.32, 0.35, 0.10]),
    }
    rows.append(row)

df = pd.DataFrame(rows)

# ---- Inject realistic messiness ----

# 1. Missing values scattered across several columns
for col, frac in [("customer_age", 0.04), ("Gender", 0.02), ("region", 0.015),
                   ("payment method", 0.01), ("unit price", 0.008)]:
    idx = rng.choice(df.index, size=int(frac * N), replace=False)
    df.loc[idx, col] = np.nan

# 2. Some duplicate rows (exact dupes)
dupe_idx = rng.choice(df.index, size=40, replace=False)
df = pd.concat([df, df.loc[dupe_idx]], ignore_index=True)

# 3. Inconsistent text casing / whitespace in categorical columns
noisy_idx = rng.choice(df.index, size=150, replace=False)
df.loc[noisy_idx, "product_category"] = df.loc[noisy_idx, "product_category"].str.upper()
noisy_idx2 = rng.choice(df.index, size=100, replace=False)
df.loc[noisy_idx2, "region"] = df.loc[noisy_idx2, "region"].astype(str) + "  "

# 4. Outliers / bad values
out_idx = rng.choice(df.index, size=25, replace=False)
df.loc[out_idx, "customer_age"] = rng.integers(95, 140, size=25)  # implausible ages
out_idx2 = rng.choice(df.index, size=15, replace=False)
df.loc[out_idx2, "customer_age"] = rng.integers(-5, 0, size=15)  # negative ages (bad data)
out_idx3 = rng.choice(df.index, size=20, replace=False)
df.loc[out_idx3, "Revenue"] = df.loc[out_idx3, "Revenue"] * rng.uniform(15, 40, size=20)  # revenue spikes

# 5. order date stored as string (mixed formats) to force a dtype fix
df["order date"] = df["order date"].dt.strftime("%Y-%m-%d")

# 6. Shuffle rows and reset index
df = df.sample(frac=1, random_state=1).reset_index(drop=True)

df.to_csv("/home/claude/apexplanet-data-analytics/data/raw/ecommerce_sales_raw.csv", index=False)
print("Raw shape:", df.shape)
print(df.dtypes)
print(df.isna().sum())
