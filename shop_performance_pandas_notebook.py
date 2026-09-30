# Databricks notebook source
# MAGIC %md
# MAGIC # How Is the Shop Performing? – simple pandas version
# MAGIC Uses only the functions from Lesson 10 (head, shape, info, isnull, dropna, fillna, drop_duplicates, replace, astype, merge, filtering) plus `groupby` for the summaries.
# MAGIC
# MAGIC **Setup:** upload the 4 CSV files to a Volume (Catalog → Create → Volume → `shop_data`) and set `BASE` below. Run top to bottom.

# COMMAND ----------

import pandas as pd
import matplotlib.pyplot as plt

BASE = "/Volumes/workspace/default/shop_data"     # <-- change to your Volume path

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1 – Get to know the data

# COMMAND ----------

customers = pd.read_csv(BASE + "/customers.csv")
orders    = pd.read_csv(BASE + "/orders.csv")
products  = pd.read_csv(BASE + "/products.csv")

payments  = pd.read_csv(BASE + "/payments.csv")

# COMMAND ----------

# Look at each file: first rows, size, column types, summary
print("CUSTOMERS  (one row = one customer)")
display(customers.head(3))
print(customers.shape)
customers.info()

# COMMAND ----------

print("ORDERS  (one row = one order line)")
display(orders.head(3))
print(orders.shape)
orders.info()
display(orders.describe())

# COMMAND ----------

print("PRODUCTS  (one row = one product)")
display(products)
print(products.shape)

# COMMAND ----------

print("PAYMENTS  (one row = one payment attempt)")
display(payments.head(3))
print(payments.shape)
payments.info()
print(payments["PaymentStatus"].value_counts())        # Paid / Failed / Refunded

# COMMAND ----------
# MAGIC %md
# MAGIC **One question each file can answer**
# MAGIC - customers → which cities and segments are most valuable?
# MAGIC - orders → how much did we sell and is it growing?
# MAGIC - products → which products and categories earn the most?
# MAGIC - payments → what share of payments fail?
# MAGIC
# MAGIC **How the files connect:** orders.CustomerID → customers | orders.ProductID → products | orders.OrderID → payments

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2 – Clean the data

# COMMAND ----------

# 2a. Missing values
print(customers.isnull().sum())
print(orders.isnull().sum())
print(payments.isnull().sum())                   # 35 blank PaymentDate - not used in the analysis, so I keep the rows

# 2b. Duplicates
print("Duplicate orders:", orders.duplicated().sum())
print("Duplicate payments:", payments.duplicated().sum(), "| orders with 2+ payments:", payments["OrderID"].duplicated().sum())

# 2c. Values that make no sense
print(orders["Quantity"].min(), orders["Quantity"].max())
print(orders["Discount"].min(), orders["Discount"].max())
display(products.sort_values("UnitPrice"))      # Monitor = 21 looks too low

# 2d. Inconsistent text
print(customers["City"].value_counts())          # 'tehran' and 'Mashad' are spelled differently

# 2e. Broken links
print("Orders with unknown customer:", (~orders["CustomerID"].isin(customers["CustomerID"])).sum())
print("Orders with unknown product :", (~orders["ProductID"].isin(products["ProductID"])).sum())
print("Payments with unknown order  :", (~payments["OrderID"].isin(orders["OrderID"])).sum())
print("Orders with no payment       :", (~orders["OrderID"].isin(payments["OrderID"])).sum())

# COMMAND ----------

# Apply the cleaning decisions
print("Rows at start:", len(orders))

orders = orders.drop_duplicates()                                          # remove duplicate orders
print("after removing duplicates:", len(orders))

orders = orders[orders["CustomerID"].isin(customers["CustomerID"])]        # remove orders of unknown customers
print("after removing broken customer links:", len(orders))

orders = orders.dropna(subset=["OrderDate"])                               # no date = cannot be used in a trend
print("after removing missing dates:", len(orders))

orders = orders.dropna(subset=["Quantity"])                                # no quantity = no revenue
orders = orders[orders["Quantity"] > 0].copy()                                    # zero / negative quantity makes no sense
print("after removing bad quantities:", len(orders))

orders["Discount"] = orders["Discount"].fillna(0)                          # blank discount = no discount
orders["PaymentMethod"] = orders["PaymentMethod"].fillna("Unknown")        # keep the row, label it
orders["Quantity"] = orders["Quantity"].astype("int")
orders["OrderDate"] = pd.to_datetime(orders["OrderDate"])

customers["City"] = customers["City"].str.strip().str.title()              # 'tehran' -> 'Tehran'
customers["City"] = customers["City"].replace("Mashad", "Mashhad")
customers["City"] = customers["City"].fillna("Unknown")

print("Clean orders:", len(orders))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3 – Combine the tables (check the row count after each merge)

# COMMAND ----------

df = orders.merge(products, on="ProductID", how="left")
print("after products :", len(df))

df = df.merge(customers, on="CustomerID", how="left")
print("after customers:", len(df))

df = df.merge(payments[["OrderID", "PaymentStatus"]], on="OrderID", how="left")
print("after payments :", len(df))                          # each order has exactly one payment row, so the count must not change
print(df["PaymentStatus"].isnull().sum(), "orders without a payment")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4 – New columns
# MAGIC Revenue counts **only Completed orders whose payment was Paid**. Cancelled and Returned orders, and payments that Failed or were Refunded, are money the shop did not keep.

# COMMAND ----------

df["Revenue"] = df["Quantity"] * df["UnitPrice"] * (1 - df["Discount"])
df["GrossValue"] = df["Quantity"] * df["UnitPrice"]
df["Year"] = df["OrderDate"].dt.year
df["Month"] = df["OrderDate"].dt.month
df["YearMonth"] = df["OrderDate"].dt.strftime("%Y-%m")

sales = df[(df["Status"] == "Completed") & (df["PaymentStatus"] == "Paid")]

df.to_csv(BASE + "/shop_clean.csv", index=False)      # your cleaned working file
display(df.head())

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5 & 6 – Business questions and charts
# MAGIC ### Q1. Total revenue, orders, average order value

# COMMAND ----------

total_revenue = sales["Revenue"].sum()
total_orders = sales["OrderID"].nunique()
avg_order_value = total_revenue / total_orders

print("Total revenue       :", round(total_revenue))
print("Orders              :", total_orders)
print("Average order value :", round(avg_order_value, 2))

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q2. Is revenue growing? Any seasonal peaks?

# COMMAND ----------

monthly = sales.groupby("YearMonth")["Revenue"].sum().reset_index()
display(monthly)

plt.figure(figsize=(11, 4))
plt.plot(monthly["YearMonth"], monthly["Revenue"], marker="o", color="orange")
plt.title("Monthly revenue, Jan 2024 – Jun 2026")
plt.xlabel("Month"); plt.ylabel("Revenue")
plt.xticks(range(0, len(monthly), 3), monthly["YearMonth"][::3], rotation=45)
plt.show()

# Fair comparison: January–June of each year
first_half = sales[sales["Month"] <= 6].groupby("Year")["Revenue"].sum()
print(first_half)
print((first_half.pct_change() * 100).round(1))

# Are new customers still arriving?
print("Latest customer signup:", customers["SignupDate"].max())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q3. Products and categories – revenue vs units

# COMMAND ----------

by_product = sales.groupby("ProductName").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum")).reset_index()
by_product = by_product.sort_values("Revenue", ascending=False)
display(by_product)

by_category = sales.groupby("Category").agg(Revenue=("Revenue", "sum"), Units=("Quantity", "sum")).reset_index()
by_category = by_category.sort_values("Revenue", ascending=False)
display(by_category)

top_rev = by_product.sort_values("Revenue").tail(10)
plt.barh(top_rev["ProductName"], top_rev["Revenue"], color="orange")
plt.title("Top 10 products by revenue"); plt.xlabel("Revenue")
plt.show()

top_units = by_product.sort_values("Units").tail(10)
plt.barh(top_units["ProductName"], top_units["Units"], color="navy")
plt.title("Top 10 products by units sold"); plt.xlabel("Units")
plt.show()

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q4. Cities and customer segments

# COMMAND ----------

by_city = sales.groupby("City").agg(Revenue=("Revenue", "sum"), Buyers=("CustomerID", "nunique")).reset_index()
by_city = by_city.sort_values("Revenue", ascending=False)
display(by_city)

plt.barh(by_city["City"][::-1], by_city["Revenue"][::-1], color="orange")
plt.title("Revenue by city"); plt.xlabel("Revenue")
plt.show()

by_segment = sales.groupby("CustomerSegment").agg(Revenue=("Revenue", "sum"), Orders=("OrderID", "nunique")).reset_index()
customers_per_segment = customers["CustomerSegment"].value_counts().reset_index()
customers_per_segment.columns = ["CustomerSegment", "Customers"]
by_segment = by_segment.merge(customers_per_segment, on="CustomerSegment", how="left")
by_segment["RevenuePerCustomer"] = (by_segment["Revenue"] / by_segment["Customers"]).round(1)
display(by_segment)

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q5. Cancelled / returned orders and payment methods

# COMMAND ----------

print((df["Status"].value_counts(normalize=True) * 100).round(1))

df["Cancelled"] = (df["Status"] == "Cancelled") * 100
df["Returned"] = (df["Status"] == "Returned") * 100
by_method = df.groupby("PaymentMethod")[["Cancelled", "Returned"]].mean().round(1)
display(by_method)

# COMMAND ----------

# Payment status: how many payments fail, and does one method fail more?
print((df["PaymentStatus"].value_counts(normalize=True) * 100).round(1))

df["PaymentFailed"] = (df["PaymentStatus"] == "Failed") * 100
fail_by_method = df.groupby("PaymentMethod")["PaymentFailed"].mean().round(1).reset_index()
display(fail_by_method)

plt.bar(fail_by_method["PaymentMethod"], fail_by_method["PaymentFailed"], color="orange")
plt.title("Payment failure rate by payment method"); plt.xlabel("Payment method"); plt.ylabel("% of payments failed")
plt.show()

# Completed orders that were NOT paid (money not received)
completed = df[df["Status"] == "Completed"]
print(completed["PaymentStatus"].value_counts())

# COMMAND ----------
# MAGIC %md
# MAGIC ### Q6. Do bigger discounts mean bigger orders?

# COMMAND ----------

done = sales                       # revenue-earning orders only
by_discount = done.groupby("Discount").agg(Orders=("OrderID", "count"),
                                           AvgUnits=("Quantity", "mean"),
                                           AvgOrderValue=("Revenue", "mean"),
                                           TotalRevenue=("Revenue", "sum")).round(2).reset_index()
display(by_discount)

plt.bar((by_discount["Discount"] * 100).astype(int).astype(str) + "%", by_discount["AvgOrderValue"], color="orange")
plt.title("Average order value by discount"); plt.xlabel("Discount"); plt.ylabel("Order value")
plt.show()
