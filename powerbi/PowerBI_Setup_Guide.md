# Enterprise Power BI Setup & Publishing Guide for ECIP Platform

This guide explains how to connect Power BI Desktop to the ECIP data warehouse artifacts and embed reports directly into the Streamlit dashboard.

---

## 1. Connect Power BI Desktop to Data Warehouse
1. Open **Power BI Desktop**.
2. Click **Get Data** -> **Text/CSV**.
3. Select `output/master_dataset.csv` and `output/feature_store.csv`.
4. Create data relationships:
   - `master_dataset.customer_unique_id` <---> `feature_store.customer_unique_id` (1-to-1 or 1-to-many).

---

## 2. Recommended Measures (DAX)
```dax
Total Revenue = SUM(master_dataset[price])

Total Orders = DISTINCTCOUNT(master_dataset[order_id])

Average Order Value = DIVIDE([Total Revenue], [Total Orders], 0)

Repeat Buyer Count = CALCULATE(COUNTROWS(feature_store), feature_store[total_orders] > 1)

Repeat Rate % = DIVIDE([Repeat Buyer Count], COUNTROWS(feature_store), 0)
```

---

## 3. Publish to Power BI Service & Embed URL
1. Click **Publish** in Power BI Desktop to upload the report to your Power BI Workspace.
2. Open the published report in [Power BI Web Service](https://app.powerbi.com).
3. Go to **File** -> **Embed Report** -> **Website or Portal** (or **Publish to Web**).
4. Copy the `<iframe>` URL or embed URL link.
5. Open `powerbi/config.yaml` in this project and paste the URL into `embed_url`.
