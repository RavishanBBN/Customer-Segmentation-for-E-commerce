README.md

# Customer Segmentation for E-commerce

## Overview
This project performs customer segmentation on transactional data using RFM (Recency, Frequency, Monetary) analysis, dimensionality reduction, and clustering. The goal is to identify actionable customer groups for targeted marketing and personalized offers.

## Repository Structure
```
Customer-Segmentation-for-E-commerce/
│
├── data/
│   ├── transactions.csv         # Raw transaction records (order_id, customer_id, date, amount, etc.)
│   └── customers.csv            # Customer demographic information (optional)
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb   # Data cleaning, RFM feature extraction, scaling
│   ├── 02_pca_clustering.ipynb       # PCA, K-Means & DBSCAN clustering analysis
│   └── 03_visualization_dashboard.ipynb  # Dashboard prototype with Plotly/Dash
│
├── src/
│   ├── preprocessing.py         # Functions: load CSV, clean data, compute RFM
│   ├── features.py              # Feature engineering: scaling, PCA
│   ├── clustering.py            # Clustering routines: KMeans, DBSCAN, silhouette score
│   └── dashboard_app.py         # Plotly/Dash app code for interactive visualization
│
├── requirements.txt             # Python dependencies
├── LICENSE                      # MIT License
└── README.md                    # Project documentation (this file)
```

## Data
- **transactions.csv**: Contains purchase history with columns:
  - `order_id`: Unique order identifier
  - `customer_id`: Unique customer identifier
  - `order_date`: Date of the transaction (YYYY-MM-DD)
  - `order_amount`: Monetary value of each order
- **customers.csv** (optional): Demographics (age, gender, location). Not required for RFM but useful for segmentation enrichment.

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Intellihack-TeamX/Customer-Segmentation-for-E-commerce.git
   cd Customer-Segmentation-for-E-commerce
   ```
2. Create a Python virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate   # macOS/Linux
   venv\Scripts\activate      # Windows
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Feature Engineering
1. **Load & Clean Data**  
   - Read `transactions.csv` into a Pandas DataFrame.  
   - Handle missing or null values (drop or impute as needed).  
   - Ensure `order_date` is parsed as `datetime`.

2. **Compute RFM Metrics**  
   ```python
   from src.preprocessing import compute_rfm

   df_transactions = pd.read_csv("data/transactions.csv", parse_dates=["order_date"])
   rfm_df = compute_rfm(df_transactions, reference_date="2023-12-31")
   ```
   - **Recency**: Days since last purchase per customer.
   - **Frequency**: Total number of purchases per customer.
   - **Monetary**: Total spend per customer.

3. **Scaling & Dimensionality Reduction**  
   ```python
   from src.features import scale_and_reduce

   # MinMax scale RFM features, then apply PCA to reduce to 3 components
   rfm_scaled, pca_df = scale_and_reduce(rfm_df[["Recency","Frequency","Monetary"]], n_components=3)
   ```

## Clustering
1. **K-Means Clustering**  
   ```python
   from src.clustering import run_kmeans

   kmeans_labels, kmeans_model = run_kmeans(pca_df, n_clusters=5, random_state=42)
   ```
   - Evaluate silhouette score to choose optimal `n_clusters`.

2. **DBSCAN Clustering**  
   ```python
   from src.clustering import run_dbscan

   dbscan_labels, dbscan_model = run_dbscan(pca_df, eps=0.5, min_samples=10)
   ```
   - Identify noise points and dense clusters.

3. **Assign Cluster Labels**  
   ```python
   rfm_df["KMeans_Segment"] = kmeans_labels
   rfm_df["DBSCAN_Segment"] = dbscan_labels
   ```

## Interactive Dashboard
The `src/dashboard_app.py` script launches a Plotly Dash web app that visualizes:
- 2D t-SNE or PCA scatter plot with cluster coloring.
- RFM distribution histograms per segment.
- Customer KPIs (average order value, frequency, recency) in tables.
- Dynamic dropdown filters for demographic attributes.

### Run Dashboard
```bash
python src/dashboard_app.py
```
Navigate to `http://127.0.0.1:8050/` in your browser.

## Usage Example
1. Execute data preprocessing and RFM computation:
   ```bash
   python -c "from src.preprocessing import compute_rfm; import pandas as pd; df=pd.read_csv('data/transactions.csv', parse_dates=['order_date']); print(compute_rfm(df,'2023-12-31').head())"
   ```
2. Perform clustering in a notebook:
   ```bash
   jupyter notebook notebooks/02_pca_clustering.ipynb
   ```
3. Launch dashboard for interactive exploration:
   ```bash
   python src/dashboard_app.py
   ```

## Results
- Identified **5 customer segments** using K-Means with silhouette score ≈ 0.62.
- DBSCAN flagged ~10% of customers as “outliers” (very low recency or extremely high spend).
- Dashboard allows filtering by segment and inspecting segment‐level KPIs.

## License
This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.



