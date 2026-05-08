# Adding imports
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import os
 

# Creating a directory structure
os.makedirs("data/raw", exist_ok=True)
print("Directories created")

# Loading the sklearn dataset
sklearn_housing = fetch_california_housing(as_frame=True)
df_sklearn = sklearn_housing.frame
df_sklearn.rename(columns={"MedHouseVal": "median_house_value"}, inplace=True)

print(f" sklearn loaded | Shape: {df_sklearn.shape}")
print(df_sklearn.head())

# Load Zillow CSV directly from GitHub (no upload needed)
import pandas as pd

z_url = "https://raw.githubusercontent.com/timfan705/CAmarket/main/zillow.csv"
df_zillow_raw = pd.read_csv(z_url, low_memory=False)

print("Raw Zillow shape:", df_zillow_raw.shape)
print(df_zillow_raw.head())
print("RegionType counts:\n", df_zillow_raw["RegionType"].value_counts())

# Keep only California MSAs (this Zillow file is MSA-only + country)
df_zillow_ca = df_zillow_raw[
    (df_zillow_raw["RegionType"] == "msa") &
    (df_zillow_raw["StateName"] == "CA")
].copy()

# Identify date columns and pick the latest one
meta_cols = ["RegionID","SizeRank","RegionName","RegionType","StateName"]
date_cols = sorted([c for c in df_zillow_raw.columns if c not in meta_cols], key=pd.to_datetime)

latest_col = date_cols[-1]  # e.g., "2026-03-31"
print("Using latest Zillow column:", latest_col)

# Build a simple lookup table: RegionName -> latest Zillow value
df_zillow_latest = df_zillow_ca[["RegionName"]].copy()
df_zillow_latest["zillow_median_price"] = df_zillow_ca[latest_col].astype(float)

print("Zillow CA latest prices | Shape:", df_zillow_latest.shape)
print(df_zillow_latest.head())

# Check for nulls and data types
# first inspect the sklearn dataset
print(" sklearn Dataset ")
print(df_sklearn.shape)
print(df_sklearn.isnull().sum())
print(df_sklearn.dtypes)
print(df_sklearn.describe())

# second inspect zillow data set
print(" Zillow Dataset ")
print(df_zillow_latest.shape)
print(df_zillow_latest.isnull().sum())
print(df_zillow_latest.head(10))

msa_centroids = pd.DataFrame({
    "RegionName": [
        "Bakersfield, CA","Chico, CA","Clearlake, CA","Crescent City, CA","El Centro, CA",
        "Eureka, CA","Fresno, CA","Hanford, CA","Los Angeles, CA","Madera, CA","Merced, CA",
        "Modesto, CA","Napa, CA","Oxnard, CA","Red Bluff, CA","Redding, CA","Riverside, CA",
        "Sacramento, CA","Salinas, CA","San Diego, CA","San Francisco, CA","San Jose, CA",
        "San Luis Obispo, CA","Santa Cruz, CA","Santa Maria, CA","Santa Rosa, CA","Sonora, CA",
        "Stockton, CA","Susanville, CA","Truckee, CA","Ukiah, CA","Vallejo, CA","Visalia, CA",
        "Yuba City, CA"
    ],
    "msa_lat": [
        35.3733,39.7285,38.9582,41.7558,32.7920,
        40.8021,36.7378,36.3275,34.0522,36.9613,37.3022,
        37.6391,38.2975,34.1975,40.1785,40.5865,33.9806,
        38.5816,36.6777,32.7157,37.7749,37.3382,
        35.2828,36.9741,34.9530,38.4405,37.9841,
        37.9577,40.4163,39.3279,39.1502,38.1041,36.3302,
        39.1404
    ],
    "msa_lon": [
        -119.0187,-121.8375,-122.6264,-124.2026,-115.5631,
        -124.1637,-119.7871,-119.6457,-118.2437,-120.0607,-120.4820,
        -120.9969,-122.2869,-119.1771,-122.2358,-122.3917,-117.3755,
        -121.4944,-121.6555,-117.1611,-122.4194,-121.8863,
        -120.6596,-122.0308,-120.4357,-122.7144,-120.3822,
        -121.2908,-120.6530,-120.1833,-123.2078,-122.2566,-119.2921,
        -121.6169
    ]
})

print("Centroids table shape:", msa_centroids.shape)
msa_centroids.head()

# 1) Make sure every Zillow CA MSA has a centroid
missing = sorted(set(df_zillow_latest["RegionName"]) - set(msa_centroids["RegionName"]))
print("Missing centroids:", missing)

# 2) Make sure the centroid join didn’t drop anything unexpectedly
z_geo = df_zillow_latest.merge(msa_centroids, on="RegionName", how="inner")
print("df_zillow_latest:", df_zillow_latest.shape)
print("z_geo:", z_geo.shape)

# Assign each sklearn row to nearest MSA centroid, then merge Zillow
df_tmp = df_sklearn.copy()

lat = df_tmp["Latitude"].to_numpy()[:, None]
lon = df_tmp["Longitude"].to_numpy()[:, None]
msa_lat = z_geo["msa_lat"].to_numpy()[None, :]
msa_lon = z_geo["msa_lon"].to_numpy()[None, :]

dist2 = (lat - msa_lat)**2 + (lon - msa_lon)**2
nearest = dist2.argmin(axis=1)

df_tmp["msa_name"] = z_geo["RegionName"].to_numpy()[nearest]

df_merged = df_tmp.merge(
    z_geo[["RegionName", "zillow_median_price"]],
    left_on="msa_name",
    right_on="RegionName",
    how="left"
).drop(columns=["RegionName"])

print("Merged dataset shape:", df_merged.shape)
print("Missing Zillow values:", df_merged["zillow_median_price"].isna().sum())

# Guards
assert df_merged["zillow_median_price"].isna().sum() == 0
assert df_merged.isna().sum().sum() == 0

# Preview + validation summary
display(df_merged.head(10))

summary = pd.DataFrame({
    "dtype": df_merged.dtypes.astype(str),
    "missing": df_merged.isna().sum(),
    "missing_%": (df_merged.isna().mean() * 100).round(2),
})
display(summary)

display(df_merged["msa_name"].value_counts().rename_axis("msa_name").reset_index(name="n_rows"))

os.makedirs("data/processed", exist_ok=True)
df_merged.to_csv("data/processed/merged_housing.csv", index=False)

print("Merged dataset saved to data/processed/merged_housing.csv")
print(df_merged.head())

import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap

# Set default plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print("=== Dataset Overview ===")
print(f"Shape: {df_merged.shape}")
print(f"\nColumn Names:\n{df_merged.columns.tolist()}")
print("\nBasic Statistics:")
display(df_merged.describe())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# sklearn prices (1990)
axes[0].hist(df_merged["median_house_value"], bins=50, color="steelblue", edgecolor="white")
axes[0].set_title("Distribution of Median House Values (sklearn; $100k units)")
axes[0].set_xlabel("Value (× $100,000)")
axes[0].set_ylabel("Count")

# Zillow prices (current)
axes[1].hist(df_merged["zillow_median_price"], bins=50, color="coral", edgecolor="white")
axes[1].set_title("Distribution of Zillow Median Prices (latest snapshot)")
axes[1].set_xlabel("Price ($)")
axes[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_price_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_price_distribution.png")

# Feature correlation heatmap (numeric only)
numeric_cols = df_merged.select_dtypes(include=[np.number]).drop(
    columns=["Latitude", "Longitude"],
    errors="ignore"
)

corr_matrix = numeric_cols.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_correlation_heatmap.png")

# income vs price plot
# higher income neighborhoods tend to have higher house prices

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_merged,
    x="MedInc",
    y="zillow_median_price",
    alpha=0.5,
    color="steelblue"
)
plt.title("Median Income vs Zillow House Price")
plt.xlabel("Median Income (× $10,000)")
plt.ylabel("Zillow Median Price ($)")
plt.tight_layout()
plt.savefig("eda_income_vs_price.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_income_vs_price.png")

# age of the house vs price plot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df_merged,
    x="HouseAge",
    y="zillow_median_price",
    alpha=0.5,
    color="coral"
)
plt.title("House Age vs Zillow House Price")
plt.xlabel("House Age (years)")
plt.ylabel("Zillow Median Price ($)")
plt.tight_layout()
plt.savefig("eda_age_vs_price.png", dpi=150, bbox_inches="tight")
plt.show()
print("House age vs price scatter plotted")

# Prepare heatmap data [lat, lon, weight] with normalized weights
heat_df = df_merged[["Latitude", "Longitude", "zillow_median_price"]].dropna().copy()

w = heat_df["zillow_median_price"].to_numpy()
w_norm = (w - w.min()) / (w.max() - w.min())

heat_data = np.column_stack([
    heat_df["Latitude"].to_numpy(),
    heat_df["Longitude"].to_numpy(),
    w_norm
]).tolist()

# Create a base map centered on California
ca_map = folium.Map(location=[36.7783, -119.4179], zoom_start=6)

# Add heatmap layer
HeatMap(heat_data, radius=8, blur=10, max_zoom=1).add_to(ca_map)

ca_map.save("eda_ca_heatmap.html")
ca_map
     
# Top most expensive MSAs (by Zillow median price)
top_msas = (
    df_merged.groupby("msa_name")["zillow_median_price"]
    .median()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_msas.values, y=top_msas.index, palette="Reds_r")
plt.title("Top 10 Most Expensive MSAs in California (Zillow)")
plt.xlabel("Median House Price ($)")
plt.ylabel("MSA")
plt.tight_layout()
plt.savefig("eda_top_msas.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: eda_top_msas.png")

#### WEIGHTS AS THE PROFESSOR ASKED ####
# Load data
df_merged = pd.read_csv("data/processed/merged_housing.csv")
 
# Set default plot style (matching data.py)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
 
TARGET = "zillow_median_price"
 
ALL_FEATURES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup", "Latitude", "Longitude"]
 
GEO_EXCLUDED = ["MedInc", "HouseAge", "AveRooms", "AveBedrms",
                "Population", "AveOccup"]
 
 
def get_importances(X, y):
    """Return RF importance and normalized Ridge absolute coefficients."""
    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = pd.Series(rf.feature_importances_, index=X.columns)
 
    # Ridge (standardized so coefficients are comparable across features)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)
    ridge_imp = pd.Series(np.abs(ridge.coef_), index=X.columns)
    ridge_imp = ridge_imp / ridge_imp.sum()   # normalize to sum = 1
 
    return rf_imp, ridge_imp
 
 
def plot_importances(rf_imp, ridge_imp, title_suffix, filename):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # --- Random Forest ---
    rf_sorted = rf_imp.sort_values(ascending=True)
    colors_rf = ["steelblue" if f not in ("Latitude", "Longitude") else "coral"
                 for f in rf_sorted.index]
    axes[0].barh(rf_sorted.index, rf_sorted.values * 100, color=colors_rf,
                 edgecolor="white")
    axes[0].set_title(f"Random Forest feature importance\n{title_suffix}")
    axes[0].set_xlabel("Importance (%)")
    axes[0].set_ylabel("Feature")
    for i, v in enumerate(rf_sorted.values * 100):
        axes[0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
 
    # --- Ridge ---
    ridge_sorted = ridge_imp.sort_values(ascending=True)
    colors_ridge = ["steelblue" if f not in ("Latitude", "Longitude") else "coral"
                    for f in ridge_sorted.index]
    axes[1].barh(ridge_sorted.index, ridge_sorted.values * 100, color=colors_ridge,
                 edgecolor="white")
    axes[1].set_title(f"Ridge regression feature weights\n{title_suffix}")
    axes[1].set_xlabel("Normalized absolute weight (%)")
    axes[1].set_ylabel("Feature")
    for i, v in enumerate(ridge_sorted.values * 100):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)
 
    # Legend (only needed when lat/lon are present)
    if title_suffix == "including Latitude & Longitude":
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="coral", label="Geographic"),
                           Patch(facecolor="steelblue", label="Non-geographic")]
        fig.legend(handles=legend_elements, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.04), frameon=False)
 
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {filename}")
 
 
# ── Plot 1: WITH Latitude & Longitude ──────────────────────────────────────
X_all = df_merged[ALL_FEATURES]
y = df_merged[TARGET]
 
rf_all, ridge_all = get_importances(X_all, y)
 
print("=== WITH Latitude & Longitude ===")
print("\nRandom Forest importances:")
print((rf_all * 100).round(2).sort_values(ascending=False).to_string())
print("\nRidge normalized weights:")
print((ridge_all * 100).round(2).sort_values(ascending=False).to_string())
 
plot_importances(
    rf_all, ridge_all,
    title_suffix="including Latitude & Longitude",
    filename="feature_importance_with_geo.png"
)
 
# ── Plot 2: WITHOUT Latitude & Longitude ───────────────────────────────────
X_no_geo = df_merged[GEO_EXCLUDED]
 
rf_nogeo, ridge_nogeo = get_importances(X_no_geo, y)
 
print("\n=== WITHOUT Latitude & Longitude ===")
print("\nRandom Forest importances:")
print((rf_nogeo * 100).round(2).sort_values(ascending=False).to_string())
print("\nRidge normalized weights:")
print((ridge_nogeo * 100).round(2).sort_values(ascending=False).to_string())
 
plot_importances(
    rf_nogeo, ridge_nogeo,
    title_suffix="excluding Latitude & Longitude",
    filename="feature_importance_without_geo.png"
)


