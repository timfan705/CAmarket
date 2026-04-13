# California Median Housing Price Predictor (CAmarket)

---

## Team Members

| Name | Email | Student ID |
|------|-------|------------|
| Timothy Fan | timothy.fan@sjsu.edu | 016844984 |
| Vy Tran | vylophuong.tran@sjsu.edu | 018063851 |
| Jeffery De Prima | jeffery.deprima@sjsu.edu | 012157847 |

---

## Problem Statement

Housing prices in California vary significantly based on location, demographics, and property characteristics. Our project aims to develop an interactive machine learning application that predicts median housing prices in California using tract-level socioeconomic indicators (median income, population), housing characteristics (rooms, bedrooms, housing age), and geographic features (latitude, longitude, ocean proximity).

The application will provide users with an intuitive interface where they can select a location on a California map and adjust housing-specific attributes to receive real-time median house value predictions powered by machine learning.

---

## Dataset and Data Source

**Primary Dataset**: California Housing Dataset (scikit-learn built-in)

**Source**: `sklearn.datasets.fetch_california_housing()`  
**Original Data**: 1990 U.S. Census Bureau

**Dataset Details**:
- **Size**: 20,640 observations (California census block groups)
- **Features**: 8 input variables
  - `MedInc`: Median income (in $10,000s)
  - `HouseAge`: Median house age (years)
  - `AveRooms`: Average rooms per household
  - `AveBedrms`: Average bedrooms per household
  - `Population`: Block group population
  - `AveOccup`: Average household occupancy
  - `Latitude`: Latitude coordinate
  - `Longitude`: Longitude coordinate
- **Target**: `MedHouseVal`: Median house value (in $100,000s)

---

## Planned Model/System Approach

### System Architecture

```
┌─────────────────────────────────────┐
│  Google Colab Notebook              │
│  - Data exploration                 │
│  - Model training                   │
│  - Model evaluation                 │
│  - Export: housing_model.pkl        │
└──────────────┬──────────────────────┘
               │
               ▼ (Download model file)
┌─────────────────────────────────────┐
│  GitHub Repository                  │
│  - housing_model.pkl                │
│  - main.py (Streamlit app)          │
│  - requirements.txt                 │
└──────────────┬──────────────────────┘
               │
               ▼ (Deploy)
┌─────────────────────────────────────┐
│  Streamlit Cloud / Local            │
│  Interactive Web App                │
│  Users interact with predictions    │
└─────────────────────────────────────┘
```

### Technical Approach

**1. Data Loading and Exploration**
   - Load dataset using `fetch_california_housing(as_frame=True)`
   - Generate statistical summaries and correlation matrix
   - Create visualizations: geographic distribution, price distribution, correlation heatmap

**2. Feature Engineering**
   - Derive `ocean_proximity` categorical feature from latitude/longitude coordinates
   - Create `bedrooms_per_room` ratio feature
   - One-hot encode `ocean_proximity` variable

**3. Data Preparation**
   - Train/test split (80/20) with random_state=42
   - Feature scaling using StandardScaler
   - Prepare feature matrix with correct column ordering

**4. Model Training and Selection**
   - Train multiple regression models:
     - Linear Regression
     - Ridge Regression
     - Lasso Regression
     - Random Forest Regressor
     - Gradient Boosting Regressor
   - Evaluate using RMSE, MAE, and R-squared metrics
   - Select best performing model based on lowest RMSE

**5. Model Export**
   - Package trained model, scaler, and feature names into dictionary
   - Serialize using pickle as `housing_model.pkl`
   - Download from Colab for deployment

**6. Web Application**
   - Interactive California map using Folium
   - User input controls: sliders for location, median income, house age, rooms, bedrooms, population, occupancy
   - Load saved model and scaler
   - Generate predictions based on user inputs
   - Display predicted median house value and comparison visualizations

**Technologies**: Python, scikit-learn, pandas, numpy, Streamlit, Folium, Plotly

---

## Current Implementation Progress

**Completed**
- Project planning and architecture design
- Dataset selection and source identification
- GitHub repository structure planning

**Current Status**: Early development phase, focusing on model training in Google Colab notebook.

---
```
