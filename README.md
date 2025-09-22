# Baku House Price Prediction System

Machine learning project for predicting new house prices across 8 districts in Baku, Azerbaijan using data scraped from bina.az with an interactive Streamlit web application.

**Author:** Nigar Rustamova                                                                                                                                                                                               
**Email:** rustamovanigar11@gmail.com                                                                                                                                                                                         
**Data Source:** bina.az

## Project Overview

This project develops predictive models for house prices in Baku using web-scraped real estate data. The system provides price predictions across 8 major districts through an interactive web application with geospatial visualization capabilities.

## Districts Covered

- ABSHERON
- BINAGADI
- NARIMANOV
- NASIMI
- NIZAMI
- SEBAIL
- KHATAI
- YASAMAL

## Technology Stack

**Data Processing:** Python, Pandas, NumPy  
**Machine Learning:** Scikit-learn  
**Web Application:** Streamlit  
**Geospatial Visualization:** OpenStreetMap, GeoJSON  
**Development:** Jupyter Notebooks

## Machine Learning Models

Three algorithms implemented and compared:
- Linear Regression
- Random Forest  
- XGBoost

**Evaluation Metrics:**
- R² Score
- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)

Individual models trained separately for each district.

## Project Structure

```
├── Bina_az_final.ipynb/                    # Jupyter notebook for analysis
│   ├── Data scraping
│   ├── Data cleaning  
│   ├── EDA 
│   └── Modeling (by district)
├── models/                      # Trained models (one per district)
├── bina_az_app.py              # Streamlit application
└── README.md                   # Documentation
```

## Key Features

**Price Prediction**
- District-specific predictions
- Interactive map with polygon boundaries
- Real-time price estimation

**Geospatial Integration**
- OpenStreetMap base maps
- GeoJSON district boundaries
- Visual price distribution analysis

## Usage

The Streamlit application provides:
- Interactive price prediction interface
- District selection and property feature input
- Geospatial visualization with district boundaries
- Model performance metrics and insights

## Technical Implementation

- Separate trained models for each district optimize local market predictions
- OpenStreetMap integration provides interactive geospatial context
- GeoJSON polygon boundaries enable precise district visualization
- Streamlit framework delivers responsive user interface

## Streamlit App Video
[![House Price Prediction Streamlit App Demo](https://img.youtube.com/vi/0X-EyD6Y6fg/maxresdefault.jpg)](https://www.youtube.com/watch?v=0X-EyD6Y6fg)
