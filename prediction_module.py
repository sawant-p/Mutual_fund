import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from api_utils import call_gemini_api_for_suggestions # CORRECTED: Changed from relative to absolute import

# --- Configuration ---
FEATURE_COLUMNS = [
    'aum_funds_individual_lst', 
    'nav_funds_individual_lst', 
    'minimum_funds_individual_lst', 
    'debt_per', 
    'equity_per',
    'rating_of_funds_individual_lst'
]
TARGET_COLUMNS = ['one_year_returns', 'three_year_returns', 'five_year_returns']
CSV_FILE_NAME = "cleaned_data.xlsx - Sheet1.csv"


# --- Data Loading and Model Training ---

@st.cache_data
def load_data(file_path):
    """Loads the CSV data and performs initial cleanup."""
    try:
        # Explicitly set the encoding to 'latin1' to handle decoding errors
        df = pd.read_excel("cleaned_data.xlsx")
        
        # Clean up commas and convert essential columns to numeric, coercing errors to NaN
        all_cols_to_process = FEATURE_COLUMNS + TARGET_COLUMNS
        for col in all_cols_to_process:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=FEATURE_COLUMNS)
        
        return df

    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found. Please ensure it is uploaded or in the correct directory.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_models(df):
    """Trains a separate Random Forest Regressor model for each target column."""
    models = {}
    if df.empty:
        return models
        
    for target in TARGET_COLUMNS:
        df_target = df.dropna(subset=[target])
        
        if df_target.empty:
            st.warning(f"No valid data points found to train the model for {target}.")
            continue
        
        missing_features = [f for f in FEATURE_COLUMNS if f not in df_target.columns]
        if missing_features:
            st.error(f"Missing required features in data for model training: {missing_features}")
            continue

        X = df_target[FEATURE_COLUMNS]
        y = df_target[target]
        
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42, 
            n_jobs=-1
        )
        model.fit(X, y)
        models[target] = model
        
    return models


# --- Input Widget Helper ---

def get_input_widget(df, data_ranges, col_name, label, column_widget, input_type='number', options=None):
    """Helper function to create Streamlit input widgets with dynamic ranges."""
    r = data_ranges.get(col_name, {'min': 0.0, 'max': 100.0, 'mean': 50.0})
    
    default_value = r['mean']
    current_value = st.session_state.input_data_df.get(col_name, pd.Series([default_value])).iloc[0]
    
    # Handle select_slider's strict value requirement by rounding the mean
    if input_type == 'select':
        if options and r.get('mean') is not None:
            # Find the option closest to the mean
            rounded_value = min(options, key=lambda x: abs(x - r['mean']))
            current_value = rounded_value
    
    with column_widget:
        if input_type == 'slider':
            return st.slider(
                label, 
                min_value=r['min'], 
                max_value=r['max'], 
                value=current_value,
                step=0.1,
                key=f'input_{col_name}'
            )
        elif input_type == 'select':
            return st.select_slider(
                label, 
                options=options or [1.0, 2.0, 3.0, 4.0, 5.0],
                value=current_value,
                key=f'input_{col_name}'
            )
        else: # number_input
            return st.number_input(
                label,
                min_value=r['min'], 
                max_value=r['max'], 
                value=current_value,
                step=100.0 if r['max'] > 1000 else 1.0,
                format="%.2f",
                key=f'input_{col_name}'
            )


# --- Main Application Function for Prediction Module ---

def run_prediction_app():
    """The main function to run the prediction dashboard."""
    st.header("ML Fund Returns Prediction Dashboard")
    st.markdown("Predict 1, 3, and 5-year returns using a Random Forest Model.")

    # Load data and train models once
    df = load_data(CSV_FILE_NAME)
    models = train_models(df)

    if not models:
        st.error("Model training failed. Please verify that your CSV file contains all required columns.")
        return

    # --- 3. Interactive Input Feature ---

    st.subheader("1. Input Fund Metrics")
    
    # Calculate min/max for realistic slider ranges
    data_ranges = {}
    valid_features = [col for col in FEATURE_COLUMNS if col in df.columns]

    for col in valid_features:
        col_series = df[col].dropna()
        
        if col_series.empty:
            min_val, max_val, mean_val = 0.0, 100.0, 50.0
        else:
            min_val = col_series.min()
            max_val = col_series.max()
            mean_val = col_series.mean()
            
            if mean_val < min_val: mean_val = min_val
            if mean_val > max_val: mean_val = max_val
            
        data_ranges[col] = {'min': float(min_val), 'max': float(max_val), 'mean': float(mean_val)}


    # State initialization
    if 'input_data_df' not in st.session_state:
        initial_data = [data_ranges[col]['mean'] for col in valid_features]
        st.session_state.input_data_df = pd.DataFrame([initial_data], columns=valid_features)
        st.session_state.predictions = {}
        st.session_state.suggestions = None
    
    
    # Create input widgets
    with st.container():
        col_aum, col_nav, col_min = st.columns(3)
        input_aum = get_input_widget(df, data_ranges, 'aum_funds_individual_lst', 'AUM Funds Individual (AUM in ₹)', col_aum)
        input_nav = get_input_widget(df, data_ranges, 'nav_funds_individual_lst', 'NAV Funds Individual', col_nav, input_type='slider')
        input_min = get_input_widget(df, data_ranges, 'minimum_funds_individual_lst', 'Minimum Funds Individual (₹)', col_min)

        col_debt, col_equity, col_rating = st.columns(3)
        input_debt = get_input_widget(df, data_ranges, 'debt_per', 'Debt Percentage', col_debt, input_type='slider')
        input_equity = get_input_widget(df, data_ranges, 'equity_per', 'Equity Percentage', col_equity, input_type='slider')
        input_rating = get_input_widget(df, data_ranges, 'rating_of_funds_individual_lst', 'Fund Rating (Score)', col_rating, input_type='select', options=[1.0, 2.0, 3.0, 4.0, 5.0])

    # Reassemble the current inputs into a DataFrame for prediction
    current_input_data = {
        'aum_funds_individual_lst': input_aum, 
        'nav_funds_individual_lst': input_nav, 
        'minimum_funds_individual_lst': input_min, 
        'debt_per': input_debt, 
        'equity_per': input_equity, 
        'rating_of_funds_individual_lst': input_rating
    }
    input_df = pd.DataFrame([current_input_data])
    st.session_state.input_data_df = input_df 

    # --- 4. Predicted Returns Display ---
    st.subheader("2. Predicted Returns")
    
    col1, col3, col5 = st.columns(3)
    predicted_1y, predicted_3y, predicted_5y = None, None, None

    try:
        # Predict 1-Year Return
        if 'one_year_returns' in models:
            predicted_1y = models['one_year_returns'].predict(input_df[FEATURE_COLUMNS])[0]
            col1.metric("Predicted 1-Year Return", f"{predicted_1y:,.2f}%")
        else: col1.error("1-Year Model Unavailable")

        # Predict 3-Year Return
        if 'three_year_returns' in models:
            predicted_3y = models['three_year_returns'].predict(input_df[FEATURE_COLUMNS])[0]
            col3.metric("Predicted 3-Year Return", f"{predicted_3y:,.2f}%")
        else: col3.error("3-Year Model Unavailable")
        
        # Predict 5-Year Return
        if 'five_year_returns' in models:
            predicted_5y = models['five_year_returns'].predict(input_df[FEATURE_COLUMNS])[0]
            col5.metric("Predicted 5-Year Return", f"{predicted_5y:,.2f}%")
        else: col5.error("5-Year Model Unavailable")
        
        st.session_state.predictions = {
            '1y': predicted_1y, '3y': predicted_3y, '5y': predicted_5y,
            'debt': input_debt, 'equity': input_equity, 'rating': input_rating
        }
            
    except Exception as e:
        st.error(f"Prediction Error: One or more models failed to predict. (Debug: {e})")

    st.markdown("---")

    # --- 5. Fund Suggestion Feature (Using Gemini API) ---
    st.subheader("3. Fund Suggestions (Gemini Search)")
    st.markdown("Search for real-world funds that meet your predicted 1-Year Return and risk profile.")
    
    if st.button("Suggest Matching Funds"):
        if predicted_1y is not None:
            with st.spinner('Searching the web for fund suggestions...'):
                # Call the absolute imported function
                suggestions = call_gemini_api_for_suggestions(
                    predicted_1y, 
                    input_debt, 
                    input_equity, 
                    input_rating
                )
                st.session_state.suggestions = suggestions
        else:
            st.error("Cannot search for suggestions: 1-Year prediction is unavailable.")

    if st.session_state.suggestions:
        suggestions = st.session_state.suggestions
        if 'error' in suggestions:
            st.error(f"API Error: {suggestions['error']}")
        else:
            st.markdown(suggestions['text'])
            if suggestions.get('sources'):
                st.subheader("Source Citations")
                for i, source in enumerate(suggestions['sources']):
                    if source.get('uri') and source.get('title'):
                        st.write(f"**[{i+1}]** [{source['title']}]({source['uri']})")