import os
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st

root_folder = Path.cwd()
if str(root_folder) not in sys.path:
    sys.path.insert(0, str(root_folder))

from src.logger import logging
from src.exception import CustomException

st.set_page_config(layout="centered")
st.title("ATM Withdrawal Volume Predictor")

def load_saved_files():
    try:
        with open(os.path.join("artifacts", "preprocessor.pkl"), "rb") as f:
            prep = pickle.load(f)
        with open(os.path.join("artifacts", "model.pkl"), "rb") as f:
            mdl = pickle.load(f)
        return prep, mdl
    except Exception as e:
        st.error("Missing model files in artifacts folder.")
        return None, None

preprocessor, model = load_saved_files()

if preprocessor and model:
    
    # Define standard weekday order
    days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        transaction_date = st.date_input("Transaction Calendar Date", min_value=pd.to_datetime("2011-01-01"), max_value=pd.to_datetime("2028-12-31"))
    with row2_col2:
        working_day = st.selectbox("Operational Work Day Status (W=Working, H=Holiday)", ["W", "H"])

    # Calculate exact weekday based on calendar input to prevent alignment mismatch
    calculated_day_index = transaction_date.weekday()  # Monday is 0, Sunday is 6

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        atm_name = st.selectbox("ATM Branch Identifier", ["ATM A", "ATM B", "ATM C"])
    with row1_col2:
        # Dynamic index auto-aligns the dropdown with the calendar date selected
        weekday = st.selectbox("Day of the Week", days_list, index=calculated_day_index)

    row3_col1, row3_col2 = st.columns(2)
    with row3_col1:
        festival_religion = st.selectbox("Festival Calendar Type (NH=No Holiday)", ["NH", "H", "M", "C"])
    with row3_col2:
        holiday_sequence = st.selectbox("Holiday Sequence Matrix Pattern", ["WW", "HH", "WH", "HW"])

    row4_col1, row4_col2 = st.columns(2)
    with row4_col1:
        no_xyz_withdrawals = st.number_input("Number of XYZ Card Withdrawals", min_value=0, value=45)
    with row4_col2:
        total_withdrawals = st.number_input("Total Transactions / Withdrawals Volume", min_value=1, value=100)

    is_valid = True
    if no_xyz_withdrawals > total_withdrawals:
        st.error("Card specific operations cannot exceed total withdrawal volumes.")
        is_valid = False

    if st.button("Calculate Forecasted Demand", disabled=not is_valid):
        try:
            year = transaction_date.year
            month = transaction_date.month
            day = transaction_date.day
            
            xyz_card_share = (no_xyz_withdrawals / total_withdrawals) * 100
            
            raw_data = {
                'Year': [year],
                'Month': [month],
                'Day': [day],
                'XYZ Card Share %': [xyz_card_share],
                'ATM Name': [atm_name.strip().title()],
                'Weekday': [weekday.strip().title()],
                'Festival Religion': [festival_religion.strip().title()],
                'Working Day': [working_day.strip().title()],
                'Holiday Sequence': [holiday_sequence.strip().title()]
            }
            
            input_df = pd.DataFrame(raw_data)
            
            features = preprocessor.transform(input_df)
            if hasattr(features, "toarray"):
                features = features.toarray()
                
            prediction_array = model.predict(features)
            final_prediction = float(prediction_array[0])
            
            # Currency Rounding Adjustments
            rounded_dollar = int(np.round(final_prediction))
            rounded_vault_cash = int(np.ceil(final_prediction / 20.0) * 20) # Nearest $20 note for physical loading
            
            st.success(f"**Predicted Cash Requirement:** ${rounded_dollar:,} USD")
            st.info(f"**Suggested Vault Order (\$20 Notes):** ${rounded_vault_cash:,} USD")
            
        except Exception as prediction_error:
            raise CustomException(prediction_error, sys)
else:
    st.warning("Waiting for pipeline artifacts.")
