import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np
import joblib
import datetime

def app():
    # Convert the image to RGB and save it
    image_path = "static/images/price_prediction_banner.png"
    output_path = "static/images/price_prediction_banner_converted.png"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((600, 200))
            img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    st.image(output_path, use_container_width=True)

    # Title
    st.title("Sale Price Predictions")
    st.markdown("### Predict house prices with ease!")
    st.markdown("---")
    # Section for inherited houses predictions
    st.header("Predicted Prices for Inherited Houses")

    try:
        # Load predicted prices for inherited houses
        display_ready_path = "outputs/datasets/processed/final/inherited_properties_display_ready.csv"
        display_ready_dataset = pd.read_csv(display_ready_path)
        
        # Create tabs for each property
        tabs = st.tabs([f"Property {i + 1}" for i in range(len(display_ready_dataset))])

        # Define attribute units
        attribute_units = {
            "LotFrontage": "ft",
            "LotArea": "sqft",
            "OpenPorchSF": "sqft",
            "MasVnrArea": "sqft",
            "BsmtFinSF1": "sqft",
            "GrLivArea": "sqft",
            "1stFlrSF": "sqft",
            "2ndFlrSF": "sqft",
            "BsmtUnfSF": "sqft",
            "GarageArea": "sqft",
            "GarageYrBlt": None,
            "YearBuilt": None,
            "YearRemodAdd": None,
            "BedroomAbvGr": "bedrooms",
            "OverallCond": "/10",
            "OverallQual": "/10",
            "Age": "years",
            "LivingLotRatio": ":1",
            "FinishedBsmtRatio": None,
            "OverallScore": "/100",
            "HasPorch": None,
        }

        for i, tab in enumerate(tabs):
            with tab:
                # Get property attributes
                property_data = display_ready_dataset.iloc[i].to_dict()
                predicted_price = property_data.pop("PredictedPrice")
                property_id = property_data.pop("Property_ID")

                # Split attributes into 3 columns
                attributes = list(property_data.items())
                third = len(attributes) // 3
                first_third = attributes[:third]
                second_third = attributes[third:2 * third]
                final_third = attributes[2 * third:]

                # Display property card
                st.markdown(f"""
                <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #4CAF50;">Property {i + 1}</h3>
                    <p><b>Predicted Sale Price:</b> £{predicted_price:,.2f}</p>
                    <hr>
                    <div style="display: flex; justify-content: space-between;">
                        <div style="width: 33%;">
                            {"".join([f"<p><b>{k}:</b> {int(v) if isinstance(v, float) and v.is_integer() else v}"
                                f"{' ' + attribute_units[k] if attribute_units[k] else ''}</p>" for k, v in first_third])}
                        </div>
                        <div style="width: 33%;">
                            {"".join([f"<p><b>{k}:</b> {int(v) if isinstance(v, float) and v.is_integer() else v}"
                                f"{' ' + attribute_units[k] if attribute_units[k] else ''}</p>" for k, v in second_third])}
                        </div>
                        <div style="width: 33%;">
                            {"".join([f"<p><b>{k}:</b> {v * 100:.2f}%" if k == 'FinishedBsmtRatio' else f"<p><b>{k}:</b> {int(v) if isinstance(v, float) and v.is_integer() else v}"
                                f"{' ' + attribute_units[k] if attribute_units[k] else ''}</p>" for k, v in final_third])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Calculate and display the summed predicted prices
        total_price = display_ready_dataset["PredictedPrice"].sum()
        st.subheader("Summed Sale Price")
        st.write(f"The total predicted sale price for all 4 properties is **£{total_price:,.2f}**.")
    
    except FileNotFoundError:
        st.error("Inherited house predictions file not found.")
    except Exception as e:
        st.error(f"An error occured while loading the predicted prices: {e}")

    st.markdown("---")

    # Custom Predictions
    st.header("Custom Price Predictions")
    st.write(
        """
        Use the interactive form below to input house details and generate a predicted sale price:
        """
    )
    
    with st.form(key="custom_prediction_form"):
        st.subheader("Enter Property Details")

        current_year = datetime.datetime.now().year
        
        # Input fields for user to fill in
        lot_frontage = st.number_input("Lot Frontage (ft)", min_value=0, step=1)
        lot_area = st.number_input("Lot Area (sqft)", min_value=0, step=1)
        open_porch_sf = st.number_input("Open Porch Surface Area (sqft)", min_value=0, step=1)
        mas_vnr_area = st.number_input("Masonry Veneer Area (sqft)", min_value=0, step=1)
        bsmt_fin_sf1 = st.number_input("Finished Basement Surface Area (sqft)", min_value=0, step=1)
        total_bsmt_sf = st.number_input("Total Basement Surface Area (sqft)", min_value=0, step=1)
        year_built = st.number_input("Year Built", min_value=1800, max_value=current_year, step=1)
        gr_liv_area = st.number_input("Above Ground Living Area (sqft)", min_value=0, step=1)
        year_remod_add = st.number_input("Year Remodeled", min_value=1800, max_value=current_year, step=1)
        overall_qual = st.slider("Overall Quality (/10)", min_value=1, max_value=10, step=1)
        overall_cond = st.slider("Overall Condition (/10)", min_value=1, max_value=10, step=1)
        has_porch = st.radio("Has Porch?", options=["Yes", "No"])
        bedroom_abv_gr = st.number_input("Number of Bedrooms", min_value=0, step=1)
        second_flr_sf = st.number_input("Second Floor Surface Area (sqft)", min_value=0, step=0)
        bsmt_unf_sf = st.number_input("Unfinished Basement Surface Area (sqft)", min_value=0, step=1)
        garage_area = st.number_input("Garage Area (sqft)", min_value=0, step=1)
        garage_yr_built = st.number_input("Year Garage Built", min_value=1800, max_value=current_year)
        first_flr_sf = st.number_input("First Floor Surface Area (sqft)", min_value=0, step=1)

        # Submit button
        submitted = st.form_submit_button("Predict Price!")

        if submitted:
            # Convert inputs to the format expected by the model
            inputs = {
                "LotFrontage": lot_frontage,
                "LotArea": lot_area,
                "OpenPorchSF": open_porch_sf,
                "MasVnrArea": mas_vnr_area,
                "BsmtFinSF1": bsmt_fin_sf1,
                "TotalBsmtSF": total_bsmt_sf,
                "YearBuilt": year_built,
                "GrLivArea": gr_liv_area,
                "YearRemodAdd": year_remod_add,
                "OverallQual": overall_qual,
                "OverallCond": overall_cond,
                "HasPorch": 1 if has_porch == "Yes" else 0,
                "BedroomAbvGr": bedroom_abv_gr,
                "2ndFlrSF": second_flr_sf,
                "BsmtUnfSF": second_flr_sf,
                "GarageArea": garage_area,
                "GarageYrBlt": garage_yr_built,
                "1stFlrSF": first_flr_sf,
            }

            # Convert to DataFrame
            input_df = pd.DataFrame([inputs])

            # Load the saved pipeline
            pipeline_path = "outputs/pipelines/random_forest_pipeline_general.pkl"
            try:
                pipeline = joblib.load(pipeline_path)

                # Predict the sale price
                log_predicted_price = pipeline.predict(input_df)[0]
                predicted_price = np.expm1(log_predicted_price)
                st.success(f"The predicted sale price for the entered property is: **£{predicted_price:,.2f}**")
            except FileNotFoundError:
                st.error("Prediction pipeline not found. Please ensure the pipeline file exists.")
            except Exception as e:
                st.error(f"An error occured during prediction: {e}")
