import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
import numpy as np

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
    st.write("**Placeholder for inherited widgets and predicted price display**")
