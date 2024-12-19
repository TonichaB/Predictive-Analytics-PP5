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
        inherited_prices_path = "outputs/predictions/inherited_predictions.csv"
        processed_dataset_path = "outputs/datasets/processed/final/inherited_properties_processed.csv"

        inherited_prices = pd.read_csv(inherited_prices_path)
        processed_dataset = pd.read_csv(processed_dataset_path)

        # Reverse log transformation for all columns in processed dataset
        reversed_processed_dataset = processed_dataset.apply(np.exp).round(2)

        # Filter for Random Forest model predictions
        rf_predictions = inherited_prices[inherited_prices["Model"] == "Random Forest"].copy()

        # Reverse the log transformation for predicted prices
        rf_predictions["PredictedPrice"] = rf_predictions["Actual LogSalePrice"].apply(lambda x: round(np.exp(x), 2))
        
        # Create tabs for each property
        tabs = st.tabs([f"Property {i + 1}" for i in range(len(rf_predictions))])

        for i, tab in enumerate(tabs):
            with tab:
                # Get property attributes
                property_data = reversed_processed_dataset.iloc[i].to_dict()
                predicted_price = rf_predictions.iloc[i]["PredictedPrice"]

                # Display property card
                st.markdown(f"""
                <div style="border: 1px solid #e6e6e6; padding: 20px; border-radius: 10px;">
                    <h3 style="color: #4CAF50;">Property {i + 1}</h3>
                    <p><b>Predicted Sale Price:</b> £{predicted_price:,.2f}</p>
                    <hr>
                    <div style="display: flex; justify-content: space-between;">
                        <div style="width: 48%;">
                            {"".join([f"<p><b>{k}</b> {v:,.2f}</p>" for k, v in list(property_data.items())[:len(property_data)//2]])}
                        </div>
                        <div style="width: 48%;">
                            {"".join([f"<p><b>{k}:</b> {v:,.2f}</p>" for k, v in list(property_data.items())[:len(property_data)//2:]])}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Calculate and display the summed predicted prices
        total_price = rf_predictions["PredictedPrice"].sum()
        st.subheader("Summed Sale Price")
        st.write(f"The total predicted sale price for all 4 properties is **${total_price:,.2f}**.")
    
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

    st.markdown("---")
    
    # Visualization Placeholder
    st.subheader("Predictions Visualization")
    st.write("**Placeholder for predictions bar chart.**")