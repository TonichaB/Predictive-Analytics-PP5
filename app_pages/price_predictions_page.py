import streamlit as st
from PIL import Image

def app():
    # Convert the image to RGB and save it
    image_path = "static/images/price_prediction_banner.png"
    output_path = "static/images/price_prediction_banner_converted.png"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((400, 200))
            img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    st.image(output_path, use_column_width=True)

    # Title
    st.title("Sale Price Predictions")
    st.markdown("### Predict house prices with ease!")

    # Section for inherited houses predictions
    st.header("Predicted Prices for Inherited Houses")
    st.write("The predicted sale prices for the 4 inherited properties are displayed here.")
    st.write("**Placeholder for predicted prices table**")

    # Summed predicted price
    st.subheader("Summed Sale Price")
    st.write("The toal predicted price for all 4 inherited properties is displayed here.")
    st.write("**Placeholder for total price message.**")

    # Custom Predictions
    st.header("Custom Price Predictions")
    st.write(
        """
        Use the interactive form below to input house details and generate a predicted sale price:
        """
    )
    st.write("**Placeholder for inherited widgets and predicted price display**")

    # Visualization Placeholder
    st.subheader("Predictions Visualization")
    st.write("**Placeholder for predictions bar chart.**")