import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

def app():
    # Convert the image to RGB and save it
    image_path = "static/images/feature_corr_banner.png"
    output_path = "static/images/feature_corr_banner_converted.png"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((400, 200))
            img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    st.image(output_path, use_column_width=True)

    # Title and Description
    st.title("Feature Correlation")
    st.markdown("### Analyzing how house attributes correlate with sale prices.")
    st.markdown(
        """
        ---
        This section explores the relationship between various house attributes and sale prices using correlation analysis.
        Correlation values range between -1 and 1:
        - **+1** indicates a strong positive correlation.
        - **-1** indicates a strong negative correlation.
        - **0** indicates no correlation.
        """
    )

    # Load Dataset
    dataset_path = "outputs/datasets/processed/final/x_test_final.csv"
    try:
        data = pd.read_csv(dataset_path)
        st.write(f"The dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**.")
        st.write("Sample of the processed dataset:")
        st.dataframe(data.head())
    except FileNotFoundError:
        st.error("Processed dataset file not found. Please ensure the dataset is available at the specified path.")
    except Exception as e:
        st.error(f"An error occured while loading the dataset: {e}")
    
    # Visualizing Feature Correlations
    st.header("Visualizing Feature Correaltions")
    st.markdown(
        """
        Below are some visualizations and key insights derived from the correlations between features and the sale price:
        """
    )

    # Placeholder Heatmap
    st.subheader("Heatmap of Feature Correlation")
    st.write("Heatmap visualization placeholder..")
    
    # PLaceholder Bar Chart
    st.subheader("Top Features Correlated with Sale Price")
    st.write("Placeholder bar chat visualization..")

    # Key Insights
    st.header("Key Insights")
    st.write(
        """
        Based on the analysis:
        - Features like **Overall Quality** and **GrLivArea** show the strongest positive correlation with sale price.
        - Features like the property **Age** tend to show a negative correlation with sale price.
        - Other features like **Garage Area** and **Basement Size** also contribute significantly to sale price predictions.

        These finding are integral to understanding the data and building and effective predictive model.
        """
    )

    st.markdown(
        """
        ---
        *Visualizations and anaylsis are based on the processed dataset used for training the prediction model.*
        """
    )
