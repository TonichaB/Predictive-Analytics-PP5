import streamlit as st
import pandas as pd
from PIL import Image

def app():
    # Dashboard Banner
    # Convert the image to RGB and save it
    image_path = "static/images/banner_image.png"
    output_path = "static/images/banner_image_converted.png"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    st.image(output_path, use_column_width=True)

    # Title and Introduction
    st.title("Sale Price Predictions Summary")
    st.markdown("### Welcome to the Sale Price Predictions Dashboard!")
    st.markdown(
        """
        This dashboard is designed to provide insights and predictive analytics for house sale prices in Ames, Iowa.
        It combines statistical analysis and machine learning techniques to help make informed decisions about real estate investments and property sales.
        """
    )

    # Overview of the dashboard pages
    st.header("Dashboard Overview")
    st.write(
        """
        This dashboard contains the following sections:
        - **Sale Price Predictions Summary**: You are here! This is the introduction page for the project containing key details about the dataset and business goals.
        - **Feature Correlation**: Here you will find an analysis of the features most strongly correlated with house sale prices.
        - **Sale Price Predictions**: A display of predicted prices for the inherited properties and interactive price predictions for custom inputs.
        - **Hypothesis Validation**: Exploration of project hypothesis and the validation steps taken.
        - **Technical Summary**: Overview of the model performance, pipeline steps, and other technical details.
        """
    )

    # Project Goals / Business Requirements
    st.header("Project Goals and Business Requirements")
    st.write(
        """
        The primary goals of this project are:
        - **Understand correlations**: Analyze how various house attributes impact sale prices.
        - **Predict house prices**: Use machine learning models to estimate the sale prices of inherited properties and other houses in Ames, Iowa.
        - ** Deliver actionable insights**: Provide visualizations and summaries to inform decision-making.
        """
    )

    # Dataset details
    st.header("Dataset Overview")
    st.write(
        """
        The dataset used in this project contains detailed information about house attributes and their respective sale prices. It has been sourced from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data).
        Below is a summary of the dataset:
        """
    )

    # Load and Display Dataset Information
    dataset_path = "outputs/datasets/raw/house_prices_records.csv"
    try:
        data = pd.read_csv(dataset_path)
        st.subheader("Dataset Overview")
        st.write(f"The dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**.")
        st.write("Sample of the dataset:")
        st.dataframe(data.head())

        # Display Dataset Columns
        st.subheader("Dataset Columns")
        column_info = pd.DataFrame({
            "Data Type": data.dtypes.astype(str)
        })
        st.dataframe(column_info)
    except FileNotFoundError:
        st.error("Dataset file not found. Please ensure the dataset is available at the specified path.")
    except Exception as e:
        st.error(f"An error occured: {e}")

    # README.md reference
    st.header("Additional Information")
    st.write(
        """
        For more details about this project, including methodology and setup instructions, please refer to the [README.md file](https://github.com/TonichaB/Predictive-Analytics-PP5/blob/main/README.md).
        """
    )

    # Footer note
    st.markdown(
        """
        ---
        *We welcome your feedback and suggestions to improve this dashboard. Please contact us for more information.*
        """
    )