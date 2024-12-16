import streamlit as st

def app():
    # Title and Introduction
    st.title("Sale Price Predictions Summary")
    st.markdown("### Welcome to the Sale Price Predictions Dashboard!")
    st.markdown(
        """
        This dashboard is designed to provide insights and predictive analytics for house sale prices in Ames, Iowa.
        It combines statistical analysis and machine learning techniques to help make informed decisions.
        """
    )

    # Overview of the dashboard pages
    st.markdown("### Dashboard Overview")
    st.markdown(
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
    st.markdown("### Project Goals and Business Requirements")
    st.markdown(
        """
        The primary goals of this project are:
        - **Understand correlations**: Analyze how various house attributes impact sale prices.
        - **Predict house prices**: Use machine learning models to estimate the sale prices of inherited properties and other houses in Ames, Iowa.
        ** Deliver actionable insights**: Provide visualizations and summaries to inform decision-making.
        """
    )

    # Dataset details
    st.markdown("### Dataset Overview")
    st.markdown(
        """
        The dataset used in this project contains detailed information about house attributes and their respective sale prices.
        Below is a summary of the dataset:
        """
    )

    # Placeholder for dataset statistics
    dataset_summary = {
        "Number of Rows": 2930,
        "Number of Columns": 80,
        "Target Variable": "SalePrice",
        "Key Features": ["OverallQual", "GrLivArea", "GarageArea", "TotalBsmtSF"],
    }

    st.write(dataset_summary)

    # Dataset content guidelines
    st.markdown("### Dataset Content Guidelines")
    st.markdown(
        """
        When providing new data for predictions or analysis, ensure that:
        - All required columns are included in the dataset.
        - Missing values are handled appropriately before submission.
        - Data types for each column align with the expected format (e.g., numerical, categorical).
        - Values for categorical features match the categories used during training.
        """
    )

    # README.md reference
    st.markdown("### Additional Information")
    st.markdown(
        """
        For more details about this project, including methodolofy and setup instructions, please refer to the [README.md file](https://github.com/TonichaB/Predictive-Analytics-PP5/blob/main/README.md).
        """
    )