import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

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
    
    st.image(output_path, use_container_width=True)

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
    saleprice_path = "outputs/datasets/processed/final/y_test_final.csv"
    try:
        data = pd.read_csv(dataset_path)
        target = pd.read_csv(saleprice_path)
        processed_data = pd.concat([data, target], axis=1)

        st.write(f"The dataset contains **{processed_data.shape[0]} rows** and **{processed_data.shape[1]} columns**.")
        st.write("Sample of the processed dataset:")
        st.dataframe(processed_data.head())

        # Calculate correlation matrix
        correlation_matrix = processed_data.corr()

        # Toggle between heatmap views
        heatmap_view = st.radio(
            "Select Heatmap View",
            options=["Standard Heatmap", "Customizable Heatmap"],
            help="Choose between a standard heatmap showing all correlations or a customizable heatmap to view specific features."
        )
        # Standard Heatmap
        if heatmap_view == "Standard Heatmap":
            st.subheader("Heatmap of Feature Correlation")
            with st.expander("ℹ️ About this visualisation", expanded=False):
                st.write(
                    """
                    The heatmap provides a visual representation of the correlation between different features and the sale price.
                    Darker colors represent stronger positive correlations, while lighter colors indicate weaker or negative correlations.
                    Use this to identify which features contribute most significantly to predicting sale prices.
                    """
                )
            corr_matrix = processed_data.corr()

            fig = px.imshow(
                corr_matrix,
                color_continuous_scale="Viridis",
                title="Feature Correlation Heatmap",
                labels={"color": "Correlation"},
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                zmin=-1,
                zmax=1,
                text_auto=".2f",
            )
            fig.update_traces(hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>")
            st.plotly_chart(fig, use_container_width=True)

        # Customizable Heatmap
        elif heatmap_view == "Customizable Heatmap":
            st.subheader("Customizable Heatmap of Feature Correlations")
            selected_features = st.multiselect(
                "Select features to include in the heatmap:",
                options=processed_data.columns,
                default=["LogSalePrice", "num__OverallQual", "num__GrLivArea"],
                help="Choose features to focus on specific correlations."
            )
            if len(selected_features) > 1:
                selected_corr = processed_data[selected_features].corr()

                fig = px.imshow(
                    selected_corr,
                    color_continuous_scale="Viridis",
                    title="Customizable Feature Correlation Heatmap",
                    labels={"color": "Correlation"},
                    x=selected_corr.columns,
                    y=selected_corr.columns,
                    zmin=-1,
                    zmax=1,
                    text_auto=".2f",
                )
                fig.update_traces(hovertemplate="%{x}<br>%{y}<br>Correlation: %{z:.2f}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Please select at least two features to display the heapmap.")

        # Top Correlated Features - Bar Chart
        st.subheader("Top Features Correlated with Sale Price")
        top_n = st.slider("Select number of top features", 5, 15, 10)

        top_corr = corr_with_target.head(top_n)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.barplot(x=top_corr.values, y=top_corr.index, palette="viridis", ax=ax)
        plt.title("Top Features Correlated with Sale Price")
        plt.xlabel("Correlation with Sale Price")
        plt.ylabel("Features")
        st.pyplot(fig)

        # Pair Plot for Top Correlated Features
        st.subheader("Pair Plot of Top Correlated Features")
        st.write("Pairwise scatterplots of the top correlated features with sale price.")
        pairplot_features = ["LogSalePrice"] + top_corr.index.tolist()[:4]
        fig = sns.pairplot(processed_data[pairplot_features], corner=True, diag_kind="kda")
        st.pyplot(fig)

        # Scatter Plot for Top Positive and Negative Correlations
        st.subheader("Scatter Plots for Key Features")
        st.write("Visualizing relationships between selected features and the target variable.")

        # Top positively correlated feature
        top_positive = top_corr.index[0]
        fig, ax = plt.subplots()
        sns.scatterplot(x=processed_data[top_positive], y=processed_data["LogSalePrice"], ax=ax)
        plt.title(f"Sale Price vs {top_positive}")
        plt.xlabel(top_positive)
        plt.ylabel("Sale Price")
        st.pyplot(fig)

        # Top negatively correlated feature
        top_negative = top_corr.index[-1]
        fig, ax = plt.subplots()
        sns.scatterplot(x=processed_data[top_negative], y=processed_data["LogSalePrice"], ax=ax)
        plt.title(f"Sale Price vs {top_negative}")
        plt.xlabel(top_negative)
        plt.ylabel("Sale Price")
        st.pyplot(fig)

        # Key Insights
        st.header("Key Insights")
        st.write(
            """
            Based on the correlation analysis:
            - **Overall Quality (OveralQual)** shows the strongest positive correlation with sale prices, indicating that higher overall quality significantly increases house value.
            - **Ground Floor Living Area (GrLivArea)** is also strongly correlated, highlighting the importance of usable living space in determining sale price.
            - **1st Floor Surface Area (1stFlrSF)** and the **Overall Score (OverallScore)** further reinforce that larger, high-quality properties command higher prices.
            - **Garage Area (Garage Area)** and **Age of Property (Age)** contribute meaningfully to predicting sale prices.

            Moderate and weaker correlations:
            - Features like **Masonry Veneer Area (MasVnrArea)**, **Open Porch Size (OpenPorchSF)**, and **Porch Presence (HasPorch)** add moderate value to a property.
            - **Lot Frontage (LotFrontage)**, **Lot Area size (LotArea), and **Basement Finished Area (BsmtFinSF1)** have weaker but still positive relationships with sale price.

            Features with negative correlations:
            - **Year of build (YearBuilt)** and **Year Remodel Added (YearRemodAdd)** indicate that older houses, or those in need of remodeling tend to have lower sale prices.
            - The **Number of Bedrooms Above Ground (BedroomAbvGr)** and **Overall Property Condition (OverallCond)** show limited or weak correlation with sale price.

            ### Summary
            - **Key Drivers:** Overall quality, living area size, garage area, and ground floor size are the strongest predictors of sale price.
            - **Negative Impact:** Older houses and houses requiring remodeling tend to reduce in value.
            - **Moderate Influences:** Porch size, masonry veneer, and lot-related features moderately affect sale prices.
            """
        )

    except FileNotFoundError:
        st.error("Processed dataset file not found. Please ensure the dataset is available at the specified path.")
    except Exception as e:
        st.error(f"An error occured while loading the dataset: {e}")
    
        st.markdown(
        """
        ---
        *The feature correlation analysis uses the processed dataset that informed the final predictive model.*
        """
    )
