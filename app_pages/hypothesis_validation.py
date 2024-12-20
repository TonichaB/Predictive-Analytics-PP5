import streamlit as st
from PIL import Image

def app():
    # Convert the image to RGB and save it
    image_path = "static/images/hypothesis_banner.png"
    output_path = "static/images/hypothesis_banner_converted.png"

    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize((400, 200))
            img.save(output_path, format="PNG")
    except Exception as e:
        print(f"Error processing image: {e}")
    
    st.image(output_path, use_container_width=True)

    st.title("Hypothesis Validation")
    st.markdown("### Testing the project hypothesis.")

    # Hypothesis Introduction
    st.header("Project Hypothesis")
    st.write(
        """
        This project seeks to validate the following hypothesis regarding house sale prices in Ames, Iowa.

        1. **Attribute Correlation**
            - Attributes such as overall quality, living area size, and garage area will have strong positive correlations with house sale prices.
            - Year of construction and basement finish quality will moderately impact sale prices.
        2. **Predictive Model**
            - A regression-based machine learning model trained on the house_prices_records.csv dataset will achieve an R2 score of at least 0.75 when tested on unseen data, meeting the client's performance criteria.

        3. **Inherited Properties**
            - The trained regression model will predict sale prices for the four inherited properties that align with the Ames market trends.
        """
    )

    # Hypothesis 1
    st.subheader("Hypothesis 1: Attribute Correlation")
    st.write(
        """
        Hypothesis: Certain house attributes, such as overall quality, living area size, and garage area, will have strong positive correlations with house sale prices. Other attributes, such as year of construction and basement finish quality, will have a moderate impact.

        Validation Steps:
        - Analyzed correlation between house attributes and sale prices, in the processed dataset.
        - Visualized correlations using heatmaps and scatter plots.
        """
    )
    
    # Hypothesis 2
    st.subheader("Hypothesis 2: Predictive Model")
    st.write(
        """
        Hypothesis: A regression-based machine learning model will achieve an R2 score of at least 0.75 when tested on unseen data.

        Validation Steps:
        - Trained multiple regression models and evaluated their performance.
        - Calculated R2 score and compared against the benchmark.
        """
    )
    st.write("**Placeholder for metrics or findings:**")
    st.write("**Placeholder for R2 visualization or table**")

    # Hypothesis 3
    st.subheader("Hypothesis 3: Inherited Properties")
    st.write(
        """
        Hypothesis: The trained regression model will predict sale prices for the four inherited properties that align with Ames market trends.

        Validation Steps:
        - Predicted sale prices for the inherited properties.
        - Compared predictions against market trends and expectations.
        """
    )
    st.write("**Placeholder for predictions or findings:**")
    st.write("**Placeholder for table or visualizations of inherited properties**")

    # Conclusion
    st.header("Conclusion")
    st.write(
        """
        The above validation provide insights into the significance of each hypothesis. These findings inform the predictive model and contribute to actionable insights for the client.
        """
    )