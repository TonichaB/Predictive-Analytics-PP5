import streamlit as st
from app_pages.multi_page import MultiPage
from app_pages import project_summary, feature_correlation, price_predictions_page, hypothesis_validation, technical_summary

# Set up the app
app = MultiPage()

# Configure the page title and icon
st.set_page_config(page_title="Property Value Analytics Dashboard", page_icon="🏠")

# Add app pages
app.add_page("Project Summary", project_summary.app)
app.add_page("Feature Correlation", feature_correlation.app)
app.add_page("Price Predictions Page", price_predictions_page.app)
app.add_page("Hypothesis Validation", hypothesis_validation.app)
app.add_page("Technical Summary", technical_summary.app)

# Run the app
app.run()