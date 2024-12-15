class MultiPage:
    """
    Framework for combining multiple Streamlit applications into a single dashboard.
    """

    def __init__(self):
        self.pages = [] # Initializes the app list to store multiple applications.

    def add_page(self, title, func):
        self.pages.append({
            "title": title,
            "function": func
        })
    
    def run(self):
        import streamlit as st

        # Sidebar selector for navigation
        page = st.sidebar.selectbox(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )

        # Render the selected page
        page["function"]()