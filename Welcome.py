import streamlit as st
from PIL import Image
import pandas as pd
from import_data import f_get_Normalization
import numpy as np

# from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

st.set_page_config(
    page_title="Welcome to our Interface",
    page_icon="👋",
)

st.sidebar.success("Access interface functionality here.")

# st.title('NASH Project Interface')
st.markdown("<h2 style='text-align: center; color: black;'>NASH Project Interface</h2>", unsafe_allow_html=True)

# license_key = "For_Trialing_ag-Grid_Only-Not_For_Real_Development_Or_Production_Projects-Valid_Until-18_March_2021_[v2]_MTYxNjAyNTYwMDAwMA==948d8f51e73a17b9d78e03e12b9bf934"

st.markdown(
    """
    This interface has been designed to interact with the trained models apart of the NASH research project. 
    Move through the toggles on the sidebar to upload patient logs and generate plot predictions.
    ### Title of the Paper
    Predicting the one-year trajectory of Non-alcoholic steatohepatitis (NASH) cirrhosis patients awaiting liver transplant
    ### Contributors
    Yingji Sun, Gopika Punchhi, Madison Mussari, Sumeet Asrani, Sirisha Rambhatla, Mamatha Bhat
"""
)



