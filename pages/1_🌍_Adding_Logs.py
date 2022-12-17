import streamlit as st
from PIL import Image
import pandas as pd
from import_data import f_get_Normalization
import numpy as np

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

st.set_page_config(page_title="Uploading Patient Logs", page_icon="🌍")

# st.title('NASH Project Interface')
st.markdown("<h2 style='text-align: center; color: black;'>Uploading Patient Logs</h2>", unsafe_allow_html=True)

license_key = "For_Trialing_ag-Grid_Only-Not_For_Real_Development_Or_Production_Projects-Valid_Until-18_March_2021_[v2]_MTYxNjAyNTYwMDAwMA==948d8f51e73a17b9d78e03e12b9bf934"





uploaded_file = st.file_uploader('Choose a file containing patient logs: ')
if uploaded_file:
#read xls or xlsx
    data=pd.read_csv(uploaded_file)
    ag = AgGrid(data, key='grid1', editable=True, reloadData = False, rowSelection="multiple", suppressRowClickSelection= True, enableRangeSelection=True, enable_enterprise_modules=True, license_key=license_key)

else:
    st.warning('You need to upload file in xlsx or xls format.')




