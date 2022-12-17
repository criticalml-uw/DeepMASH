import streamlit as st
from PIL import Image
import pandas as pd
# image = Image.open('practice_image.jpeg')

# st.title('NASH Project Interface')
st.markdown("<h2 style='text-align: center; color: white;'>NASH Project Interface</h2>", unsafe_allow_html=True)
# st.image(image)
# st.header('Enter the characteristics of the patient:')
# age = st.number_input('Age:', min_value=0, max_value=100)
# gender = st.selectbox('Gender:', ['Female', 'Male', 'Non-Binary', 'Other'])

patient_data = st.file_uploader('Choose a file containing patient logs: ')
if patient_data:
   #read xls or xlsx
    # data = pd.read_excel(uploaded_file)
    data = pd.read_csv(patient_data)
    # st.dataframe(data)
    # st.table(data)
    st.dataframe(patient_data, use_container_width=True)
else:
    st.warning('You need to upload a csv file.')
    
