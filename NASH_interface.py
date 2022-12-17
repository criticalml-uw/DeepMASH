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


uploaded_file = st.file_uploader('Choose a file containing patient logs: ')
if uploaded_file:
   #read xls or xlsx
    # data = pd.read_excel(uploaded_file)
    data = pd.read_csv(uploaded_file)
    # st.dataframe(data)
    st.table(data)
else:
    st.warning('You need to upload a csv file.')


