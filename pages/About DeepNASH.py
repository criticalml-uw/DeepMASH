import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="About DeepNASH",
    page_icon="👩‍💻",
    menu_items={
    # 'Get Help': 'https://www.extremelycoolapp.com/help',
    'Report a bug': "https://github.com/criticalml-uw/DeepNASH/issues/new",
    # 'About': "# A header introducint the app
    }
)

st.title("DeepNASH Project")

st.markdown(
    """
    This is web app provides an interactive visualization of the NASH project. 
    See the sidebar to upload patient data, generate predictions and view graphical outcome.
    
    If you would like more information, check out [our code](https://github.com/criticalml-uw/DeepNASH).
    
    #### DeepHit
    > Paper submitted for review
    Predicting the one-year death and transplant rate trajectory of Non-alcoholic steatohepatitis (NASH) cirrhosis patients awaiting liver transplant
    """
)

# image_death = Image.open('feature_importance_death.png')
# image_transplant = Image.open('feature_importance_transplant.png')

# st.image(image_death, caption='Feature Importance Graph for Patient Death')
# st.image(image_transplant, caption='Feature Importance Graph for Patient Transplant')

st.markdown(
    """
    #### Interpretability
    > Upcoming
    Provide model interpretability metrics for DeepHit, the black-box deep learning model.
    
    ##### Contributors
        
    Yingji Sun, Gopika Punchhi, [Chang Liu](https://github.com/hellochang), [Sirisha Rambhatla](https://sirisharambhatla.com), Mamatha Bhat
    
    """
)
