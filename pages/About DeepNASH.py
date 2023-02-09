import streamlit as st

st.set_page_config(
    page_title="About DeepNASH",
    page_icon="👩‍💻",
    # menu_items={
    # 'Get Help': 'https://www.extremelycoolapp.com/help',
    # 'Report a bug': "https://www.extremelycoolapp.com/bug",
    # 'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

st.title("DeepNASH Project")

st.markdown(
    """
    This is web app provides an interactive visualization of the NASH project. 
    See the sidebar to upload patient data, generate predictions and view graphical outcome.
    
    #### DeepHit
    > Paper submitted for review
    Predicting the one-year death and transplant rate trajectory of Non-alcoholic steatohepatitis (NASH) cirrhosis patients awaiting liver transplant
    
    #### Interpretability
    > Upcoming
    Provide model interpretability metrics for DeepHit, the black-box deep learning model.
    
    ##### Contributors
        
    Yingji Sun, Gopika Punchhi, [Chang Liu](https://github.com/hellochang), [Sirisha Rambhatla](https://sirisharambhatla.com), Mamatha Bhat
    
    """
)
