import streamlit as st
from PIL import Image
import pandas as pd
from import_data import f_get_Normalization
import numpy as np
# from 1_🌍_Adding_Logs import data

# from pages.load_model import model

from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder

st.set_page_config(
    page_title="Plotting Predictions",
    page_icon="📈",
)

st.sidebar.success("Access interface functionality here.")

# st.title('NASH Project Interface')
st.markdown("<h2 style='text-align: center; color: black;'>Plotting Predictions</h2>", unsafe_allow_html=True)

license_key = "For_Trialing_ag-Grid_Only-Not_For_Real_Development_Or_Production_Projects-Valid_Until-18_March_2021_[v2]_MTYxNjAyNTYwMDAwMA==948d8f51e73a17b9d78e03e12b9bf934"





if st.button("Generate Predictions"):
    st.write("Please wait while predictions are being generated...")
    st.image("practice_graph.jpeg")
    #convert and predict 
    # processed_data = data

    # get_x = lambda df: (df
    #                     .drop(columns=["event","wl_to_event"])
    #                     .values.astype('float32'))

    # data = np.asarray(get_x(processed_data))

    # data = f_get_Normalization(data, 'standard')

    # #prediction and convert to dataframe
    # pred = load_model.model.predict(data)

    # pred =pred[:,:,:]

    # m,n,r = pred.shape
    # out_arr = np.column_stack((np.repeat(np.arange(m),n),pred.reshape(m*n,-1)))
    # out_df = pd.DataFrame(out_arr)

    # predrisk = predrisk.iloc[: , 1:]

    # ################################################
    # #graph
    # patient1 = pd.DataFrame({
    # 'transplant': predrisk.iloc[1][0:13],
    # 'death': predrisk.iloc[0][0:13]
    # })

    # fig, (ax1, ax2,ax3,ax4) = plt.subplots(4,sharex=True)

    # fig.set_figheight(10)
    # fig.set_figwidth(10)

    # plt.rc('font', size=15)

    # x=range(0,13,1)

    # fig.suptitle('Patient predicted trajectory',x=0.5, y=0.93)

    # fig, ax1= plt.subplots(1,sharex=True)

    # fig.set_figheight(10)
    # fig.set_figwidth(10)

    # mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#006400','#c1272d']) 

    # ax1.plot(x,patientt1,"-o")
    # plt.setp(ax1, ylim=(0,0.01))
    # ax1.grid()
    # ax1.title.set_text('Patient predicted one year trajectory')

    # line_labels=["Transplant",'Death']

    # plt.figlegend(line_labels, loc ='lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 13},
    #             bbox_to_anchor=(0.5, 0.04))

    # fig.text(0.5,0.08, 'Months', ha='center',fontsize=15)
    # fig.text(0, 0.5, 'Surrogate risk of event', va='center', rotation='vertical',fontsize=15)

    # print(fig)


else:
    st.write("Press to generate predictions.")




