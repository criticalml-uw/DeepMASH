import streamlit as st
from PIL import Image
import pandas as pd
import tensorflow as tf
import pandas as pd
import numpy as np

from class_DeepHit import Model_DeepHit
from tf_slim import fully_connected as FC_Net
from import_data import f_get_Normalization
    
st.set_page_config(
    page_title="DeepHit",
    page_icon="📈",
)

st.title('DeepHit – Patient Prediction')

tab_sample, tab_customize = st.tabs(["Sample dataset", "Upload"])

with tab_sample:
    processed_patient_df = pd.read_csv("./data/sample_processed_data.csv", index_col=0)
        
    data_load_state = st.text('Making predictions...')


    ####################################
    # Load model 

    tf.compat.v1.disable_eager_execution()

    def load_logging(filename):
        data = dict()
        with open(filename) as f:
            def is_float(input):
                try:
                    num = float(input)
                except ValueError:
                    return False
                return True

            for line in f.readlines():
                if ':' in line:
                    key,value = line.strip().split(':', 1)
                    if value.isdigit():
                        data[key] = int(value)
                    elif is_float(value):
                        data[key] = float(value)
                    elif value == 'None':
                        data[key] = None
                    else:
                        data[key] = value
                else:
                    pass 
        return data

    # Load the saved optimized hyperparameters

    in_hypfile = 'model/hyperparameters_log.txt'
    in_parser = load_logging(in_hypfile)

    # Forward the hyperparameters
    mb_size                     = in_parser['mb_size']

    iteration                   = in_parser['iteration']

    keep_prob                   = in_parser['keep_prob']
    lr_train                    = in_parser['lr_train']

    h_dim_shared                = in_parser['h_dim_shared']
    h_dim_CS                    = in_parser['h_dim_CS']
    num_layers_shared           = in_parser['num_layers_shared']
    num_layers_CS               = in_parser['num_layers_CS']

    if in_parser['active_fn'] == 'relu':
        active_fn                = tf.nn.relu
    elif in_parser['active_fn'] == 'elu':
        active_fn                = tf.nn.elu
    elif in_parser['active_fn'] == 'tanh':
        active_fn                = tf.nn.tanh
    else:
        print('Error!')


    initial_W                   = tf.keras.initializers.glorot_normal()

    alpha                       = in_parser['alpha']  #for log-likelihood loss
    beta                        = in_parser['beta']  #for ranking loss


    # Create the dictionaries 
    # For the input settings
    input_dims                  = { 'x_dim'         : 26,
                                    'num_Event'     : 2,
                                    'num_Category'  : 143}

    # For the hyperparameters
    network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                    'h_dim_CS'          : h_dim_CS,
                                    'num_layers_shared'    : num_layers_shared,
                                    'num_layers_CS'    : num_layers_CS,
                                    'active_fn'      : active_fn,
                                    'initial_W'         : initial_W }

    # Create the DeepHit network architecture

    tf.compat.v1.reset_default_graph()

    #imported_graph = tf.compat.v1.train.import_meta_graph('model/model/model_itr_0.meta')

    #with tf.compat.v1.Session() as sess:
        # restore the saved vairable
        
    #    imported_graph.restore(sess,'models/checkpoint')
        
    #    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)

    tf.compat.v1.reset_default_graph()

    config = tf.compat.v1.ConfigProto

    sess = tf.compat.v1.Session()

    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)

    saver = tf.compat.v1.train.Saver()

    sess.run(tf.compat.v1.global_variables_initializer())

    # Restoring the trained model
    saver.restore(sess, 'model/model/model_itr_0')


    ##########################################
    # import data and predict

    processed_data = pd.read_csv('data/sample_processed_data.csv', index_col=0)

    get_x = lambda df: (df
                        .drop(columns=["event","wl_to_event","PX_ID"])
                        .values.astype('float32'))

    data = np.asarray(get_x(processed_data))

    data = f_get_Normalization(data, 'standard')

    #prediction and convert to dataframe
    pred = model.predict(data)

    m,n,r = pred.shape
    out_arr = np.column_stack((np.repeat(np.arange(m),n),pred.reshape(m*n,-1)))
    out_df = pd.DataFrame(out_arr)

    out_no_index = out_df.iloc[: , 1:]

    out_no_index.to_csv('pred_risk.csv')


    pred_death_risk = out_no_index.iloc[::2, :].reset_index()
    pred_death_risk= pred_death_risk.drop(['index'],axis=1)
    pred_transplant_risk = out_no_index.iloc[1:, :]
    pred_transplant_risk = pred_transplant_risk.iloc[::2, :].reset_index()
    pred_transplant_risk= pred_transplant_risk.drop(['index'],axis=1)
    pred_death_risk.to_csv("pred_risk_death.csv")
    pred_transplant_risk.to_csv("pred_risk_transplant.csv")

    # st.success('Predictions made!', icon="✅")

    data_load_state.text("Predictions Successfully Made!")
    
    st.markdown("""
                
                Predictions are made with `sample_test_data.csv` in our 
                [code repo](https://github.com/criticalml-uw/DeepNASH/blob/main/data/sample_test_data.csv).
                
                If you would like to know more information, please refer to the actual event 
                and time in the patient examples in our paper.""")


    st.markdown("#### Plot Predictions")
    import pandas as pd
    import numpy as np

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import itertools
    from itertools import cycle


    # Import prediction data
    pred_risk_death = pd.read_csv('./data/pred_risk_death.csv',index_col=0)
    pred_risk_transplant = pd.read_csv('./data/pred_risk_transplant.csv',index_col=0)


    # Specifiy patient 
    # st.write(pred_risk_death)
    selected_patient = st.selectbox(
        'Select patient',
        tuple(pred_risk_death.index.tolist()))
        
    patient_1 = pd.DataFrame({
    'transplant': pred_risk_transplant.iloc[selected_patient][0:12],
    'death': pred_risk_death.iloc[selected_patient][0:12]
    })

    marker2 = itertools.cycle(('o', 's')) 
    lines2 = itertools.cycle(("-","--"))


    x=range(0,12,1)


    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#006400','#c1272d']) 

    fig,ax = plt.subplots(figsize=(10,10))

    #plt.plot(x,patientt1,"-o")
    ax.plot(x,patient_1['transplant'],
            linestyle=next(lines2),
            marker=next(marker2))

    ax.plot(x,patient_1['death'],
            linestyle=next(lines2),
            marker=next(marker2))

    ax.grid()
    fig.suptitle('Surrogate risk of death and transplant using DeepNash for sample patient #3', fontsize=30)
    ax.set_ylabel('Surrogate risks',fontsize=30)
    ax.set_xlabel('Time in months',fontsize=30)

    line_labels=["Transplant",'Death']
    ax.legend(line_labels, loc ='lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 22},
                bbox_to_anchor=(0.5, -0.2))

    plt.xticks(visible=False)
    plt.yticks(visible=False)

    fig.set_size_inches(15,12, forward=True)

    # Currently just for next year
    import plotly.express as px
    # st.markdown(f"Plot for Patient {selected_patient}")
    pl_fig = px.line(patient_1, x=patient_1.index, y=["transplant", "death"], title=f'Patient {selected_patient} Prediction',
                    labels={
                        "index": "Month",
                        "value": "Risks",
                    }, color_discrete_sequence = ["green", "red"])
    st.plotly_chart(pl_fig, theme="streamlit", use_container_width=True)
    view_pred_df = st.checkbox("View Predictions")
    if view_pred_df:
        st.dataframe(patient_1)



    # Download
    with open('./data/pred_risk.csv') as pred_risk:
        btn  = st.download_button(
                label="Download Patient Prediction",
                data=pred_risk,
                file_name='pred_risk.csv',
                mime='text/csv',
                )


    download_img_name = 'patient.png'
    plt.savefig(download_img_name)
    with open(download_img_name, "rb") as img:
        btn = st.download_button(
                label="Download image",
                data=img,
                file_name="patient.png",
                mime="image/png"
            )
            
with tab_customize:                      
    st.markdown("### Step 1: Upload Data")

    upload_csv_msg = st.markdown("""You need to upload a csv file. """)
    with open('./data/sample_processed_data.csv') as sample_input:
        btn  = st.download_button(
                label="Download input template",
                data=sample_input,
                file_name='template.csv',
                mime='text/csv',
                )
    uploaded_csv = st.file_uploader('Choose a file containing patient logs: ')

    if uploaded_csv:
        processed_patient_df = pd.read_csv(uploaded_csv, index_col=0)
        upload_csv_msg.text = "Data successfully uploaded"
        
        view_uploaded_data = st.checkbox("View Uploaded Data")
        if view_uploaded_data:
            st.dataframe(processed_patient_df.head(), use_container_width=True)
        
        st.markdown("### Step 2: Make Predictions")    
        data_load_state = st.text('Making predictions...')

        # Step 2: Making Predictions
        
        ####################################
        # Load model 

        tf.compat.v1.disable_eager_execution()

        def load_logging(filename):
            data = dict()
            with open(filename) as f:
                def is_float(input):
                    try:
                        num = float(input)
                    except ValueError:
                        return False
                    return True

                for line in f.readlines():
                    if ':' in line:
                        key,value = line.strip().split(':', 1)
                        if value.isdigit():
                            data[key] = int(value)
                        elif is_float(value):
                            data[key] = float(value)
                        elif value == 'None':
                            data[key] = None
                        else:
                            data[key] = value
                    else:
                        pass 
            return data

        # Load the saved optimized hyperparameters

        in_hypfile = 'model/hyperparameters_log.txt'
        in_parser = load_logging(in_hypfile)

        # Forward the hyperparameters
        mb_size                     = in_parser['mb_size']

        iteration                   = in_parser['iteration']

        keep_prob                   = in_parser['keep_prob']
        lr_train                    = in_parser['lr_train']

        h_dim_shared                = in_parser['h_dim_shared']
        h_dim_CS                    = in_parser['h_dim_CS']
        num_layers_shared           = in_parser['num_layers_shared']
        num_layers_CS               = in_parser['num_layers_CS']

        if in_parser['active_fn'] == 'relu':
            active_fn                = tf.nn.relu
        elif in_parser['active_fn'] == 'elu':
            active_fn                = tf.nn.elu
        elif in_parser['active_fn'] == 'tanh':
            active_fn                = tf.nn.tanh
        else:
            print('Error!')


        initial_W                   = tf.keras.initializers.glorot_normal()

        alpha                       = in_parser['alpha']  #for log-likelihood loss
        beta                        = in_parser['beta']  #for ranking loss


        # Create the dictionaries 
        # For the input settings
        input_dims                  = { 'x_dim'         : 26,
                                        'num_Event'     : 2,
                                        'num_Category'  : 143}

        # For the hyperparameters
        network_settings            = { 'h_dim_shared'         : h_dim_shared,
                                        'h_dim_CS'          : h_dim_CS,
                                        'num_layers_shared'    : num_layers_shared,
                                        'num_layers_CS'    : num_layers_CS,
                                        'active_fn'      : active_fn,
                                        'initial_W'         : initial_W }

        # Create the DeepHit network architecture

        tf.compat.v1.reset_default_graph()

        #imported_graph = tf.compat.v1.train.import_meta_graph('model/model/model_itr_0.meta')

        #with tf.compat.v1.Session() as sess:
            # restore the saved vairable
            
        #    imported_graph.restore(sess,'models/checkpoint')
            
        #    model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)

        tf.compat.v1.reset_default_graph()

        config = tf.compat.v1.ConfigProto

        sess = tf.compat.v1.Session()

        model = Model_DeepHit(sess, "DeepHit", input_dims, network_settings)

        saver = tf.compat.v1.train.Saver()

        sess.run(tf.compat.v1.global_variables_initializer())

        # Restoring the trained model
        saver.restore(sess, 'model/model/model_itr_0')


        ##########################################
        # import data and predict

        processed_data = pd.read_csv('data/sample_processed_data.csv', index_col=0)

        get_x = lambda df: (df
                            .drop(columns=["event","wl_to_event","PX_ID"])
                            .values.astype('float32'))

        data = np.asarray(get_x(processed_data))

        data = f_get_Normalization(data, 'standard')

        #prediction and convert to dataframe
        pred = model.predict(data)

        m,n,r = pred.shape
        out_arr = np.column_stack((np.repeat(np.arange(m),n),pred.reshape(m*n,-1)))
        out_df = pd.DataFrame(out_arr)

        out_no_index = out_df.iloc[: , 1:]

        out_no_index.to_csv('pred_risk.csv')


        pred_death_risk = out_no_index.iloc[::2, :].reset_index()
        pred_death_risk= pred_death_risk.drop(['index'],axis=1)
        pred_transplant_risk = out_no_index.iloc[1:, :]
        pred_transplant_risk = pred_transplant_risk.iloc[::2, :].reset_index()
        pred_transplant_risk= pred_transplant_risk.drop(['index'],axis=1)
        pred_death_risk.to_csv("pred_risk_death.csv")
        pred_transplant_risk.to_csv("pred_risk_transplant.csv")
    

        # st.success('Predictions made!', icon="✅")

        data_load_state.text("Predictions Successfully Made!")



        st.markdown("### Step 3: Plot Predictions")
        import pandas as pd
        import numpy as np

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import itertools
        from itertools import cycle


        #import prediction data
        pred_risk_death = pd.read_csv('./data/pred_risk_death.csv', index_col=0)
        pred_risk_transplant = pd.read_csv('./data/pred_risk_transplant.csv', index_col=0)


        #specifiy patient 
        selected_patient = st.selectbox(
            'Select patient',
            tuple(pred_risk_death.index.tolist()),
            key="select-box-custom-data"
            )
            
        patient_1 = pd.DataFrame({
        'transplant': pred_risk_transplant.iloc[selected_patient][0:12],
        'death': pred_risk_death.iloc[selected_patient][0:12]
        })
        marker2 = itertools.cycle(('o', 's')) 
        lines2 = itertools.cycle(("-","--"))


        x=range(0,12,1)


        mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['#006400','#c1272d']) 

        fig,ax = plt.subplots(figsize=(10,10))

        #plt.plot(x,patientt1,"-o")
        ax.plot(x,patient_1['transplant'],
                linestyle=next(lines2),
                marker=next(marker2))

        ax.plot(x,patient_1['death'],
                linestyle=next(lines2),
                marker=next(marker2))

        ax.grid()
        fig.suptitle('Surrogate risk of death and transplant using DeepNash for sample patient #3',fontsize=30)
        ax.set_ylabel('Surrogate risks',fontsize=30)
        ax.set_xlabel('Time in months',fontsize=30)

        line_labels=["Transplant",'Death']
        ax.legend(line_labels, loc ='lower center', borderaxespad=0.1, ncol=6, labelspacing=0.,  prop={'size': 22},
                    bbox_to_anchor=(0.5, -0.2))

        plt.xticks(visible=False)
        plt.yticks(visible=False)

        fig.set_size_inches(15,12, forward=True)

        # Currently just for next year
        import plotly.express as px
        st.markdown(f"Sample Plot for Patient {selected_patient}")
        pl_fig = px.line(patient_1, x=patient_1.index, y=["transplant", "death"], title=f'Patient {selected_patient} Prediction',
                        labels={
                            "index": "Month",
                            "value": "Risks",
                        },
                        color_discrete_sequence=['green', 'red'])
        # pl_fig.update_traces(line_color='red')

        st.plotly_chart(pl_fig, theme="streamlit", use_container_width=True)
        view_pred_df = st.checkbox("View Predictions", key="checkbox-custom-data")
        if view_pred_df:
            st.dataframe(patient_1)



        # Download
        st.markdown("### Step 4: Download Results")
        with open('./data/pred_risk.csv') as pred_risk:
            btn  = st.download_button(
                    label="Download Patient Prediction",
                    data=pred_risk,
                    file_name='pred_risk.csv',
                    mime='text/csv',
                    key="download-custom-data")


        download_img_name = 'patient.png'
        plt.savefig(download_img_name)
        with open(download_img_name, "rb") as img:
            btn = st.download_button(
                    label="Download image",
                    data=img,
                    file_name="patient.png",
                    mime="image/png",
                    key="download-img-custom-data"
                )
            