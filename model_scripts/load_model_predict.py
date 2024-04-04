import tensorflow as tf
import pandas as pd
import numpy as np

from class_DeepHit import Model_DeepHit
from tf_slim import fully_connected as FC_Net
from import_data import f_get_Normalization

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
