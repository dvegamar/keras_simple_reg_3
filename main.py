import keras.optimizers
import tensorflow as tf
import numpy as np
import streamlit as st
from keras.callbacks import History
import pandas as pd
import plotly.graph_objects as go
import io
from statsmodels.stats.outliers_influence import variance_inflation_factor

##################################################################################################
####  SET WEB CONFIGURATION AND TITLES   #########################################################
##################################################################################################

st.set_page_config (page_title='Playing with keras - linear regression', layout="wide")
st.write ("""
    # Simple Keras Regression  
    ### Multivariate regression with keras to play with different model params.  
    In the first two versions of this Simple Keras Regression app we used random numbers and 3 features
    to play with the different params in a simple neural network formed by a single neurone.  
    In this third approach  we replace the numpy random numbers with inputs from the Auto-MPG dataset.  
    Also I am adding some new options to the app as the features selector, that helps understanding the role
    of each feature in the predictions.  
    Also find a selector to understand how standardization of data influences the results.
    I don´t stop in the cleaning of the dataset as it is already done in multiple sites.  
    """)

#################################################################################################
####  READ ORIGINAL DATA AND SHOW IT   ##########################################################
#################################################################################################
st.write ('Auto-MPG dataset for regression analysis. The target (y) is defined as the miles per gallon.')
st.write ('You can download the dataset from here',
          '[Auto-MPG dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset?select=auto-mpg.csv)')
st.write ('''
    1. mpg, miles per gallon
    2. cylinders: multi-valued discrete
    3. displacement: continuous
    4. horsepower: continuous
    5. weight: continuous
    6. acceleration: continuous
    7. model year: multi-valued discrete
    8. origin: multi-valued discrete
    9. car name: string (unique for each instance)
    ''')
df = pd.read_csv ('csv_file', sep=',')
st.write ('#### This is the downloaded csv file content')
st.dataframe (df)

# cleaning the dataframe as our target is to work with NN, not to do data wrangling in this exercise.
# this csv source has horsepower as str son need to change to float
df ['horsepower'] = pd.to_numeric (df ['horsepower'], errors='coerce')
# clean null as there are 6 nulls in the horsepower field
df = df [~df.isnull ().any (axis=1)]
df.reset_index (inplace=True)
df.drop ('index', inplace=True, axis=1)

st.markdown('<hr>',unsafe_allow_html=True)
#################################################################################################
####  GET SOME INFO OF THE DATAFRAME   ##########################################################
#################################################################################################
st.write ('#### Some information about the variables and structure after cleaning')
col1, col2 = st.columns ([1, 1])


# this functions is not used, only is here as informative, an alternative to get_df_info
def get_df_info2 (df):
    buffer = io.StringIO ()
    df.info (buf=buffer)
    lines = buffer.getvalue ()
    st.text (lines)


def get_df_info (df):
    buffer = io.StringIO ()
    df.info (buf=buffer)
    lines = buffer.getvalue ().split ('\n')
    # lines to print directly
    lines_to_print = [0, 1, 2, -3]
    for i in lines_to_print:
        st.write (lines [i])
    # lines to arrange in a df
    list_of_list = []
    for x in lines [5:-3]:
        list = x.split ()
        list_of_list.append (list)
    info_df = pd.DataFrame (list_of_list, columns=['index', 'Column', 'Non-null-Count', 'null', 'Dtype'])
    info_df.drop (columns=['index', 'null'], axis=1, inplace=True)
    st.dataframe (info_df)


with col1:
    st.write ('##### dataframe.info')
    get_df_info (df)

with col2:
    st.write ('##### dataframe.summary')
    st.write (df.describe ())

st.markdown('<hr>',unsafe_allow_html=True)
#################################################################################################
####  STANDARDIZATION OF DATA PRE MODEL SPLITTING #######3#######################################
#################################################################################################
st.write ('#### Data standardization for correlation study')
list_col_to_stnd = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
df_stand = df.copy ()
df_stand.drop ('name', axis=1, inplace=True)
df_stand [list_col_to_stnd] = (df_stand [list_col_to_stnd] - df_stand [list_col_to_stnd].mean ()) / df_stand [
    list_col_to_stnd].std ()
st.dataframe (df_stand)

st.markdown('<hr>',unsafe_allow_html=True)
#################################################################################################
####  CHECK FOR CORRELATIONS   ##########################################################
#################################################################################################

# with corr() method
st.write ('#### Is there correlation?')
st.write ('A coefficient of 1 indicates a perfect positive correlation, meaning that as one variable increases,'
          ' the other variable increases as well. A coefficient of -1 indicates a perfect negative correlation,'
          ' meaning that as one variable increases, the other variable decreases.'
          ' A coefficient of 0 indicates no correlation between the variables.')
st.write (df_stand.corr ())

col7, col8 = st.columns([1,1])

with col7:
    # with heatmap
    st.write ('**Another way with a heatmap**')
    column_list = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
    fig = go.Figure (data=go.Heatmap (z=df_stand.corr (), x=column_list, y=column_list, colorscale='Picnic'))
    st.plotly_chart (fig, width='70%')

with col8:
    # with VIF
    st.write ('**Another way with VIF  Variation Inflation Factor**')
    st.write ('VIF = 1 → No correlation, VIF = 1 to 5 → Moderate correlation, VIF >10 → High correlation. Correlation'
              'matrix shows the correlations between two features, VIF shows correlation of one feature with'
              'all the rest of features'
              )
    # need a df only with independent numerical variables
    df_stand_ind = df_stand.copy ()
    df_stand_ind.drop (columns=['mpg'], axis=1, inplace=True)

    def vif_scores (data):
        VIF_Scores = pd.DataFrame ()
        VIF_Scores ["Independent Features"] = data.columns
        VIF_Scores ["VIF Scores"] = [variance_inflation_factor (data.values, i) for i in range (data.shape [1])]
        return VIF_Scores

    st.write (vif_scores (df_stand_ind))

st.markdown('<hr>',unsafe_allow_html=True)
#########################################################################################
####  FEATURES SELECTION       ##########################################################
#########################################################################################

targets = df.pop ('mpg')
inputs = df.copy ()
inputs.drop ('name', axis=1, inplace=True)

# here we  include a features selector to play with the correlations
st.write ('#### Features selection')
col5, col6 = st.columns ([1, 4])

with col5:
    st.write('Remove or add features to see the role of variables correlation')
    all_features = st.checkbox('All Features')
    remove_cylinders = st.checkbox('Remove cylinders')
    remove_displacement = st.checkbox('Remove displacement')
    remove_horsepower = st.checkbox('Remove horsepower')
    remove_weight = st.checkbox('Remove weight')
    remove_acceleration = st.checkbox('Remove acceleration')
    remove_model_year = st.checkbox('Remove model_year')
    remove_origin = st.checkbox('Remove origin')

    if all_features:
        inputs = df.copy ()
        inputs.drop ('name', axis=1, inplace=True)
    if remove_cylinders:
        inputs.drop ('cylinders', axis=1, inplace=True)
    if remove_displacement:
        inputs.drop ('displacement', axis=1, inplace=True)
    if remove_horsepower:
        inputs.drop ('horsepower', axis=1, inplace=True)
    if remove_weight:
        inputs.drop ('weight', axis=1, inplace=True)
    if remove_acceleration:
        inputs.drop ('acceleration', axis=1, inplace=True)
    if remove_model_year:
        inputs.drop ('model_year', axis=1, inplace=True)
    if remove_origin:
        inputs.drop ('origin', axis=1, inplace=True)

with col6:
    st.dataframe(inputs)


st.markdown('<hr>',unsafe_allow_html=True)
#########################################################################################
####  SPLIT THE DATASET        ##########################################################
#########################################################################################

size = len (targets)
size_train = int (0.8 * size)
size_val = int (0.1 * size)
size_test = int (0.1 * size)

data_train = inputs.iloc [:size_train].to_numpy ()
data_val = inputs.iloc [size_train:size_train + size_val].to_numpy ()
data_test = inputs.iloc [size_train + size_val:].to_numpy ()

target_train = targets.iloc [:size_train].to_numpy ()
target_val = targets.iloc [size_train:size_train + size_val].to_numpy ()
target_test = targets.iloc [size_train + size_val:].to_numpy ()

print (data_train.shape)

##################################################################################################
####  SPLIT DATA NORMALIZATION       ##########################################################
##################################################################################################
# it is a good practice to standardize data after splitting
st.write ('#### Standardization after splitting')
standarize  = st.radio ("### Choose an optimizer A", ['No Standardize ', 'Standardize'])
if standarize == 'Standardize':
    data_train = (data_train - np.mean (data_train)) / np.std (data_train)
    data_val = (data_val - np.mean (data_val)) / np.std (data_val)
    data_test = (data_test - np.mean (data_test)) / np.std (data_test)
    target_train = (target_train - np.mean (target_train)) / np.std (target_train)
    target_val = (target_val - np.mean (target_val)) / np.std (target_val)
    target_test = (target_test - np.mean (target_test)) / np.std (target_test)

st.markdown('<hr>',unsafe_allow_html=True)


##################################################################################################
####  SOME FUNCTIONS and constants      ##########################################################
##################################################################################################


# a list of the activation functions that can be used
act_fun_list = ['linear', 'relu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'selu', 'elu', 'exponential']

# a list of optimizers
optimizer_list = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adamax', 'Nadam', 'Ftrl']

# a list regression losses
losses_list = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
               'mean_squared_logarithmic_error', 'cosine_similarity', 'logcosh']

table_content_1 = '''
    <table style="background-color: #f2f2f2;">
      <tr>
        <td style="padding: 10px; text-align: left;">
          keras.Sequential ([
            tf.keras.layers.Dense (12, input_dim=7, activation=activation_f_A),
            tf.keras.layers.Dense (12, activation=activation_f_A),
            tf.keras.layers.Dense (1) ])
        </td>
      </tr>
    </table>
    '''

table_content_2 = '''
    <table style="background-color: #f2f2f2;">
      <tr>
        <td style="padding: 10px; text-align: left;">
          No hidden layers, just one neurone with a linear combination of features
          tf.keras.Sequential ([tf.keras.layers.Dense (units=1, activation=activation_f_A)]) 
        </td>
      </tr>
    </table>
    '''

# print model results
def model_weights (model):
    model.layers [0].get_weights ()
    weights = model.layers [0].get_weights () [0]
    bias = model.layers [0].get_weights () [1]
    st.write ('#### Results: weights and bias ')
    st.write ('weights --> ', str (weights))
    # st.write ('a --> ' + str (weights [0]) + '; b --> ' + str (weights [1]) + '; c --> ' + str (weights [2]))
    st.write ('bias --> ', str (bias))


# function to plot loss values vs validation loss values
def plot_ls_val (history):
    history_df = pd.DataFrame (history.history)
    fig = go.Figure ()
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['loss'],
                               mode='lines',
                               name='loss'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['val_loss'],
                               mode='lines+markers',
                               name='val_loss',
                               line=dict (color="red")))
    st.plotly_chart (fig, use_container_width=True)


# function to plot different loss functions selected in the compile, metrics params
def plot_losses (history):
    history_df = pd.DataFrame (history.history)
    fig = go.Figure ()
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['loss'],
                               mode='lines',
                               name='loss-selected'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['mse'],
                               mode='lines',
                               line=dict (color="brown"),
                               name='mean_squared_error'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['mae'],
                               mode='lines',
                               line=dict (color="green"),
                               name='mean_absolute_errors'))
    fig.add_trace (go.Scatter (x=history_df.index, y=history_df ['cosine_proximity'],
                               mode='lines',
                               line=dict (color="pink"),
                               name='cosine_proximity'))

    st.plotly_chart (fig, use_container_width=True)


# Function to plot real vs predicted values by the model
def plot_real_pred (model, targettest):
    target_test = np.squeeze (targettest)  # to remove square brackts of the numpy nd array
    data_pred = {'Predict': model.predict (data_test).flatten (),
                 'Real': target_test}
    predict_df = pd.DataFrame (data_pred)

    fig2 = go.Figure ()
    fig2.add_trace (go.Scatter (x=predict_df.index, y=predict_df ['Real'],
                                mode='lines',
                                name='Real data'))
    fig2.add_trace (go.Scatter (x=predict_df.index, y=predict_df ['Predict'],
                                mode='markers',
                                name='Prediction',
                                line=dict (color="red")))
    st.plotly_chart (fig2, use_container_width=True)


# Function to print the evaluated values by the model

def print_evaluate (model):
    evaluate_A = model.evaluate (data_test, target_test, batch_size=128)
    st.markdown (f"""
    * **Loss by selected function:** {round (evaluate_A [0], 4)}
    * **Loss by mean_squared_error:**  {round (evaluate_A [1], 4)}
    * **Loss by mean_absolute_errors:**  {round (evaluate_A [2], 4)}
    * **Loss by cosine proximity:** {round (evaluate_A [3], 4)}
    """)


##################################################################################################
####  SPLIT SCREEN TO COMPARE MODELS       #######################################################
##################################################################################################

col3, col4 = st.columns ([1, 1])

##################################################################################################
# LEFT SCREEN
##################################################################################################

with col3:

    # model epochs
    st.write ('#### Select number of epochs - A ')
    num_epochs_A = st.slider ('### epochs A', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write ('#### Select the activation function - A ')
    activation_f_A = st.radio ("### Choose activation function A", act_fun_list, index=1)

    # select model
    model_selector_A = st.radio ("### Choose a model A", ['DNN 2 hidden layers', 'Simple linear neurone'])
    if model_selector_A == 'DNN 2 hidden layers':
        st.markdown (table_content_1, unsafe_allow_html=True)
        model_A = keras.Sequential ([
            tf.keras.layers.Dense (12, activation=activation_f_A),
            tf.keras.layers.Dense (12, activation=activation_f_A),
            tf.keras.layers.Dense (1)
        ])
    else:
        st.markdown (table_content_2, unsafe_allow_html=True)
        model_A = tf.keras.Sequential ([tf.keras.layers.Dense (units=1, activation=activation_f_A)])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write ('#### Select an optimizer - A ')
    optimizer_sel = st.radio ("### Choose an optimizer A", optimizer_list, index=4)
    learn_rate_A = st.slider ('Learning rate of optimizers A: ', min_value=0.0001, max_value=0.5000, step=0.0001,
                              value=0.01, format="%f")

    optimizer_A = f"tf.keras.optimizers.{optimizer_sel}(learning_rate={learn_rate_A})"
    optimizer_A = eval (optimizer_A)

    st.write ('#### Select regression losses  - A ')
    losses_A = st.radio ("### Choose regression losses A", losses_list)

    model_A.compile (optimizer=optimizer_A,
                     loss=losses_A,
                     metrics=['mse', 'mae', 'cosine_proximity']
                     )

    # train the model
    history_A = History ()
    model_A.fit (data_train, target_train,
                 epochs=num_epochs_A,
                 batch_size=5,
                 verbose=2,
                 validation_data=(data_val, target_val),
                 callbacks=[history_A])

    # Print model results
    # model_weights (model_A)

    # Graph with loss and validation set loss
    st.write ('#### Graphs of loss vs validation loss ')
    plot_ls_val (history_A)

    # Graph of different losses
    st.write ('#### Graphs of different losses ')
    plot_losses (history_A)

    # Evaluating the model
    st.write ('#### Evaluation A  ')
    print_evaluate (model_A)

    # Predict vs actual values
    st.write ('#### Graphs of Real vx Predicted case A ')
    plot_real_pred (model_A, target_test)

##################################################################################################
# RIGHT SCREEN
##################################################################################################
with col4:

    # model epochs
    st.write ('#### Select number of epochs - B ')
    num_epochs_B = st.slider ('### epochs B', min_value=0, max_value=500, step=1, value=50)

    # activation function
    st.write ('#### Select the activation function - B ')
    activation_f_B = st.radio ("### Choose activation function B", act_fun_list, index=0)

    # model
    model_selector_B = st.radio ("### Choose a model B", ['DNN 2 hidden layers', 'Simple linear neurone'], index=1)
    if model_selector_B == 'DNN 2 hidden layers':
        st.markdown (table_content_1, unsafe_allow_html=True)
        model_B = keras.Sequential ([
            tf.keras.layers.Dense (12, activation=activation_f_A),
            tf.keras.layers.Dense (12, activation=activation_f_A),
            tf.keras.layers.Dense (1)
        ])
    else:
        st.markdown (table_content_2, unsafe_allow_html=True)
        model_B = tf.keras.Sequential ([tf.keras.layers.Dense (units=1, activation=activation_f_B)])

    # optimizer and loss function   https://keras.io/api/optimizers/
    st.write ('#### Select an optimizer - B ')
    optimizer_sel_B = st.radio ("### Choose an optimizer B", optimizer_list, index=4)
    learn_rate_B = st.slider ('Learning rate of optimizers B: ', min_value=0.0001, max_value=0.5000, step=0.0001,
                              value=0.01, format="%f")

    optimizer_B = f"tf.keras.optimizers.{optimizer_sel_B}(learning_rate={learn_rate_B})"
    optimizer_B = eval (optimizer_B)

    st.write ('#### Select regression losses  - B ')
    losses_B = st.radio ("### Choose regression losses B", losses_list)

    model_B.compile (optimizer=optimizer_B,
                     loss=losses_B,
                     metrics=['mse', 'mae', 'cosine_proximity']
                     )

    # train the model

    history_B = History ()
    model_B.fit (data_train, target_train,
                 epochs=num_epochs_B,
                 verbose=0,
                 batch_size=5,
                 validation_data=(data_val, target_val),
                 callbacks=[history_B])  # verbose 2 para ver todo

    # Print model results
    # model_weights (model_B)

    # Graph with loss and validation set loss
    st.write ('#### Graphs of loss vs validation loss ')
    plot_ls_val (history_B)

    # Graph of different losses
    st.write ('#### Graphs of different losses ')
    plot_losses (history_B)

    # Evaluating the model
    st.write ('#### Evaluation B  ')
    print_evaluate (model_B)

    # Predict vs actual values
    st.write ('#### Graphs of Real vx Predicted case B ')
    plot_real_pred (model_B, target_test)
