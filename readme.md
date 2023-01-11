# keras_simple_reg_3
streamlit app with a keras model for linear multivariate regression where you can select model params.   

In this third approach we replace the numpy random numbers with inputs from the Auto-MPG dataset.

Some additional options have been added:

- A correlation of variables study
- A features selection area very helpful to understand how some features do not improve the model
- A standardization option for data, to see how it works and how really improves the model
- A model selection option to compare two different neural networks, one with two hidden layers and 12 nodes; other a simple linear neurone.

Default values of some params (as optimizers) have been changed so after refreshing the app there are some vissible results.

After a while playing with this app you will find that both networks can give the same results, making it obvious that for this example don´t need a DNN.

Approach 1 app: https://dvegamar-keras-simple-reg-1-main-ex40td.streamlit.app/   

Approach 2 app: https://dvegamar-keras-simple-reg-2-main-nvnr5p.streamlit.app/

