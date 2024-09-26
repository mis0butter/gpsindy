using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 


## ============================================ ## 


csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/10hz_noise_0.02/rollout_1.csv" 

df_min_err_sindy, df_min_err_gpsindy, fig = cross_validate_csv( csv_path_file, 1 ) 

fig 


## ============================================ ## 


interp_factor = 1  

# extract and smooth data 
data_train, data_test = make_data_structs(csv_path_file)

df_sindy, df_gpsindy, x_train_GP, _, x_test_GP = process_data_and_cross_validate(data_train, data_test, interp_factor)

# save gpsindy min err stats 
df_min_err_sindy   = df_min_err_fn(df_sindy, csv_path_file)
df_min_err_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 


## ============================================ ## 


fig = plot_data( data_train, x_train_GP, data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file )    


## ============================================ ## 







