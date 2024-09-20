using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 


## ============================================ ## 
# # rollout_1.csv  

freq_hz = 10 
noise   = 0.01 

# rollout_1.csv  
csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/10hz_noise_0.01/rollout_2.csv"

# tuning parameters  
interpolate_gp = 2 

σn      = 0.02 
opt_σn  = true   

sim_params = ( freq_hz = freq_hz, noise = noise, interpolate_gp = interpolate_gp, σn = σn, opt_σn = opt_σn ) 

df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv_path_file( csv_path_file, σn, opt_σn, freq_hz, noise, interpolate_gp ) 

f_train 


