using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames


## ============================================ ## 
# # rollout_1.csv  

freq_hz = 50 
noise   = 0  

# rollout_1.csv  
csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0/rollout_1.csv"

# tuning parameters  
interpolate_gp = false 

σn      = 0.02 
opt_σn  = false 

## ============================================ ## 
# let's look at rollout_1.csv 

# Extract data from rollout_1.csv
# Read the CSV file
data_frame = CSV.read(csv_path_file, DataFrame)

# Extract training and testing data from CSV file
data_train, data_test = make_data_structs( csv_path_file ) 
# data_train and data_test fields: 
#   t 
#   u 
#   x_true 
#   dx_true  
#   x_noise 
#   dx_noise 

# Standard GP smoothing on original data
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_data_gp(data_train, data_test, σn, opt_σn) 

## ============================================ ## 

# Prepare for cross-validation of SINDy and GP-SINDy
λ_vector    = λ_vec_fn()  # Generate vector of λ values for regularization
header      = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ]
df_gpsindy  = DataFrame( fill( [], 5 ), header )    # DataFrame to store GP-SINDy results
df_sindy    = DataFrame( fill( [], 5 ), header )    # DataFrame to store SINDy results

# Perform cross-validation for each λ value
for i_λ = eachindex(λ_vector) 

    λ = λ_vector[i_λ] 
    
    # Apply standard SINDy
    x_sindy_train, x_sindy_test = sindy_lasso_int(data_train.x_noise, data_train.dx_noise, λ, data_train, data_test) 

    # Calculate and store errors for SINDy 
    push!(df_sindy, [λ, 
                     norm(data_train.x_noise - x_sindy_train),  # Training error
                     norm(data_test.x_noise - x_sindy_test),    # Testing error
                     x_sindy_train, x_sindy_test]) 

    # Standard GP-SINDy
    x_gpsindy_train, x_gpsindy_test = sindy_lasso_int(x_train_GP, dx_train_GP, λ, data_train, data_test)

    # Calculate and store errors for GP-SINDy
    push!(df_gpsindy, [λ, 
                       norm(data_train.x_noise - x_gpsindy_train),  # Training error
                       norm(data_test.x_noise - x_gpsindy_test),    # Testing error
                       x_gpsindy_train, x_gpsindy_test])

end 


## ============================================ ## 
# Plot results 

# save gpsindy min err stats 
df_min_err_sindy   = df_min_err_fn( df_sindy, csv_path_file ) 
df_min_err_gpsindy = df_min_err_fn( df_gpsindy, csv_path_file ) 

csv_file = replace( split( csv_path_file, "/" )[end], ".csv" => "" ) 

plot_train( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, csv_file ) 

plot_test( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, csv_file )  




