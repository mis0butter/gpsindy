using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 

## ============================================ 


# tuning parameters  
interpolate_gp = false 
σn             = 0.01 
opt_σn         = true    
freq_hz        = 10 
noise          = 0.02 

# rollout_1.csv  
csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/10hz_noise_0.02/rollout_2.csv" 

sim_params = ( freq_hz = freq_hz, noise = noise, interpolate_gp = interpolate_gp, σn = σn, opt_σn = opt_σn ) 

df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv_path_file( csv_path_file, σn, opt_σn, freq_hz, noise, interpolate_gp ) 

f_train 



## ============================================ ## 
# smooth one column of the data 

x_data = x_train_GP 
y_data = data_train.dx_noise[:, 1] 
x_pred = x_train_GP 

μ_best, σ²_best, best_gp = smooth_column_gp(x_data, y_data, x_pred)

# # Plot results
# fig = Figure()
# ax = Axis(fig[1, 1], xlabel="x", ylabel="y", title="Best Gaussian Process Kernel")

# scatter!(ax, x_data, y_data, label = "Observations", markersize = 10)
# lines!(ax, x_pred, μ_best, label = "GP prediction", linewidth = 2)
# band!(ax, x_pred, μ_best .- 2*sqrt.(σ²_best), μ_best .+ 2*sqrt.(σ²_best), color=(:blue, 0.3))

# axislegend(ax)

# print_kernel(best_gp) 
# fig 


## ============================================ ## 

## ============================================ ## 
# function cross_validate_kernel( csv_path_file, σn, opt_σn, freq_hz, noise ) 

# extract data 
data_train, data_test = make_data_structs( csv_path_file ) 

x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_train_test_data( data_train, data_test ) 

df_sindy, df_gpsindy = cross_validate_sindy_gpsindy(data_train, data_test, x_train_GP, dx_train_GP) 

# save gpsindy min err stats 
df_min_err_sindy   = df_min_err_fn( df_sindy, csv_path_file ) 
df_min_err_gpsindy = df_min_err_fn( df_gpsindy, csv_path_file ) 

# plot 
csv_file = replace( split( csv_path_file, "/" )[end], ".csv" => "" ) 
f_train  = plot_train( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, interpolate_gp, csv_file ) 
f_test   = plot_test( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, interpolate_gp, csv_file )  

#     return df_min_err_sindy, df_min_err_gpsindy, f_train, f_test 
# end 


