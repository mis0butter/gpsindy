using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using Optim 
using LineSearches  
using GaussianProcesses 
using DataFrames 


## ============================================ ## 


csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/25hz_noise_0/rollout_2.csv" 

df_best_sindy, df_best_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 

fig_csv 


## ============================================ ## 


df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 25, 0 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 25, 0.01 ) 


## ============================================ ## 

interp_factor = 1 

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    # df_sindy, df_gpsindy, x_train_GP, _  = process_data_and_cross_validate(data_train, data_test, interp_factor)
    # x_train_GP, dx_train_GP = smooth_train_data(data_train, data_test)
    # first - smooth measurements with Gaussian processes 
    x_train_GP, _, _  = smooth_array_gp(data_train.t, data_train.x_noise, data_train.t)

    println("using u to smooth dx3 and dx4") 

    # dx_train_GP, _, _ = smooth_array_gp(x_train_GP, data_train.dx_noise, x_train_GP) 
    dx_train_GP = similar(data_train.dx_noise) 

    x_data = x_train_GP 
    y_data = data_train.dx_noise[:,1:2]
    x_pred = x_train_GP  
    dx_train_GP[:,1:2], _, _ = smooth_array_gp(x_data, y_data, x_pred) 

    x_data = data_train.u[:,1] 
    y_data = data_train.dx_noise[:,3]
    x_pred = data_train.u[:,1] 
    dx_train_GP[:,3], _, _ = smooth_column_gp(x_data, y_data, x_pred) 

    x_data = data_train.u[:,2] 
    y_data = data_train.dx_noise[:,4]
    x_pred = data_train.u[:,2] 
    dx_train_GP[:,4], _, _ = smooth_column_gp(x_data, y_data, x_pred)   

    # # save gpsindy min err stats 
    # df_best_sindy   = df_min_err_fn(df_sindy, csv_path_file)
    # df_best_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 

    # fig = plot_data( data_train, x_train_GP, data_test, df_best_sindy, df_best_gpsindy, interp_factor, csv_path_file )    

