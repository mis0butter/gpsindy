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

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0/rollout_3.csv" 

df_best_sindy, df_best_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 
fig_csv 

## ============================================ ## 


# df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.01 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.04 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.05 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.06 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.07 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.08 ) 
df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err = run_save_csv_files( 5, 0.09 ) 


## ============================================ ## 
# find bad sindy 

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0/rollout_3.csv" 

interp_factor = 1 


    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    df_sindy, df_gpsindy, x_train_GP, _  = process_data_and_cross_validate(data_train, data_test, interp_factor)

    # save gpsindy min err stats 
    df_best_sindy   = df_min_err_fn(df_sindy, csv_path_file)
    df_best_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 

    fig = plot_data( data_train, x_train_GP, data_test, df_best_sindy, df_best_gpsindy, interp_factor, csv_path_file )    

