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

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0.02/rollout_3.csv" 

df_min_err_sindy, df_min_err_gpsindy, fig_csv = cross_validate_csv( csv_path_file, 1 ) 

fig_csv 

## ============================================ ## 

df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = run_save_csv_files( 10, 0.03 ) 

## ============================================ ## 
# find bad sindy 

csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/50hz_noise_0.02/rollout_3.csv" 

interp_factor = 1 

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_sindy   = DataFrame( fill( [], 5 ), header ) 

    for i_λ = eachindex(λ_vec) 

        λ = λ_vec[i_λ] 
        
        # sindy!!! 
        x_sindy_train, x_sindy_test = integrate_sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

        push!(df_sindy, [
            λ,                                          # lambda  
            norm(data_train.x_noise - x_sindy_train),   # train error  
            norm(data_test.x_noise - x_sindy_test),     # test error  
            x_sindy_train,                              # train trajectory  
            x_sindy_test                                # test trajectory  
        ])

    end 

    # save gpsindy min err stats 
    df_best_sindy = df_min_err_fn(df_sindy, csv_path_file)

    x_sindy_train = df_best_sindy.train_traj[1] 
    x_sindy_test  = df_best_sindy.test_traj[1] 

    fig = plot_data( data_train, x_sindy_train, data_test, x_sindy_test, df_best_sindy, df_best_sindy, interp_factor, csv_path_file )    
