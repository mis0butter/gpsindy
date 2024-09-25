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

df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv(csv_path_file, 2) 

f_train 







## ============================================ ## 
## ============================================ ## 

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    # if interp_factor == 1 

    #     x_train_GP, dx_train_GP, x_test_GP, _ = smooth_train_test_data(data_train, data_test)

    #     # cross validate sindy and gpsindy  
    #     df_sindy, df_gpsindy = cross_validate_sindy_gpsindy(data_train, data_test, x_train_GP, dx_train_GP)
        
    # else 

        t_train_interp, u_train_interp, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = interpolate_train_test_data(data_train, data_test, interp_factor) 

        # cross validate sindy and gpsindy   
        df_sindy, df_gpsindy = cross_validate_sindy_gpsindy_interp(data_train, data_test, x_train_GP, dx_train_GP, t_train_interp, u_train_interp) 

    # end 

    # save gpsindy min err stats 
    df_min_err_sindy   = df_min_err_fn(df_sindy, csv_path_file)
    df_min_err_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 

    f_train = plot_data( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "train" )  
    f_test  = plot_data( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "test" ) 