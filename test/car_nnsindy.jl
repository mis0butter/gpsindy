using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using CSV, DataFrames 


## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

freq_hz = 5 
noise   = 0 
σn      = 0.02 
opt_σn  = false 
GP_intp = false 

## ============================================ ##
# cross_validate_csv_path 

csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path_σn( csv_path, σn, opt_σn, GP_intp ) 

# create dataframe to store results 
header = [ "csv_file", "λ_min", "train_err", "test_err", "train_traj", "test_traj" ] 
df_min_err_csvs_nnsindy   = DataFrame( fill( [], 6 ), header ) 

# loop 
for i_csv in eachindex( csv_files_vec ) 

    csv_path_file = csv_files_vec[i_csv] 

    # df_min_err_nnsindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv_path_file( csv_path_file, σn, opt_σn, freq_hz, noise, GP_intp ) 
    
    ## ============================================ ##
    # cross_valiate_csv_path 

    # extract data 
    data_train, data_test = car_data_struct( csv_path_file ) 

    x0_train = data_train.x_noise[1,:] 
    x0_test  = data_test.x_noise[1,:]
    
    # cross-validate nnsindy 
    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_nnsindy   = DataFrame( fill( [], 5 ), header ) 
    for i_λ = eachindex( λ_vec ) 

        λ = λ_vec[i_λ] 
        
        # nnsindy!!! 
        # x_sindy_train, x_sindy_test = sindy_lasso_int( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 
    
        # Train NN on the data
        Ξ_nn_lasso = nn_lasso_Ξ_fn( data_train.dx_noise, data_train.x_noise, λ, data_train.u ) 
        
        dx_fn_nn = build_dx_fn(poly_order, x_vars, u_vars, Ξ_nn_lasso)

        x_nn_train = integrate_euler( dx_fn_nn, x0_train, data_train.t, data_train.u ) 
        x_nn_test = integrate_euler( dx_fn_nn, x0_test, data_test.t, data_test.u ) 

        push!( df_nnsindy, [ λ, norm( data_train.x_noise - x_sindy_train ),  norm( data_test.x_noise - x_sindy_test ), x_sindy_train, x_sindy_test ] ) 

    end 

    # save nnsindy min err stats 
    df_min_err_nnsindy   = df_min_err_fn( df_nnsindy, csv_path_file ) 

    ## ============================================ ## 

    push!( df_min_err_csvs_nnsindy, df_min_err_nnsindy[1,:] ) 

end 

# save df_min_err for gpsindy and sindy 
CSV.write( string( save_path_dfs, "df_min_err_csvs_nnsindy.csv" ), df_min_err_csvs_nnsindy ) 






## ============================================ ##
## ============================================ ##

noise_vec = [] 
push!( noise_vec, 0 ) 
push!( noise_vec, 0.01 ) 
push!( noise_vec, 0.02 ) 

for noise = noise_vec  
    for σn = [ 0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3 ] 
        df_min_err_csvs_nnsindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn, GP_intp ) 
    end 
end 
