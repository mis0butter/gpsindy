using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 


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
## ============================================ ## 

μ_best, σ²_best, best_gp = smooth_column_gp(x, y, x_pred) 

## ============================================ ##
# posterior GP and optimize hps 

# function smooth_gp( x_pred, x_data, y_data ) 
function smooth_gp( x_prior, x_train, y_train ) 
    
    σ_n     = 1e-1 
    σ_n_opt = true 

    # set up posterior 
    x_rows = size( x_prior, 1 ) ; n_vars = size(y_train, 2) 
    y_post = zeros( x_rows, n_vars ) 
    
    # optimize hyperparameters, compute posterior y_post for each state 
    for i = 1 : n_vars 
    
        # kernel  
        mZero     = MeanZero()              # zero mean function 
        kern      = SE( 0.0, 0.0 )          # squared eponential kernel (hyperparams on log scale) 
        
        # log_noise = log( σ_n )              # (optional) log std dev of obs 
        log_noise = log( σ_n )              # (optional) log std dev of obs 
        
        # y_train = dx_noise[:,i] - dx_mean[:,i]
        gp      = GP( x_train', y_train[:,i], mZero, kern, log_noise ) 
        optimize!( gp, method = LBFGS( linesearch = LineSearches.BackTracking() ), noise = σ_n_opt ) 

        y_post[:,i] = predict_y( gp, x_prior' )[1]  
    
    end 

    return y_post 

end 

## ============================================ ##
# smooth training and test data with GPs 

# export smooth_train_test_data 
function smooth_train_test_data( data_train, data_test, σ_n = 0.1, opt_σn = true ) 

    # # first - smooth measurements with Gaussian processes 
    # x_train_GP  = smooth_gp( data_train.t, data_train.t, data_train.x_noise ) 
    # dx_train_GP = smooth_gp( x_train_GP, x_train_GP, data_train.dx_noise ) 
    # x_test_GP   = smooth_gp( data_test.t, data_test.t, data_test.x_noise ) 
    # dx_test_GP  = smooth_gp( x_test_GP, x_test_GP, data_test.dx_noise ) 

    # first - smooth measurements with Gaussian processes 
    x_train_GP, _, _  = smooth_array_gp(data_train.t, data_train.x_noise, data_train.t)
    dx_train_GP, _, _ = smooth_array_gp(x_train_GP, data_train.dx_noise, x_train_GP)
    x_test_GP, _, _   = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)
    dx_test_GP, _, _  = smooth_array_gp(x_test_GP, data_test.dx_noise, x_test_GP)

    return x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 

## ============================================ ## 
# function cross_validate_kernel( csv_path_file, σn, opt_σn, freq_hz, noise ) 

# extract data 
data_train, data_test = make_data_structs( csv_path_file ) 

x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_train_test_data( data_train, data_test, σn, opt_σn ) 

# cross-validate gpsindy 
λ_vec      = λ_vec_fn() 
header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
df_gpsindy = DataFrame( fill( [], 5 ), header ) 
df_sindy   = DataFrame( fill( [], 5 ), header ) 
for i_λ = eachindex(λ_vec) 

    λ = λ_vec[i_λ] 
    
    # sindy!!! 
    x_sindy_train, x_sindy_test = integrate_sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 
    push!( df_sindy, [ λ, norm( data_train.x_noise - x_sindy_train ),  norm( data_test.x_noise - x_sindy_test ), x_sindy_train, x_sindy_test ] ) 

    # gpsindy!!! 
    x_gpsindy_train, x_gpsindy_test = integrate_sindy_lasso( x_train_GP, dx_train_GP, λ, data_train, data_test ) 
    push!( df_gpsindy, [ λ, norm( data_train.x_noise - x_gpsindy_train ),  norm( data_test.x_noise - x_gpsindy_test ), x_gpsindy_train, x_gpsindy_test ] ) 

end 

# save gpsindy min err stats 
df_min_err_sindy   = df_min_err_fn( df_sindy, csv_path_file ) 
df_min_err_gpsindy = df_min_err_fn( df_gpsindy, csv_path_file ) 

# plot 
csv_file = replace( split( csv_path_file, "/" )[end], ".csv" => "" ) 
f_train  = plot_train( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, interpolate_gp, csv_file ) 
f_test   = plot_test( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, σn, opt_σn, freq_hz, noise, interpolate_gp, csv_file )  

#     return df_min_err_sindy, df_min_err_gpsindy, f_train, f_test 
# end 


