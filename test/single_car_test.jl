using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using DataFrames 


## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

freq_hz = 50 
noise   = 0.02 

# optimize GPs with GPs 
σn      = 0.2 
opt_σn  = false 

# csv_num = 18 

# ----------------------- #
# run cross-validation 

header = [ "csv_file", "λ_min", "train_err", "test_err", "train_traj", "test_traj" ] 
df_min_err_csvs_sindy   = DataFrame( fill( [], 6 ), header ) 
df_min_err_csvs_gpsindy = DataFrame( fill( [], 6 ), header ) 
for csv_num = 1:45 

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )
    csv_file = string( "rollout_", csv_num, ".csv" )   
    csv_path_file = string(csv_path, csv_file ) 

    df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_dfs( csv_path_file, σn, opt_σn, freq_hz, noise ) 
    display(f_train) ; display(f_test) 

    push!( df_min_err_csvs_sindy, df_min_err_sindy[1,:] ) 
    push!( df_min_err_csvs_gpsindy, df_min_err_gpsindy[1,:] ) 

end 

# report mean error for testing 
mean_err_sindy   = mean( df_min_err_csvs_sindy.test_err ) 
mean_err_gpsindy = mean( df_min_err_csvs_gpsindy.test_err ) 
println( "mean_err_sindy   = ", mean_err_sindy) 
println( "mean_err_gpsindy = ", mean_err_gpsindy ) 
















































## ============================================ ##
# ok, this is taking too long ... let's just break sindy 

freq_hz  = 50 
noise    = 0.02 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

# create save path 
csv_files_vec, save_path, save_path_fig = mkdir_save_path( csv_path ) 

λ_vec = λ_vec_fn() 

min_train_err_hist = [ 0.0 0.0 0.0 ] 
for i in eachindex(csv_files_vec) 
# i = 1 
    
    # extract data 
    data_train, data_test = car_data_struct( csv_files_vec[i] ) 

    sindy_train_err_hist = [] 
    sindy_test_err_hist  = [] 
    for i_λ = eachindex( λ_vec ) 

        λ = λ_vec[i_λ] 

        # get sizes 
        x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 
        
        # get x0 from noisy and smoothed data 
        x0_train    = data_train.x_noise[1,:]  
        x0_test     = data_test.x_noise[1,:]  
        
        # ----------------------- # 
        # SINDY-lasso ! 
        Ξ_sindy = sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
        
        # integrate discovered dynamics 
        dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
        x_sindy_train = integrate_euler( dx_fn_sindy, x0_train, data_train.t, data_train.u ) 
        x_sindy_test  = integrate_euler( dx_fn_sindy, x0_test, data_test.t, data_test.u ) 

        # save error norm 
        sindy_train_err = norm( x_sindy_train - data_train.x_noise ) 
        sindy_test_err  = norm( x_sindy_test - data_test.x_noise ) 
        push!( sindy_train_err_hist, sindy_train_err ) 
        push!( sindy_test_err_hist,  sindy_test_err  ) 

    end 

    # save index of minimum error 
    i_min_err = argmin( sindy_train_err_hist ) 
    min_train_err = [ λ_vec[i_min_err] sindy_train_err_hist[i_min_err] sindy_test_err_hist[i_min_err] ] 
    # push!( min_train_err_hist, min_train_err ) 
    min_train_err_hist = [ min_train_err_hist ; min_train_err ] 
    
end 

min_train_err_hist = min_train_err_hist[2:end,:] 

## ============================================ ##

using CSV 

for i in eachindex( csv_files_vec )
    csv_files_vec[i] = replace( csv_files_vec[i], csv_path => "" ) 
end 

header = [ "csv_file", "λ_min", "min_train_err", "--> test_err" ]
data   = [ csv_files_vec min_train_err_hist ] 
df = DataFrame( data, header ) 

# save to csv 
csv_save = string( "min_train_err_hist.csv" ) 
CSV.write( csv_save, df ) 


