using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using DataFrames 


## ============================================ ##
# test all the files  

freq_hz  = 50 

df_λ_vec_hist     = [] 
for noise = 0.01 : 0.01 : 0.02
    
    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    df_min_err_hist = cross_validate_csv_path( csv_path, freq_hz, true ) 
    push!( df_λ_vec_hist, df_min_err_hist ) 
end 



## ============================================ ##

freq_hz = 50 
noise   = 0.02 

csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

df_min_err_hist = cross_validate_csv_path( csv_path, freq_hz, true ) 



## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

csv_path_file = string(csv_path, "rollout_8.csv" ) 

# extract data 
data_train, data_test = car_data_struct( csv_path_file ) 

# smooth with GPs 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 

## ============================================ ##

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

# get x0 from noisy and smoothed data 
x0_train    = data_train.x_noise[1,:] ; x0_train_GP = x_train_GP[1,:] 
x0_test     = data_test.x_noise[1,:]  ; x0_test_GP  = x_test_GP[1,:] 


# ----------------------- # 
# SINDY-lasso ! 
λ = 6.0 
Ξ_sindy = sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 

# integrate discovered dynamics 
dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
x_sindy_train = integrate_euler( dx_fn_sindy, x0_train, data_train.t, data_train.u ) 
x_sindy_test  = integrate_euler( dx_fn_sindy, x0_test, data_test.t, data_test.u ) 

# save sindy stats as dataframe 
sindy_train_err = norm( data_train.x_noise - x_sindy_train ) 
sindy_test_err  = norm( data_test.x_noise  - x_sindy_test  ) 
df_min_err_sindy = DataFrame( [ λ sindy_train_err sindy_test_err ], [ "λ", "train_err", "test_err" ] ) 

# optimize GPs with GPs 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test, 0.1, true ) 

# cross-validate gpsindy 
header = [ "λ", "train_err", "test_err" ] 
df_gpsindy = DataFrame( fill( [],3 ), header ) 
for i_λ = eachindex( λ_vec ) 

    λ = λ_vec[i_λ] 
    
    # ----------------------- #
    # GPSINDy-lasso ! 
    Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, data_train.u ) 
    
    # integrate discovered dynamics 
    dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, data_train.t, data_train.u ) 
    x_gpsindy_test  = integrate_euler( dx_fn_gpsindy, x0_test, data_test.t, data_test.u ) 

    push!( df_gpsindy, [ λ norm( data_train.x_noise - x_gpsindy_train ) norm( data_test.x_noise - x_gpsindy_test ) ] ) 

end 

# save gpsindy min err stats 
i_min_gpsindy      = argmin( df_gpsindy.train_err )
λ_min_gpsindy      = df_gpsindy.λ[i_min_gpsindy] 
df_min_err_gpsindy = DataFrame( [ λ_min_gpsindy df_gpsindy.train_err[i_min_gpsindy] df_gpsindy.test_err[i_min_gpsindy] ], [ "λ", "train_err", "test_err" ] ) 

f = Figure() 

for i_x = 1:4 
    ax = Axis( f[i_x,1], xlabel="time [s]", ylabel="x$i_x" ) 
        CairoMakie.scatter!( ax, data_train.t, data_train.x_noise[:,i_x], color=:black, label="noise" )     
        lines!( ax, data_train.t, x_sindy_train[:,i_x], linewidth = 2, label="sindy" ) 
        lines!( ax, data_train.t, x_gpsindy_train[:,i_x], linewidth = 2, label="gpsindy" ) 
end 

display(f) 


























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





## ============================================ ##
# single car test 

freq_hz = 25 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

x_min_err_hist_25hz_σn = cross_validate_csv_path( csv_path )





