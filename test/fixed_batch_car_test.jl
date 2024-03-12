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
mean_5hz_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for noise = 0.01 : 0.01 : 0.1 

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    x_min_err_hist, df_λ_vec = cross_validate_csv_path( csv_path, freq_hz, true ) 
    
    sindy_train_err_mean   = mean( x_min_err_hist.sindy_train   ) 
    sindy_test_err_mean    = mean( x_min_err_hist.sindy_test    ) 
    gpsindy_train_err_mean = mean( x_min_err_hist.gpsindy_train ) 
    gpsindy_test_err_mean  = mean( x_min_err_hist.gpsindy_test  )     

    push!( df_λ_vec_hist, df_λ_vec ) 
    push!( mean_5hz_err_hist.sindy_train,   sindy_train_err_mean    ) 
    push!( mean_5hz_err_hist.sindy_test,    sindy_test_err_mean     ) 
    push!( mean_5hz_err_hist.gpsindy_train, gpsindy_train_err_mean  ) 
    push!( mean_5hz_err_hist.gpsindy_test,  gpsindy_test_err_mean   ) 

end 


## ============================================ ##
# let's break sindy 

freq_hz  = 50 

df_λ_vec_hist     = [] 
mean_5hz_err_hist = x_train_test_err_struct( [], [], [], [] ) 
# for noise = 0.01 : 0.01 : 0.1 
noise = 0.01 

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    x_min_err_hist, df_λ_vec = cross_validate_csv_path( csv_path, freq_hz, true ) 
    
    sindy_train_err_mean   = mean( x_min_err_hist.sindy_train   ) 
    sindy_test_err_mean    = mean( x_min_err_hist.sindy_test    ) 
    gpsindy_train_err_mean = mean( x_min_err_hist.gpsindy_train ) 
    gpsindy_test_err_mean  = mean( x_min_err_hist.gpsindy_test  )     

    push!( df_λ_vec_hist, df_λ_vec ) 
    push!( mean_5hz_err_hist.sindy_train,   sindy_train_err_mean    ) 
    push!( mean_5hz_err_hist.sindy_test,    sindy_test_err_mean     ) 
    push!( mean_5hz_err_hist.gpsindy_train, gpsindy_train_err_mean  ) 
    push!( mean_5hz_err_hist.gpsindy_test,  gpsindy_test_err_mean   ) 

# end 







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

## ============================================ ##

header = [ "λ" "train_err" "test_err" ]
df = DataFrame( min_train_err_hist, header ) 

# convert any matrix to float64 
# vv2m( min_train_err_hist ) 





## ============================================ ##
# single car test 

freq_hz = 25 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

x_min_err_hist_25hz_σn = cross_validate_csv_path( csv_path )





