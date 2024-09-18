using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 
using CSV, DataFrames 


## ============================================ ##
## ============================================ ##
# let's look at 50 hz noise = 0.02 rollout_8.csv 

freq_hz = 50 
noise   = 0 
σn      = 0.02 
opt_σn  = false 
interpolate_gp = true 

noise_vec = [] 
push!( noise_vec, 0 ) 
push!( noise_vec, 0.01 ) 
push!( noise_vec, 0.02 ) 

for noise = noise_vec  
    for σn = [ 0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.1, 0.2, 0.3 ] 
        df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn, interpolate_gp ) 
    end 
end 


## ============================================ ## 

for freq_hz = 10 

    noise_vec = [] 
    push!( noise_vec, 0 ) 
    push!( noise_vec, 0.01 ) 
    push!( noise_vec, 0.02 ) 

    for noise = noise_vec 

        for σn = [ 0.001, 0.002, 0.003, 0.03, 0.3 ] 
            
            opt_σn  = false  
            df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
        
            opt_σn  = true   
            df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, df_mean_err = cross_validate_csv_path( freq_hz, noise, σn, opt_σn ) 
        
        end 

    end 

end 


## ============================================ ##
## ============================================ ##
# let's compile all the stats of the files that I just made 

csv_path_up = string( "test/results/jake_car_csvs_ctrlshift_no_trans/interpolate_gp/" ) 
csv_path_vec = readdir( csv_path_up ) 

header = [ "freq_hz", "noise", "σn", "opt_σn", "mean_err_sindy_train", "mean_err_gpsindy_train", "mean_err_sindy_test", "mean_err_gpsindy_test" ]
df_mean_err_all = DataFrame( fill( [], 8 ), header ) 
for i = eachindex(csv_path_vec) 
# i = 1 

    csv_path = string( csv_path_up, csv_path_vec[i] ) 
    csv_path_df = string( csv_path, "/df_mean_err.csv" )

    # read in the csv 
    if isfile( csv_path_df )
        df_mean_err = CSV.read( csv_path_df, DataFrame ) 
        push!( df_mean_err_all, df_mean_err[1,:] )             
    end 

end 

# save the dataframe 
CSV.write( "df_mean_err_all_interpolate_gp.csv", df_mean_err_all) 

















## ============================================ ##
## ============================================ ##
 
# let's break it out 

freq_hz = 5 
noise   = 0 
σ_n     = 0.02  
opt_σn  = false 

csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path_σn( csv_path, σn, opt_σn ) 

i_csv = 1 

csv_path_file = csv_files_vec[i_csv] 

# extract data 
data_train, data_test = make_data_structs( csv_path_file ) 

# x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = smooth_data_gp( data_train, data_test, σn, opt_σn ) 
t_train_dbl, u_train_dbl, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_double_test( data_train, data_test, σ_n, opt_σn ) 



## ============================================ ##
# let's plot raw vs GP 

f = Figure( size = ( 800,800 ) ) 

for i_x = 1 : 4 

    ax = Axis( f[i_x,1], xlabel="t", ylabel="x$i_x" ) 
    scatter!( ax, data_train.t, data_train.x_noise[:,i_x], color = :black, label="x$i_x" ) 
    lines!( ax, data_train.t, data_train.x_noise[:,i_x], color = :black ) 
    scatter!( ax, t_train_dbl, x_train_GP[:,i_x], color = :red, markersize = 5, label="x$i_x GP" ) 
    # lines!( ax, t_train_dbl, x_train_GP[:,i_x], linestyle = :dash, color = :red ) 
    axislegend() 

    ax = Axis( f[i_x,2], xlabel="t", ylabel="dx$i_x" ) 
    scatter!( ax, data_train.t, data_train.dx_noise[:,i_x], color = :black, label="x$i_x" ) 
    lines!( ax, data_train.t, data_train.dx_noise[:,i_x], color = :black ) 
    scatter!( ax, t_train_dbl, dx_train_GP[:,i_x], color = :red, markersize = 5, label="x$i_x GP" ) 
    # lines!( ax, t_train_dbl, dx_train_GP[:,i_x], color = :red ) 
    axislegend() 
    
end 

display(f) 

## ============================================ ##

# cross-validate gpsindy 
λ_vec      = λ_vec_fn() 

# get x0 from noisy and smoothed data 
x0_train = data_train.x_noise[1,:]  
x0_test  = data_test.x_noise[1,:]  

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 
# i_λ = 25 ; λ = λ_vec[i_λ] 
λ = 80 

# ----------------------- #
# sindy!!! 
x_sindy_train, x_sindy_test = sindy_lasso_int( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

# ----------------------- #
# gpsindy!!! 
Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, u_train_dbl ) 

# integrate discovered dynamics 
dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
# x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, data_train.t, data_train.u ) 
x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, t_train_dbl, u_train_dbl ) 
x_gpsindy_test  = integrate_euler( dx_fn_gpsindy, x0_test, data_test.t, data_test.u )

## ============================================ ##
# let's plot some stuff 

f = Figure( size = ( 800,800 ) ) 

for i_x = 1 : 4 
    ax = Axis( f[i_x,1], xlabel="t", ylabel="x$i_x" ) 
        scatter!( ax, data_train.t, data_train.x_noise[:,i_x], color = :black, label="x1" ) 
        scatter!( ax, t_train_dbl, x_train_GP[:,i_x], color = :red, label="GP" ) 
        lines!( ax, data_train.t, x_sindy_train[:,i_x], label="sindy" ) 
        lines!( ax, t_train_dbl, x_gpsindy_train[:,i_x], label="gpsindy" ) 
        axislegend() 
end 

display(f) 

f = Figure( size = ( 800,800 ) ) 

for i_x = 1 : 4 
    ax = Axis( f[i_x,1], xlabel="t", ylabel="x$i_x" ) 
        scatter!( ax, data_test.t, data_test.x_noise[:,i_x], color = :black, label="x1" ) 
        # scatter!( ax, t_train_dbl, dx_train_GP[:,i_x], color = :red, label="GP" ) 
        lines!( ax, data_test.t, x_sindy_test[:,i_x], label="sindy" ) 
        lines!( ax, data_test.t, x_gpsindy_test[:,i_x], label="gpsindy" ) 
        axislegend() 
end 

display(f) 

# ----------------------- #
# gpsindy!!! 
# x_gpsindy_train, x_gpsindy_test = sindy_lasso_int( x_train_GP, dx_train_GP, λ, data_train, data_test ) 

# ----------------------- #
# metrics 

println( "sindy err = ", norm( data_test.x_noise - x_sindy_test ) )  
println( "gpsindy err = ", norm( data_test.x_noise - x_gpsindy_test ) ) 

