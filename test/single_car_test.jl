using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )
csv_file = "rollout_26.csv" 

# extract data 
data_train, data_test = car_data_struct( string(csv_path, csv_file) ) 

# smooth with GPs 
x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = gp_train_test( data_train, data_test ) 


# ----------------------- #
# test cross_validate_λ for sindy and gpsindy 

x_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for i_λ = eachindex( λ_vec ) 

    λ   = λ_vec[i_λ] 
    println( "λ = ", @sprintf "%.3g" λ ) 

    data_pred_train, data_pred_test = sindy_gpsindy_λ( data_train, data_test, x_train_GP, dx_train_GP, x_test_GP, λ ) 
    
    # plot and save metrics     
    f = plot_err_train_test( data_pred_train, data_pred_test, data_train, data_test, λ, freq_hz, csv_file)     
    display(f) 

    x_err_hist = push_err_metrics( x_err_hist, data_train, data_test, data_pred_train, data_pred_test ) 
    
end 

df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec ) 


## ============================================ ##
# plot df metrics 

i1 = 7 
i2 = 16 

f = Figure( size = ( 700,700 ) ) 

ax = Axis( f[1,1], xscale = log10, title = "err λ = 1e-6 : 1" ) 
    sindy_train = lines!( ax, λ_vec[1:i1], df_λ_vec.x_sindy_train_err[1:i1], color = :blue, label="sindy train" ) 
ax = Axis( f[2,1], xscale = log10 )  
    sindy_test = lines!( ax, λ_vec[1:i1], df_λ_vec.x_sindy_test_err[1:i1], color = :green, label="sindy test" ) 
ax = Axis( f[3,1], xscale = log10 ) 
    gpsindy_train = lines!( ax, λ_vec[1:i1], df_λ_vec.x_gpsindy_train_err[1:i1], color = :red, label="gpsindy train" ) 
ax = Axis( f[4,1], xlabel="λ", xscale = log10 ) 
    gpsindy_test = lines!( ax, λ_vec[1:i1], df_λ_vec.x_gpsindy_test_err[1:i1], color = :orange, label="gpsindy test" ) 


ax = Axis( f[1,2], title = "err λ = 1 : 10" ) 
    lines!( ax, λ_vec[i1:i2], df_λ_vec.x_sindy_train_err[i1:i2],color = :blue, label="sindy" ) 
ax = Axis( f[2,2], ) 
    lines!( ax, λ_vec[i1:i2], df_λ_vec.x_sindy_test_err[i1:i2], color = :green, label="sindy" ) 
ax = Axis( f[3,2], ) 
    lines!( ax, λ_vec[i1:i2], df_λ_vec.x_gpsindy_train_err[i1:i2], color = :red, label="gpsindy" ) 
ax = Axis( f[4,2], xlabel="λ" ) 
    lines!( ax, λ_vec[i1:i2], df_λ_vec.x_gpsindy_test_err[i1:i2], color = :orange, label="gpsindy" ) 

ax = Axis( f[1,3], title = "err λ = 10 : 100" ) 
    lines!( ax, λ_vec[i2:end], df_λ_vec.x_sindy_train_err[i2:end],color = :blue, label="sindy" ) 
ax = Axis( f[2,3] ) 
    lines!( ax, λ_vec[i2:end], df_λ_vec.x_sindy_test_err[i2:end], color = :green, label="sindy" ) 
ax = Axis( f[3,3] ) 
    lines!( ax, λ_vec[i2:end], df_λ_vec.x_gpsindy_train_err[i2:end], color = :red, label="gpsindy" ) 
ax = Axis( f[4,3], xlabel = "λ" ) 
    lines!( ax, λ_vec[i2:end], df_λ_vec.x_gpsindy_test_err[i2:end],color = :orange, label="gpsindy" ) 

# horizontal legend 
Legend( f[5,1:3], [ sindy_train, sindy_test, gpsindy_train, gpsindy_test ], [ "sindy train", "sindy test", "gpsindy train", "gpsindy test" ], orientation = :horizontal)
    
display(f) 







