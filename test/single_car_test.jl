using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )
csv_file = "rollout_27.csv" 

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
f = plot_λ_err_log( λ_vec, df_λ_vec, df_sindy, df_gpsindy, freq_hz, csv_file ) 
display(f) 








