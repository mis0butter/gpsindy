using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 10 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

x_min_err_hist = x_train_test_err_struct( [], [], [], [] ) 
for i in eachindex(csv_files_vec) 

    x_err_hist = cross_validate( csv_files_vec[i] ) 

    df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec ) 
    # f = plot_λ_err_log( λ_vec, df_λ_vec, df_sindy, df_gpsindy, freq_hz, csv_file ) 
    # display(f) 
    
    push!( x_min_err_hist.sindy_train, df_sindy.x_sindy_train_err[1] ) 
    push!( x_min_err_hist.sindy_test,  df_sindy.x_sindy_test_err[1]  ) 
    push!( x_min_err_hist.gpsindy_train, df_gpsindy.x_gpsindy_train_err[1] ) 
    push!( x_min_err_hist.gpsindy_test,  df_gpsindy.x_gpsindy_test_err[1]  ) 

end 

println( "sindy_train mean: ", mean(x_min_err_hist.sindy_train) ) 
println( "gpsindy_train mean: ", mean(x_min_err_hist.gpsindy_train) ) 

println( "sindy_test mean: ", mean(x_min_err_hist.sindy_test) ) 
println( "gpsindy_test mean: ", mean(x_min_err_hist.gpsindy_test) ) 


## ============================================ ##








