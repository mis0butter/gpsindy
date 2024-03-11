using GaussianSINDy 
using LinearAlgebra 
using Statistics 
using CairoMakie 
using Printf 


## ============================================ ##
# test single file 

freq_hz  = 50 
csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz/" )

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

for i in eachindex(csv_files_vec) 

    x_err_hist = cross_validate( csv_files_vec[i] ) 

    df_λ_vec, df_sindy, df_gpsindy = df_metrics( x_err_hist, λ_vec ) 
    f = plot_λ_err_log( λ_vec, df_λ_vec, df_sindy, df_gpsindy, freq_hz, csv_file ) 
    display(f)     

end 



## ============================================ ##








