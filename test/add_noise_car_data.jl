using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# control input delay  

# add how much noise to all data? 
noise = 0.01 

csv_path  = "test/data/jake_car_csvs_control_adjust/" 
save_path = replace( csv_path, "_adjust" => string( "_adjust_50hz_noise_", noise ) ) 
fig_save_path = string( save_path, "figs/"  ) 
csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

# check if save_path exists 
if !isdir( save_path ) 
    mkdir( save_path ) 
end 
if !isdir( fig_save_path ) 
    mkdir( fig_save_path ) 
end

for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 

    # get data 
    t = data[:,1] 
    x = data[:,2:5] 
    u = data[:,6:7] 
    x, dx = unroll( t, x ) 

    ## add noise to x 
    x_noise = x + noise*randn(size(x)) 
    x_noise, dx_noise = unroll( t, x_noise ) 

    # check how it looks 
    fig   = plot_car_x_dx_noise( t, x, dx, x_noise, dx ) 
    fig_path = replace( csv_file, csv_path => fig_save_path ) 
    fig_path = replace( fig_path, ".csv" => ".png" ) 
    save( fig_path , fig ) 

    # get header from dataframe 
    header = names(df) 

    # create dataframe 
    data_noise = [ t x_noise u ] 
    df_noise   = DataFrame( data_noise,  header )  

    # save noisy data 
    data_save      = replace( csv_file, csv_path => "" ) 
    data_save      = replace( data_save, ".csv" => string( "_noise", ".csv" ) ) 
    data_save_path = string( save_path, data_save ) 
    CSV.write( data_save_path, df_noise ) 

end 



