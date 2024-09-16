using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# control input delay  

# add how much noise to all data? 
noise   = 0.07  
freq_hz = 25 

csv_path  = string( "test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_0/" ) 
save_path = string( "test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" ) 
add_noise_car( csv_path, save_path, noise )  



















## ============================================ ##
## ============================================ ##
# add_noise_car 

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

# for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    i = 17 
    println(i) 

    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 

    # get data 
    t = data[:,1] 
    x = data[:,2:5] 
    u = data[:,6:7] 
    x, dx = unroll( t, x ) 
    fig   = plot_car_x_dx( t, x, dx ) 


    ## add noise to x 
    x_noise = x + noise*randn(size(x)) 
    dx_noise = fdiff(t, x_noise, 2) 

    plot_car_x_dx( t, x_noise, dx_noise ) 

    x_noise, dx_noise = unroll( t, x_noise ) 

    # check how it looks 
    fig   = plot_car_x_dx_noise( t, x, dx, x_noise, dx_noise ) 
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

# end 

## ============================================ ##
# unroll 

    # use forward finite differencing 
    dx_noise = fdiff(t, x_noise, 1) 

    # massage data, generate rollovers  
    rollover_up_ind = findall( x -> x > 10, dx_noise[:,4] ) 
    rollover_dn_ind = findall( x -> x < -10, dx_noise[:,4] ) 

    for i in eachindex(rollover_up_ind) 
        # i = 1 
    
            if rollover_up_ind[i] < rollover_dn_ind[i] 
                i0   = rollover_up_ind[i] + 1 
                ifin = rollover_dn_ind[i]     
                rollover_rng = x_noise[ i0 : ifin , 4 ]
                dθ = π .- rollover_rng 
                θ  = -π .- dθ 
            else 
                i0   = rollover_dn_ind[i] + 1 
                ifin = rollover_up_ind[i]     
                rollover_rng = x_noise[ i0 : ifin , 4 ]
                dθ = π .+ rollover_rng 
                θ  = π .+ dθ     
            end 
            x_noise[ i0 : ifin , 4 ] = θ
    
        end 

    # @infiltrate 
    
    # use central finite differencing now  
    dx_noise = fdiff(t, x_noise, 2) 



