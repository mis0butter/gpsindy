

## ============================================ ## 

function extract_kernel_components(temp_string)
    temp_string = split(temp_string, "\n")[1] 
    temp_string = split(temp_string, "{Float64}") 
    kernel_1 = temp_string[1] 
    kernel_2 = temp_string[2] 
    kernel_1 = replace(kernel_1, r"[,\s]" => "")
    kernel_2 = replace(kernel_2, r"[,\s]" => "")
    return kernel_1, kernel_2
end

export print_kernel  
function print_kernel(gp)

    # Extract the kernel type as a string and get the portion after the type name
    full_kernel_string = string(gp.kernel)

    # Extract the kernel type and handle composite kernels
    if occursin("Sum", full_kernel_string) 

        temp_string = split(full_kernel_string, "SumKernel{")[2] 
        kernel_1, kernel_2 = extract_kernel_components(temp_string)

        println("Kernel type = ", "Sum" ) 
        println("Kernel 1: ", kernel_1) 
        println("Kernel 2: ", kernel_2)  

    elseif occursin("Prod", full_kernel_string) 

        temp_string = split(full_kernel_string, "ProdKernel{")[2] 
        kernel_1, kernel_2 = extract_kernel_components(temp_string)
        
        println("Kernel type = ", "Product" ) 
        println("Kernel 1: ", kernel_1) 
        println("Kernel 2: ", kernel_2)  

    else # single kernel 

        temp_string = split(full_kernel_string, "Type: ")[2] 
        kernel = split(temp_string, "{Float64}")[1] 

        println("Kernel type = ", kernel )  

    end
end 

## ============================================ ## 

export interpolate_array  
function interpolate_array(x_orig::Union{Vector{Float64}, Matrix{Float64}}, interp_factor::Int)

    if x_orig isa Vector
        return interpolate_vector(x_orig, interp_factor)
    elseif x_orig isa Matrix
        return interpolate_matrix(x_orig, interp_factor)
    end

end

function interpolate_vector(x_orig::Vector{Float64}, interp_factor::Int)

    size_interp = interp_factor * length(x_orig) - ( interp_factor - 1 )
    x_interp    = Vector{Float64}(undef, size_interp)

    for i in 1:length(x_orig) - 1 

        dx        = x_orig[i + 1] - x_orig[i]
        dx_interp = dx / interp_factor

        for j in 1:interp_factor
            i_interp = interp_factor * (i - 1) + j
            dt_add   = (j - 1) * dx_interp
            x_interp[i_interp] = x_orig[i] + dt_add
        end
    end 
    x_interp[end] = x_orig[end] 

    return x_interp
end

function interpolate_matrix(x_orig::Matrix{Float64}, interp_factor::Int)

    rows, cols  = size(x_orig)
    rows_interp = interp_factor * rows - ( interp_factor - 1 ) 
    x_interp    = Matrix{Float64}(undef, rows_interp, cols)

    for col in 1:cols
        x_interp[:, col] = interpolate_vector(x_orig[:, col], interp_factor)
    end

    return x_interp
end


## ============================================ ## 

export interp_dbl_fn 

function interp_dbl_fn( u ) 

    u_col = size(u, 1) 

    u_interp = [] ; du = 0 
    for i = 1 : u_col - 1 
    
        du = ( u[i+1,:] - u[i,:] ) ./ 2 
    
        push!( u_interp, u[i,:] ) 
        push!( u_interp, u[i,:] .+ du ) 
    
    end 
    push!( u_interp, u[u_col,:] ) 
    push!( u_interp, u[u_col,:] .+ du )
    
    u_interp = vv2m( u_interp )     

    return u_interp 
end 


## ============================================ ##

export t_double_fn 

function t_double_fn( t ) 

    t_double = Float64[ ] ; dt = 0 ; 
    for i in 1 : length(t) - 1 
        dt = ( t[i+1] - t[i] ) / 2 
        push!( t_double, t[i] ) 
        push!( t_double, t[i] + dt ) 
    end 
    push!( t_double, t[end] ) 
    push!( t_double, t[end] + dt ) 

    return t_double 
end 


## ============================================ ##

export df_mean_err_fn 

function df_mean_err_fn( df_min_err_csvs_sindy, df_min_err_csvs_gpsindy, freq_hz, noise, σn, opt_σn ) 

    # save mean min err which is what we care about 
    mean_err_sindy_test    = mean( df_min_err_csvs_sindy.test_err ) 
    mean_err_gpsindy_test  = mean( df_min_err_csvs_gpsindy.test_err ) 
    mean_err_sindy_train   = mean( df_min_err_csvs_sindy.train_err ) 
    mean_err_gpsindy_train = mean( df_min_err_csvs_gpsindy.train_err )  

    header = [ "freq_hz", "noise", "σn", "opt_σn", "mean_err_sindy_train", "mean_err_gpsindy_train", "mean_err_sindy_test", "mean_err_gpsindy_test" ] 

    data = [ freq_hz, noise, σn, opt_σn, mean_err_sindy_train, mean_err_gpsindy_train, mean_err_sindy_test, mean_err_gpsindy_test ] 

    df_mean_err = DataFrame( fill( [], 8 ), header )
    push!( df_mean_err, data )  
    
    # report mean error for testing 
    println( "mean_err_sindy_test   = ", mean_err_sindy_test) 
    println( "mean_err_gpsindy_test = ", mean_err_gpsindy_test )

    return df_mean_err 
end 


## ============================================ ##

export df_min_err_fn 

function df_min_err_fn( df_gpsindy, csv_path_file ) 
    
    csv_string = split( csv_path_file, "/" )
    csv_file   = csv_string[end]  

    header = [ "csv_file", "λ_min", "train_err", "test_err", "train_traj", "test_traj" ] 

    # save gpsindy min err stats 
    i_min_gpsindy = argmin( df_gpsindy.train_err )
    λ_min_gpsindy = df_gpsindy.λ[i_min_gpsindy]
    data          = [ csv_file, λ_min_gpsindy, df_gpsindy.train_err[i_min_gpsindy], df_gpsindy.test_err[i_min_gpsindy], df_gpsindy.train_traj[i_min_gpsindy], df_gpsindy.test_traj[i_min_gpsindy] ]  
    df_min_err_gpsindy = DataFrame( fill( [], 6 ), header ) 
    push!( df_min_err_gpsindy, data ) 

    return df_min_err_gpsindy 
end 


## ============================================ ##

export mkdir_save_path_σn 

function mkdir_save_path_σn( csv_path, σn, opt_σn, interpolate_gp = false ) 

    csv_files_vec = readdir( csv_path ) 
    deleteat!( csv_files_vec, findall( csv_files_vec .== "figs" ) ) 

    for i in eachindex(csv_files_vec)  
        csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
    end 

    save_path  = replace( csv_path, "/data/" => "/results/" ) 
    GP_no_intp = split( save_path, "/" )[3]  
    if interpolate_gp == true 
        save_path = replace( save_path, GP_no_intp => string( GP_no_intp, "/interpolate_gp" ) ) 
    else 
        save_path = replace( save_path, GP_no_intp => string( GP_no_intp, "/no_intp" ) ) 
    end 
    csv_dir     = split( save_path, "/" )[end-1] 
    csv_dir_new = string( csv_dir, "_σn_", σn, "_opt_", opt_σn )  
    save_path   = replace( save_path, csv_dir => csv_dir_new ) 
    if !isdir( save_path ) 
        mkdir( save_path ) 
    end  

    save_path_fig = string( save_path, "figs/" ) 
    if !isdir( save_path_fig ) 
        mkdir( save_path_fig ) 
    end 

    save_path_dfs = string( save_path, "dfs/" ) 
    if !isdir( save_path_dfs ) 
        mkdir( save_path_dfs ) 
    end 

    return csv_files_vec, save_path, save_path_fig, save_path_dfs 
end 


## ============================================ ##

export mkdir_save_path 

function mkdir_save_path( csv_path ) 

    csv_files_vec = readdir( csv_path ) 
    deleteat!( csv_files_vec, findall( csv_files_vec .== "figs" ) ) 

    for i in eachindex(csv_files_vec)  
        csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
    end 

    save_path = replace( csv_path, "/data/" => "/results/" ) 
    if !isdir( save_path ) 
        mkdir( save_path ) 
    end  

    save_path_fig = string( save_path, "figs/" ) 
    if !isdir( save_path_fig ) 
        mkdir( save_path_fig ) 
    end 

    save_path_dfs = string( save_path, "dfs/" ) 
    if !isdir( save_path_dfs ) 
        mkdir( save_path_dfs ) 
    end 

    return csv_files_vec, save_path, save_path_fig, save_path_dfs 
end 


## ============================================ ##

export df_metrics 

function df_metrics( x_err_hist, λ_vec, csv_path, csv_file_path ) 

    csv_file = replace( csv_file_path, csv_path => "" ) 

    # csv_file_str = fill( csv_file, length(λ_vec) ) 

    header = [ "λ", "x_sindy_train_err", "x_gpsindy_train_err", "x_sindy_test_err", "x_gpsindy_test_err" ] 
    data = [ λ_vec x_err_hist.sindy_train x_err_hist.gpsindy_train x_err_hist.sindy_test x_err_hist.gpsindy_test ] 
    # data = convert( Matrix{Float64}, data ) 
    df_λ_vec = DataFrame( data, header )  
    
    # get indices with smallest error for TRAINING data 
    i_min_sindy   = argmin( x_err_hist.sindy_train ) 
    i_min_gpsindy = argmin( x_err_hist.gpsindy_train ) 
    
    # print above as data  frame 
    header = [ "csv_file", "λ_min_sindy", "x_sindy_train_err", "x_sindy_test_err" ]
    data   = [ csv_file λ_vec[i_min_sindy] x_err_hist.sindy_train[i_min_sindy] x_err_hist.sindy_test[i_min_sindy] ] 
    df_sindy = DataFrame( data, header )  
    
    header = ["csv_file", "λ_min_gpsindy", "x_gpsindy_train_err", "x_gpsindy_test_err" ] 
    data   = [ csv_file λ_vec[i_min_gpsindy] x_err_hist.gpsindy_train[i_min_gpsindy] x_err_hist.gpsindy_test[i_min_gpsindy] ] 
    df_gpsindy = DataFrame( data, header )   

    return df_λ_vec, df_sindy, df_gpsindy 
end 

## ============================================ ##

export push_err_metrics 

function push_err_metrics( x_err_hist, data_train, data_test, data_pred_train, data_pred_test ) 

    x_sindy_train_err   = norm( data_train.x_noise - data_pred_train.x_sindy ) 
    x_gpsindy_train_err = norm( data_train.x_noise - data_pred_train.x_gpsindy ) 
    push!( x_err_hist.sindy_train, x_sindy_train_err ) 
    push!( x_err_hist.gpsindy_train, x_gpsindy_train_err ) 

    x_sindy_test_err   = norm( data_test.x_noise - data_pred_test.x_sindy ) 
    x_gpsindy_test_err = norm( data_test.x_noise - data_pred_test.x_gpsindy ) 
    push!( x_err_hist.sindy_test, x_sindy_test_err ) 
    push!( x_err_hist.gpsindy_test, x_gpsindy_test_err ) 

    return x_err_hist 
end 


## ============================================ ##

function extract_preds( data_pred_train, data_pred_test ) 

    x_train_GP      = data_pred_train.x_GP 
    x_sindy_train   = data_pred_train.x_sindy 
    x_gpsindy_train = data_pred_train.x_gpsindy 
    x_test_GP       = data_pred_test.x_GP 
    x_sindy_test    = data_pred_test.x_sindy 
    x_gpsindy_test  = data_pred_test.x_gpsindy 

    return x_train_GP, x_sindy_train, x_gpsindy_train, x_test_GP, x_sindy_test, x_gpsindy_test 
end 

export extract_preds 

## ============================================ ##

function min_max_y( input, percent = 30 )  

    min_y   = minimum( input ) ; max_y = maximum( input ) 
    range_y = abs.(max_y - min_y)  
    max_y   = max_y + percent/100 * range_y 
    min_y   = min_y - percent/100 * range_y 

    return min_y, max_y  
end 

export min_max_y 


## ============================================ ## 

function downsample_to_original(t_orig, t_interp, x_interp) 
    
    # Initialize arrays to store downsampled data
    t_downsampled = similar(t_orig)
    x_downsampled = similar(x_interp, size(t_orig, 1), size(x_interp, 2))
    
    # Iterate through original time points
    for (i, t) in enumerate(t_orig)
        
        # Find the index in t_interp that is closest to t
        _, idx = findmin(abs.(t_interp .- t))
        
        # Store the corresponding time and state values
        t_downsampled[i] = t_interp[idx]
        x_downsampled[i, :] = x_interp[idx, :]
    end
    
    return t_downsampled, x_downsampled 
end 

export downsample_to_original  


## ============================================ ##
# SHOULD DEPRECATE THIS  

function downsample( t, t_interp, x_interp, interp_factor ) 

    if t == t_interp 

        return t_interp, x_interp 

    else 

        t_dbl_dwnsmpl = Float64[] 
        x_dbl_dwnsmpl = []
        for i in eachindex(t) 
            # println(i) 
            push!( t_dbl_dwnsmpl, t_interp[interp_factor * i - 1] ) 
            push!( x_dbl_dwnsmpl, x_interp[interp_factor * i - 1, :] ) 
        end 

        x_dbl_dwnsmpl = vv2m( x_dbl_dwnsmpl ) 

        return t_dbl_dwnsmpl, x_dbl_dwnsmpl 

    end 

end 

export downsample 


## ============================================ ##

function add_noise_car( csv_path, save_path, noise )  

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
        # x_noise, dx_noise = unroll( t, x_noise ) 
        dx_noise = fdiff(t, x_noise, 2) 
    
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
        data_save_path = string( save_path, data_save ) 
        CSV.write( data_save_path, df_noise ) 
    
    end 

end 

export add_noise_car 


## ============================================ ##
# save data at certain rate 

function Fhz_data( t, x, u, F_hz_des, F_hz_OG = 50 ) 

    N = size(x, 1) 

    x_Fhz_mat = zeros(1,13) 
    u_Fhz_mat = zeros(1,4) 
    t_Fhz_mat = zeros(1) 
    for i = 1 : Int(F_hz_OG / F_hz_des) : N      # assuming the quadcopter data is already at 100 Hz - so we can just take every 100 / F_hz-th point 
        x_Fhz = x[i,:] 
        u_Fhz = u[i,:] 
        t_Fhz = t[i] 
        if i == 1 
            x_Fhz_mat = x_Fhz' 
            u_Fhz_mat = u_Fhz' 
            t_Fhz_mat = t_Fhz 
        else 
            x_Fhz_mat = vcat( x_Fhz_mat, x_Fhz' ) 
            u_Fhz_mat = vcat( u_Fhz_mat, u_Fhz' ) 
            t_Fhz_mat = vcat( t_Fhz_mat, t_Fhz ) 
        end 
    end

    return t_Fhz_mat, x_Fhz_mat, u_Fhz_mat 
end 

export Fhz_data 


## ============================================ ##
# reject outliers 

function reject_outliers( data ) 

    # reject 3-sigma outliers 
    sigma3 = data[ data .< ( mean( data ) + 3*std( data ) ) ] 

    return sigma3 
end 

export reject_outliers 


## ============================================ ##
# for cross-validation 

export λ_vec_fn 
function λ_vec_fn(  ) 

    λ_vec = [ 1e-6 ] 
    while λ_vec[end] <= 1e-1 
        push!( λ_vec, 10.0 * λ_vec[end] ) 
    end 
    while round(λ_vec[end], digits = 3) < 1.0  
        push!( λ_vec, 0.1 + λ_vec[end] ) 
    end 
    while round(λ_vec[end], digits = 3) < 10 
        push!( λ_vec, 1.0 + λ_vec[end] ) 
    end 
    while λ_vec[end] < 100 
        push!( λ_vec, 10.0 + λ_vec[end] ) 
    end
    
    return λ_vec 
end 


## ============================================ ##

export datastruct_to_train_test 
function datastruct_to_train_test( data_train, data_test ) 

    t_train        = data_train.t 
    u_train        = data_train.u 
    x_train_true   = data_train.x_true 
    dx_train_true  = data_train.dx_true 
    x_train_noise  = data_train.x_noise 
    dx_train_noise = data_train.dx_noise 

    t_test         = data_test.t 
    u_test         = data_test.u 
    x_test_true    = data_test.x_true 
    dx_test_true   = data_test.dx_true 
    x_test_noise   = data_test.x_noise 
    dx_test_noise  = data_test.dx_noise 
    
    return t_train, u_train, x_train_true, dx_train_true, x_train_noise, dx_train_noise, t_test, u_test, x_test_true, dx_test_true, x_test_noise, dx_test_noise 
end 


## ============================================ ##
# extract Jake's car data, export as structs 

export make_data_structs 
function make_data_structs( csv_file ) 

    t, x, u = extract_car_data( csv_file ) 
    x_vars, u_vars, poly_order, n_vars = size_vars( x, u ) 
    x, dx_fd = unroll( t, x ) 
    
    # split into training and test data 
    test_fraction = 0.3 
    portion       = "last" 
    u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
    t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
    t_train = t_train[:] ; t_test = t_test[:] 
    x_train_noise,  x_test_noise  = split_train_test( x, test_fraction, portion ) 
    dx_train_noise, dx_test_noise = split_train_test( dx_fd, test_fraction, portion ) 
    
    data_train = data_struct( t_train, u_train, [], [], x_train_noise, dx_train_noise ) 
    data_test  = data_struct( t_test, u_test, [], [], x_test_noise, dx_test_noise) 
    
    return data_train, data_test 
end 


## ============================================ ##
# extract Jake's car data 

export extract_car_data 
function extract_car_data( csv_file ) 

    # wrap in data frame --> Matrix 
    df   = CSV.read(csv_file, DataFrame) 
    data = Matrix(df) 
    
    # extract variables 
    t = data[:,1] 
    x = data[:,2:end-2]
    u = data[:,end-1:end] 

    return t, x, u 
end 


## ============================================ ##
# get sizes of things 

export size_vars 
function size_vars( x, u = false ) 

    x_vars = size(x, 2)
        
    if isequal(u, false)      # if u_data = false 
        u_vars = 0  
    else            # there are u_data inputs 
        u_vars = size(u, 2) 
    end 

    n_vars = x_vars + u_vars 
    poly_order = x_vars 

    return x_vars, u_vars, poly_order, n_vars 
end 


## ============================================ ##
# rollover indices 

export unroll 
function unroll( t, x ) 

    # use forward finite differencing 
    dx = fdiff(t, x, 1) 

    # massage data, generate rollovers  
    rollover_up_ind = findall( x -> x > 10, dx[:,4] ) 
    rollover_dn_ind = findall( x -> x < -10, dx[:,4] ) 

    up_length = length(rollover_up_ind) 
    dn_length = length(rollover_dn_ind) 

    ind_min = minimum( [ up_length, dn_length ] ) 

    for i in 1 : ind_min  
        
        if rollover_up_ind[i] < rollover_dn_ind[i] 
            i0   = rollover_up_ind[i] + 1 
            ifin = rollover_dn_ind[i]     
            rollover_rng = x[ i0 : ifin , 4 ]
            dθ = π .- rollover_rng 
            θ  = -π .- dθ 
        else 
            i0   = rollover_dn_ind[i] + 1 
            ifin = rollover_up_ind[i]     
            rollover_rng = x[ i0 : ifin , 4 ]
            dθ = π .+ rollover_rng 
            θ  = π .+ dθ     
        end 
        x[ i0 : ifin , 4 ] = θ

    end 

    if up_length > dn_length 
        i0   = rollover_up_ind[end] + 1 
        rollover_rng = x[ i0 : end , 4 ]
        dθ = π .- rollover_rng 
        θ  = -π .- dθ 
        x[ i0 : end , 4 ] = θ
    elseif up_length < dn_length 
        i0   = rollover_dn_ind[end] + 1 
        rollover_rng = x[ i0 : end , 4 ]
        dθ = π .+ rollover_rng 
        θ  = π .+ dθ     
        x[ i0 : end , 4 ] = θ
    end 
    
    # use central finite differencing now  
    dx = fdiff(t, x, 2) 

    return x, dx
end 


## ============================================ ##
# standardize data for x and dx 

export stand_data 
function stand_data( t, x ) 

    n_vars = size(x, 2) 
    
    # loop through states 
    x_stand = 0 * x 
    for i = 1:n_vars 
        x_stand[:,i] = ( x[:,i] .- mean( x[:,i] ) ) ./ std( x[:,i] )
    end 
    
    return x_stand 
end 


## ============================================ ##
# convert vector of vectors into matrix 

export vv2m 
function vv2m( vecvec )

    mat = mapreduce(permutedims, vcat, vecvec)

    return mat 
end 


## ============================================ ##
# split into training and testing data based on N points 

function split_train_test_Npoints( t, x, dx, u, N_train ) 

    t_train  = t[ 1:N_train ] 
    x_train  = x[ 1:N_train, : ] 
    dx_train = dx[ 1:N_train, : ] 
    u_train  = u[ 1:N_train, : ] 
    
    t_test   = t[ N_train+1:end ] 
    x_test   = x[ N_train+1:end, : ] 
    dx_test  = dx[ N_train+1:end, : ] 
    u_test   = u[ N_train+1:end, : ] 

    return t_train, t_test, x_train, x_test, dx_train, dx_test, u_train, u_test  
end 

export split_train_test_Npoints 


## ============================================ ##
# split into training and validation data 

export split_train_test 
function split_train_test(x, test_fraction, portion = "last")

    # if test data = LAST portion 
    if portion == "last"

        ind = Int(round( size(x,1) * (1 - test_fraction) ))   

        x_train = x[1:ind,:]
        x_test  = x[ind:end,:] 

    # if test data = FIRST portion 
    elseif portion == "first" 

        ind = Int(round( size(x,1) * (test_fraction) ))  

        x_train = x[ind:end,:] 
        x_test  = x[1:ind,:]

    # test data is in MIDDLE portion 
    else 

        println( "function is broken for middle portion, fix" ) 

        ind1 = Int(round( size(x,1) * (test_fraction*( portion-1 )) )) 
        ind2 = Int(round( size(x,1) * (test_fraction*( portion )) )) 

        x_test  = x[ ind1:ind2,: ]
        x_train = [ x[ 1:ind1,: ] ; x[ ind2:end,: ] ]

    end 

    return x_train, x_test 

end 


## ============================================ ##

export min_d_max
function min_d_max( x )

    xmin = round( minimum(x), digits = 1 )  
    xmax = round( maximum(x), digits = 1 )
    # xmin = minimum(x) 
    # xmax = maximum(x)  
    # dx   = round( ( xmax - xmin ) / 2, digits = 1 ) 
    dx = ( xmax - xmin ) / 2 

    if dx == 0 
        dx = 1 
    end 

    return xmin, dx, xmax  

end 


## ============================================ ##
# derivatives: finite difference  

export fdiff 
function fdiff(t, x, fd_method) 

    # forward finite difference 
    if fd_method == 1 

        dx_fd = 0*x 
        for i = 1 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )
        end 

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # central finite difference 
    elseif fd_method == 2 

        dx_fd = 0*x 
        for i = 2 : length(t)-1
            dx_fd[i,:] = ( x[i+1,:] - x[i-1,:] ) / ( t[i+1] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

        # deal with last index 
        dx_fd[end,:] = ( x[end,:] - x[end-1,:] ) / ( t[end] - t[end-1] )

    # backward finite difference 
    else 

        dx_fd = 0*x 
        for i = 2 : length(t)
            dx_fd[i,:] = ( x[i,:] - x[i-1,:] ) / ( t[i] - t[i-1] )
        end 

        # deal with 1st index 
        i = 1 
        dx_fd[i,:] = ( x[i+1,:] - x[i,:] ) / ( t[i+1] - t[i] )

    end 

    return dx_fd 

end 


## ============================================ ##
# check polynomial combinatorics 
# thank you https://math.stackexchange.com/questions/2928712/number-of-elements-in-polynomial-of-degree-n-and-m-variables !!! 

""" 
Check possible combinations of polynomial degree with number of variables  
""" 
function check_vars_poly_deg( 
    n,      # number of variables  
    p       # polynomial degree 
)  

    num = factorial( big(p + n - 1) ) 
    den = factorial( n - 1 ) * factorial( p )
    out = num / den 

    return out 
end

export check_vars_poly_deg 


