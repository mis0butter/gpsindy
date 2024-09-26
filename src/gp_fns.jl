
# Define a function to evaluate kernel performance
export evaluate_kernel  
function evaluate_kernel(kernel, x_data, y_data)
    
    # Create the GP with a try-catch block
    try

        gp = GP(x_data', y_data, MeanZero(), kernel, log(0.1))
        
        try

            optimize!(gp, 
                method = LBFGS(linesearch = LineSearches.BackTracking()), 
                iterations = 100 
            )
            return (gp.target, gp)  # Return log marginal likelihood

        catch opt_error

            # println("Optimization error: ", opt_error)
            return (-Inf, nothing)  # Return a very low score for failed optimizations

        end

    catch gp_error

        # println("GP creation error: ", gp_error)
        return (-Inf, nothing)  # Return a very low score if GP creation fails

    end
end 


## ============================================ ## 


export evaluate_kernels   
function evaluate_kernels(kernels, x_data, y_data)

    results = []
    for (i, kernel) in enumerate(kernels) 

        score, gp = evaluate_kernel(kernel, x_data, y_data)
        result    = (i = i, kernel = kernel, score = score, gp = gp) 

        push!(results, result)
        # println("Kernel $i: Log marginal likelihood = $score")

    end

    return results
end 


## ============================================ ## 


export define_kernels  
function define_kernels(x_data, y_data) 

    # Estimate some data characteristics
    l = log(abs(median(diff(x_data, dims = 1))))     # Estimate of length scale
    σ = log(std(y_data))                             # Estimate of signal variance 
    p = log((maximum(x_data) - minimum(x_data)) / 2)      # Estimate of period

    # Define a list of kernels to try with more conservative initial parameters
    kernels = [
        SE(l/10, σ/10) + Periodic(l, σ, p),
        SE(l/10, σ/10) * Periodic(l, σ, p),
        Matern(1/2, l, σ) + Periodic(l, σ, p), 
        Matern(1/2, l, σ) * Periodic(l, σ, p), 
        Matern(3/2, l, σ) + Periodic(l, σ, p), 
        Matern(3/2, l, σ) * Periodic(l, σ, p), 
        RQ(l, σ, 1.0) + Periodic(l, σ, p), 
        RQ(l, σ, 1.0) * Periodic(l, σ, p), 
        SE(l/10, σ/10),  
        Matern(1/2, l, σ),  
        Matern(3/2, l, σ),  
        RQ(l, σ, 1.0)  
    ] 

    return kernels  
end 


## ============================================ ## 


export find_best_kernel  
function find_best_kernel(results)

    # Find the best kernel
    best_result = nothing
    best_score  = -Inf
    for result in results

        if result.score > best_score
            best_result = result
            best_score  = result.score 
        end

    end

    return best_result
end 


## ============================================ ## 


export smooth_column_gp  
function smooth_column_gp(x_data, y_data, x_pred) 

    kernels     = define_kernels(x_data, y_data) 
    results     = evaluate_kernels(kernels, x_data, y_data) 
    best_result = find_best_kernel(results) 

    if best_result === nothing
        error("No valid kernel found")
    end
    println("Best kernel: ", best_result.kernel, " with score ", best_result.score) 
 
    μ_best, σ²_best = predict_y(best_result.gp, x_pred')

    return μ_best, σ²_best, best_result.gp 
end 


## ============================================ ## 


export smooth_array_gp  
function smooth_array_gp(x_data, y_data, x_pred) 

    n_vars   = size(y_data, 2) 
    μ_best   = zeros(size(x_pred, 1), n_vars)
    σ²_best  = zeros(size(x_pred, 1), n_vars)
    best_gps = [] 

    for i in 1:n_vars
        μ_best[:, i], σ²_best[:, i], best_gp = smooth_column_gp(x_data, y_data[:, i], x_pred) 
        push!(best_gps, best_gp) 
    end 

    return μ_best, σ²_best, best_gps 
end 


## ============================================ ## 


export smooth_train_test_data 
function smooth_train_test_data( data_train, data_test ) 

    # first - smooth measurements with Gaussian processes 
    x_train_GP, _, _  = smooth_array_gp(data_train.t, data_train.x_noise, data_train.t)

    println("using u to smooth dx3 and dx4") 

    # dx_train_GP, _, _ = smooth_array_gp(x_train_GP, data_train.dx_noise, x_train_GP) 
    dx_train_GP = similar(data_train.dx_noise) 

    x_data = x_train_GP 
    y_data = data_train.dx_noise[:,1:2]
    x_pred = x_train_GP  
    dx_train_GP[:,1:2], _, _ = smooth_array_gp(x_data, y_data, x_pred) 
    x_data = data_train.u[:,1] 
    y_data = data_train.dx_noise[:,3]
    x_pred = data_train.u[:,1] 
    dx_train_GP[:,3], _, _ = smooth_column_gp(x_data, y_data, x_pred) 
    x_data = data_train.u[:,2] 
    y_data = data_train.dx_noise[:,4]
    x_pred = data_train.u[:,2] 
    dx_train_GP[:,4], _, _ = smooth_column_gp(x_data, y_data, x_pred)  

    x_test_GP, _, _   = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)
    dx_test_GP, _, _  = smooth_array_gp(x_test_GP, data_test.dx_noise, x_test_GP)

    return x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 


## ============================================ ## 


export interpolate_train_test_data 
function interpolate_train_test_data( data_train, data_test, interp_factor = 2 ) 

    # interp_factor  = Int( interpolate_gp )  
    t_train_interp = interpolate_array( data_train.t, interp_factor ) 
    u_train_interp = interpolate_array( data_train.u, interp_factor ) 

    x_train_GP, _, _  = smooth_array_gp(data_train.t, data_train.x_noise, t_train_interp)
    dx_train_GP, _, _ = smooth_array_gp(data_train.x_noise, data_train.dx_noise, x_train_GP)
    x_test_GP, _, _   = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)
    dx_test_GP, _, _  = smooth_array_gp(x_test_GP, data_test.dx_noise, x_test_GP)

    return t_train_interp, u_train_interp, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 


## ============================================ ## 


export cross_validate_sindy 
function cross_validate_sindy(csv_path_file, interp_factor = 1)

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_sindy   = DataFrame( fill( [], 5 ), header ) 

    for i_λ = eachindex(λ_vec) 

        λ = λ_vec[i_λ] 
        
        # sindy!!! 
        x_sindy_train, x_sindy_test = integrate_sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

        push!(df_sindy, [
            λ,                                          # lambda  
            norm(data_train.x_noise - x_sindy_train),   # train error  
            norm(data_test.x_noise - x_sindy_test),     # test error  
            x_sindy_train,                              # train trajectory  
            x_sindy_test                                # test trajectory  
        ])

    end 

    # save gpsindy min err stats 
    df_best_sindy = df_min_err_fn(df_sindy, csv_path_file)

    fig = plot_data( data_train, 0*data_train.x_noise, data_test, 0*data_test.x_noise, df_best_sindy, df_best_sindy, interp_factor, csv_path_file )    

    return df_sindy 
end 


## ============================================ ## 


export cross_validate_sindy_gpsindy  
function cross_validate_sindy_gpsindy(data_train, data_test, x_train_GP, dx_train_GP)

    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_gpsindy = DataFrame( fill( [], 5 ), header ) 
    df_sindy   = DataFrame( fill( [], 5 ), header ) 

    for i_λ = eachindex(λ_vec) 

        λ = λ_vec[i_λ] 

        # gpsindy!!! 
        x_gpsindy_train, x_gpsindy_test = integrate_sindy_lasso( x_train_GP, dx_train_GP, λ, data_train, data_test ) 
        
        push!(df_gpsindy, [
            λ,                                           # lambda
            norm(data_train.x_noise - x_gpsindy_train),  # train error
            norm(data_test.x_noise - x_gpsindy_test),    # test error
            x_gpsindy_train,                             # train trajectory
            x_gpsindy_test                               # test trajectory
        ])
        
        # sindy!!! 
        x_sindy_train, x_sindy_test = integrate_sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

        push!(df_sindy, [
            λ,                                          # lambda  
            norm(data_train.x_noise - x_sindy_train),   # train error  
            norm(data_test.x_noise - x_sindy_test),     # test error  
            x_sindy_train,                              # train trajectory  
            x_sindy_test                                # test trajectory  
        ])

    end 

    return df_sindy, df_gpsindy 
end 


## ============================================ ## 


export cross_validate_sindy_gpsindy_interp 
function cross_validate_sindy_gpsindy_interp(data_train, data_test, x_train_GP, dx_train_GP, t_train_interp, u_train_interp) 

    λ_vec      = λ_vec_fn() 
    header     = [ "λ", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_gpsindy = DataFrame( fill( [], 5 ), header ) 
    df_sindy   = DataFrame( fill( [], 5 ), header ) 

    for i_λ = eachindex(λ_vec) 

        λ = λ_vec[i_λ] 
        
        # sindy!!! 
        x_sindy_train, x_sindy_test = integrate_sindy_lasso( data_train.x_noise, data_train.dx_noise, λ, data_train, data_test ) 

        push!(df_sindy, [
            λ,                                          # lambda  
            norm(data_train.x_noise - x_sindy_train),   # train error  
            norm(data_test.x_noise - x_sindy_test),     # test error  
            x_sindy_train,                              # train trajectory  
            x_sindy_test                                # test trajectory  
        ])

        # gpsindy!!! 
        x_gpsindy_train, x_gpsindy_test = integrate_gpsindy_interp( x_train_GP, dx_train_GP, t_train_interp, u_train_interp, λ, data_train, data_test )  
        
        push!(df_gpsindy, [
            λ,                                           # lambda
            norm(data_train.x_noise - x_gpsindy_train),  # train error
            norm(data_test.x_noise - x_gpsindy_test),    # test error
            x_gpsindy_train,                             # train trajectory
            x_gpsindy_test                               # test trajectory
        ])

    end 

    return df_sindy, df_gpsindy
end


## ============================================ ## 


export process_data_and_cross_validate  
function process_data_and_cross_validate(data_train, data_test, interp_factor)

    if interp_factor == 1 

        x_train_GP, dx_train_GP, x_test_GP, _ = smooth_train_test_data(data_train, data_test)

        # cross validate sindy and gpsindy  
        df_sindy, df_gpsindy = cross_validate_sindy_gpsindy(data_train, data_test, x_train_GP, dx_train_GP) 

    else 

        t_train_interp, u_train_interp, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP = interpolate_train_test_data(data_train, data_test, interp_factor) 

        # cross validate sindy and gpsindy   
        df_sindy, df_gpsindy = cross_validate_sindy_gpsindy_interp(data_train, data_test, x_train_GP, dx_train_GP, t_train_interp, u_train_interp) 

        _, x_train_GP  = downsample_to_original(data_train.t, t_train_interp, x_train_GP) 
        _, dx_train_GP = downsample_to_original(data_train.t, t_train_interp, dx_train_GP) 

    end 

    return df_sindy, df_gpsindy, x_train_GP, dx_train_GP, x_test_GP

end


## ============================================ ## 


export cross_validate_csv  
function cross_validate_csv(csv_path_file, interp_factor = 1)

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)

    df_sindy, df_gpsindy, x_train_GP, _, x_test_GP = process_data_and_cross_validate(data_train, data_test, interp_factor)

    # save gpsindy min err stats 
    df_min_err_sindy   = df_min_err_fn(df_sindy, csv_path_file)
    df_min_err_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 

    fig = plot_data( data_train, x_train_GP, data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, interp_factor, csv_path_file )    

    return df_min_err_sindy, df_min_err_gpsindy, fig 
end 


## ============================================ ## 


export run_csv_files  
function run_csv_files(csv_files_vec, save_path_fig, plot_option = false)
        
    # create dataframe to store results 
    header = [ "csv_file", "λ_min", "train_err", "test_err", "train_traj", "test_traj" ] 
    df_best_csvs_sindy   = DataFrame( fill( [], 6 ), header ) 
    df_best_csvs_gpsindy = DataFrame( fill( [], 6 ), header ) 
    
    # loop 
    for i_csv in eachindex( csv_files_vec ) 
    
        csv_path_file = csv_files_vec[i_csv] 
        df_min_err_sindy, df_min_err_gpsindy, fig = cross_validate_csv( csv_path_file, 1 ) 

        if plot_option == true 
            display(fig)
        end 
    
        # save figs 
        csv_file = replace( split( csv_path_file, "/" )[end], ".csv" => "" )  
        save( string( save_path_fig, csv_file, ".png" ), fig )  
    
        push!( df_best_csvs_sindy, df_min_err_sindy[1,:] ) 
        push!( df_best_csvs_gpsindy, df_min_err_gpsindy[1,:] ) 
    
    end 

    return df_best_csvs_sindy, df_best_csvs_gpsindy
end


## ============================================ ## 


export run_save_csv_files   
function run_save_csv_files( freq_hz, noise, plot_option = false )  

    csv_path = string("test/data/jake_car_csvs_ctrlshift_no_trans/", freq_hz, "hz_noise_", noise, "/" )

    csv_files_vec, save_path, save_path_fig, save_path_dfs = mkdir_save_path( csv_path ) 

    df_best_csvs_sindy, df_best_csvs_gpsindy = run_csv_files( csv_files_vec, save_path_fig, plot_option ) 
    
    # save df_min_err for gpsindy and sindy 
    CSV.write( string( save_path_dfs, "df_min_err_csvs_sindy.csv" ), df_best_csvs_sindy ) 
    CSV.write( string( save_path_dfs, "df_min_err_csvs_gpsindy.csv" ), df_best_csvs_gpsindy ) 

    df_mean_err = save_dfs_mean( df_best_csvs_sindy, df_best_csvs_gpsindy, freq_hz, noise ) 
    CSV.write( string( save_path, "df_mean_err.csv" ), df_mean_err ) 

    return df_best_csvs_sindy, df_best_csvs_gpsindy, df_mean_err 
end 


