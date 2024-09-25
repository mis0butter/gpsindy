
# Define a function to evaluate kernel performance
export evaluate_kernel  
function evaluate_kernel(kernel, x_data, y_data)

    m = MeanZero()
    log_noise = log(0.1)
    
    # Create the GP with a try-catch block
    try
        gp = GP(x_data', y_data, m, kernel, log_noise)
        
        # Optimize with bounds and error handling
        try
            optimize!(gp, 
                method = LBFGS(linesearch = LineSearches.BackTracking()), 
                # iterations = 100 
            )
            return gp.target  # Return log marginal likelihood
        catch opt_error
            println("Optimization error: ", opt_error)
            return -Inf  # Return a very low score for failed optimizations
        end
    catch gp_error
        println("GP creation error: ", gp_error)
        return -Inf  # Return a very low score if GP creation fails

    end
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
        score = result[3] 
        if score > best_score
            best_result = result
            best_score  = score 
        end
    end

    return best_result
end 


## ============================================ ## 

export evaluate_kernels   
function evaluate_kernels(kernels, x_data, y_data)

    results = []
    for (i, kernel) in enumerate(kernels) 
        score = evaluate_kernel(kernel, x_data, y_data)
        push!(results, (i, kernel, score))
        println("Kernel $i: Log marginal likelihood = $score")
    end

    return results
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
    best_kernel = best_result[2]  
    best_score  = best_result[3]  
    println("Best kernel: ", best_kernel, " with score ", best_score) 

    # Use the best kernel for final GP 
    best_gp = GP(x_data', y_data, MeanZero(), best_kernel, log(0.1))
    optimize!(best_gp, 
        method = LBFGS(linesearch = LineSearches.BackTracking()), 
        # iterations = 100 
    )

    # Make predictions with the best kernel 
    # y_post[:,i] = predict_y( gp, x_pred' )[1]  
    μ_best, σ²_best = predict_y(best_gp, x_pred')

    return μ_best, σ²_best, best_gp 
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
    dx_train_GP, _, _ = smooth_array_gp(x_train_GP, data_train.dx_noise, x_train_GP)
    x_test_GP, _, _   = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)
    dx_test_GP, _, _  = smooth_array_gp(x_test_GP, data_test.dx_noise, x_test_GP)

    return x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
end 


## ============================================ ## 
# smooth training and test data with GPs 

# function interpolate_and_smooth_data(data_train, data_test, interp_factor = 2)

#     t_train_interp = interpolate_array(data_train.t, interp_factor)
#     u_train_interp = interpolate_array(data_train.u, interp_factor)

#     println("interpolating x_train_GP...")
#     x_train_GP, _, _ = smooth_array_gp(data_train.t, data_train.x_noise, t_train_interp)

#     println("interpolating dx_train_GP...")
#     dx_train_GP, _, _ = smooth_array_gp(data_train.x_noise, data_train.dx_noise, x_train_GP)

#     println("interpolating x_test_GP...")
#     x_test_GP, _, _ = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)

#     println("interpolating dx_test_GP...")
#     dx_test_GP, _, _ = smooth_array_gp(data_test.x_noise, data_test.dx_noise, x_test_GP)

#     return t_train_interp, u_train_interp, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP
# end

# function smooth_gp_posterior( x_pred, 0, x_data, 0, y_data ) 
export interpolate_train_test_data 
function interpolate_train_test_data( data_train, data_test, interp_factor = 2 ) 

    # interp_factor  = Int( interpolate_gp )  
    t_train_interp = interpolate_array( data_train.t, interp_factor ) 
    u_train_interp = interpolate_array( data_train.u, interp_factor ) 

    println("interpolating x_train_GP...")
    # x_train_GP  = smooth_gp_posterior( x_pred, 0, x_data, 0, y_data, σn, opt_σn )
    x_pred = t_train_interp  
    x_data = data_train.t   
    y_data = data_train.x_noise  
    # x_train_GP  = smooth_gp_posterior( t_train_interp, 0, data_train.t, 0, data_train.x_noise, σn, opt_σn ) 
    x_train_GP, _, _  = smooth_array_gp(data_train.t, data_train.x_noise, t_train_interp)

    println("interpolating dx_train_GP...") 
    x_pred = x_train_GP  
    x_data = data_train.x_noise  
    y_data = data_train.dx_noise 
    # dx_train_GP = smooth_gp_posterior( x_train_GP, 0, data_train.x_noise, 0, data_train.dx_noise, σn, opt_σn ) 
    dx_train_GP, _, _ = smooth_array_gp(data_train.x_noise, data_train.dx_noise, x_train_GP)

    println("interpolating x_test_GP...")  
    x_pred = data_test.t  
    x_data = data_test.t   
    y_data = data_test.x_noise 
    # x_test_GP   = smooth_gp_posterior( data_test.t, 0, data_test.t, 0, σn, opt_σn ) 
    x_test_GP, _, _ = smooth_array_gp(data_test.t, data_test.x_noise, data_test.t)

    println("interpolating dx_test_GP...") 
    x_pred = x_test_GP   
    x_data = data_test.x_noise  
    y_data = data_test.dx_noise  
    # dx_test_GP  = smooth_gp_posterior( x_test_GP, 0, x_test_GP, 0, σn, opt_σn ) 
    dx_test_GP, _, _ = smooth_array_gp(x_test_GP, data_test.dx_noise, x_test_GP)

    return t_train_interp, u_train_interp, x_train_GP, dx_train_GP, x_test_GP, dx_test_GP 
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
        x_gpsindy_train, x_gpsindy_test = integrate_sindy_lasso( x_train_GP, dx_train_GP, λ, data_train, data_test ) 
        
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

    f_train = plot_data( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "train" )  
    f_test  = plot_data( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "test" ) 

    return df_min_err_sindy, df_min_err_gpsindy, f_train, f_test  
end 