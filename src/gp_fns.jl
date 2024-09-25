
# Define a function to evaluate kernel performance
function evaluate_kernel(kernel, x, y)

    m = MeanZero()
    log_noise = log(0.1)
    
    # Create the GP with a try-catch block
    try
        gp = GP(x', y, m, kernel, log_noise)
        
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

function define_kernels(x, y) 

    # Estimate some data characteristics
    l = log(abs(median(diff(x, dims = 1))))     # Estimate of length scale
    σ = log(std(y))                             # Estimate of signal variance 
    p = log((maximum(x) - minimum(x)) / 2)      # Estimate of period

    # Define a list of kernels to try with more conservative initial parameters
    kernels = [
        Periodic(l, σ, p) + SE(l/10, σ/10),
        Periodic(l, σ, p) * SE(l/10, σ/10),
        Matern(1/2, l, σ) + Periodic(l, σ, p), 
        Matern(1/2, l, σ) * Periodic(l, σ, p), 
        Matern(3/2, l, σ) + Periodic(l, σ, p), 
        Matern(3/2, l, σ) * Periodic(l, σ, p), 
        RQ(l, σ, 1.0) + Periodic(l, σ, p), 
        RQ(l, σ, 1.0) * Periodic(l, σ, p), 
        SE(l, σ),  
        Matern(1/2, l, σ),  
        Matern(3/2, l, σ),  
        RQ(l, σ, 1.0)  
    ] 

    return kernels  
end 

function find_best_kernel(results)

    # Find the best kernel
    best_kernel = nothing
    best_score  = -Inf
    for result in results
        if result[3] > best_score
            best_kernel = result
            best_score  = result[3]
        end
    end

    return best_kernel
end 

function evaluate_kernels(kernels, x, y)

    results = []
    for (i, kernel) in enumerate(kernels) 
        score = evaluate_kernel(kernel, x, y)
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
    best_kernel = find_best_kernel(results) 

    if best_kernel === nothing
        error("No valid kernel found")
    end
    println("Best kernel: ", best_kernel[2], " with score ", best_kernel[3]) 

    # Use the best kernel for final GP 
    best_gp = GP(x_data', y_data, MeanZero(), best_kernel[2], log(0.1))
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

export cross_validate_csv  
function cross_validate_csv(csv_path_file)

    # extract and smooth data 
    data_train, data_test = make_data_structs(csv_path_file)
    x_train_GP, dx_train_GP, x_test_GP, _ = smooth_train_test_data(data_train, data_test)

    # cross validate sindy and gpsindy  
    df_sindy, df_gpsindy = cross_validate_sindy_gpsindy(data_train, data_test, x_train_GP, dx_train_GP)

    # save gpsindy min err stats 
    df_min_err_sindy   = df_min_err_fn(df_sindy, csv_path_file)
    df_min_err_gpsindy = df_min_err_fn(df_gpsindy, csv_path_file) 

    f_train = plot_data( data_train, x_train_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "train" )  
    f_test  = plot_data( data_test, x_test_GP, df_min_err_sindy, df_min_err_gpsindy, csv_path_file, "test" ) 

    return df_min_err_sindy, df_min_err_gpsindy, f_train, f_test  
end 