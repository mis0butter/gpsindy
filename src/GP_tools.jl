
# Define a function to evaluate kernel performance
function evaluate_kernel(kernel, x, y)

    m = MeanZero()
    logNoise = log(0.1)
    
    # Create the GP with a try-catch block
    try
        gp = GP(x', y, m, kernel, logNoise)
        
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
    l = log(abs(median(diff(x, dims = 1))))  # Estimate of length scale
    σ = log(std(y))  # Estimate of signal variance 
    p = log((maximum(x) - minimum(x)) / 2)  # Estimate of period

    # Define a list of kernels to try with more conservative initial parameters
    kernels = [
        Periodic(l, σ, p) + SE(l/10, σ/10),
        Periodic(l, σ, p) * SE(l/10, σ/10),
        SE(l/10, σ/10) + Periodic(l, σ/2, p),
        Matern(1/2, l, σ) + Periodic(l, σ, p), 
        SE(l, σ),  
        Matern(1/2, l, σ),  
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
# posterior GP and optimize hps 

export smooth_gp_posterior 
function smooth_gp_posterior( x_prior, μ_prior, x_train, μ_train, y_train, σ_n = 1e-1, σ_n_opt = true ) 

    # set up posterior 
    x_rows = size( x_prior, 1 ) ; n_vars = size(y_train, 2) 
    y_post = zeros( x_rows, n_vars ) 
    
    # optimize hyperparameters, compute posterior y_post for each state 
    for i = 1 : n_vars 
    
        # kernel  
        mZero     = MeanZero()              # zero mean function 
        kern      = SE( 0.0, 0.0 )          # squared eponential kernel (hyperparams on log scale) 
        
        # log_noise = log( σ_n )              # (optional) log std dev of obs 
        log_noise = log( σ_n )              # (optional) log std dev of obs 
        
        # y_train = dx_noise[:,i] - dx_mean[:,i]
        gp      = GP( x_train', y_train[:,i] - μ_train[:,i], mZero, kern, log_noise ) 
        optimize!( gp, method = LBFGS( linesearch = LineSearches.BackTracking() ), noise = σ_n_opt ) 

        # # report hyperparameter 
        # σ_n = exp( gp.logNoise.value )  
        # println( "opt σ_n = ", σ_n ) 

        y_post[:,i] = predict_y( gp, x_prior' )[1]  
    
    end 

    return y_post 

end 

## ============================================ ## 

# export smooth_gp_posterior 
# function smooth_gp_posterior(x_prior, μ_prior, x_train, μ_train, y_train, σ_n = 1e-1, σ_n_opt = true) 
    
#     # set up posterior 
#     x_rows = size( x_prior, 1 ) ; n_vars = size(y_train, 2) 
#     y_post = zeros( x_rows, n_vars ) 
    
#     # optimize hyperparameters, compute posterior y_post for each state 
#     for i = 1 : n_vars 
    
#         # kernel  
#         mZero = MeanZero()              # zero mean function 
        
#         # Periodic kernel with initial hyperparameters
#         l = median(diff(x_train, dims = 1))  # initial length scale: median distance between points
#         p = (maximum(x_train) - minimum(x_train)) / 2  # initial period: half the data range
#         σ = std(y_train[:,i])  # initial signal variance: standard deviation of the data

#         # kern = Periodic( l, p, σ )
#         nugget = 1e-5  # small value to ensure positive definiteness 
#         kern = Periodic(l, p, σ) + SE(nugget, nugget)
        
#         log_noise = log(max(σ_n, 1e-6))  # Ensure non-zero noise for stability
        
#         gp = GP(x_train', y_train[:,i] - μ_train[:,i], mZero, kern, log_noise) 
        
#         # Optimization with bounds and multiple restarts
#         best_nlml = Inf
#         best_gp = gp
        
#         for _ in 1:5  # Try 5 different initial conditions
#             try
#                 optimize!(gp, method = LBFGS(linesearch = LineSearches.BackTracking()), 
#                           noise = σ_n_opt,
#                           domean = false,  # Don't optimize mean parameters
#                           kern = true,     # Do optimize kernel parameters
#                           iterations = 100,
#                           lower = [-10.0, -10.0, -10.0, -10.0],  # Lower bounds for log(l), log(p), log(σ), log(σ_n)
#                           upper = [10.0, 10.0, 10.0, 0.0])      # Upper bounds

#                 if gp.logNoise.value < best_nlml
#                     best_nlml = gp.logNoise.value
#                     best_gp = gp
#                 end
#             catch e
#                 println("Optimization failed, trying again with different initial conditions")
#                 println("Error: ", e)
#             end
#         end
        
#         gp = best_gp  # Use the best GP found

#         # Add small jitter to improve numerical stability
#         jitter = 1e-6
#         # y_post[:,i], _ = predict_y(gp, x_prior', jitter)
#         y_post[:,i] = predict_y( gp, x_prior' )[1]
    
#     end 

#     return y_post 
# end 

## ============================================ ##
# sample from given mean and covariance 

export gauss_sample 
function gauss_sample(μ, K) 
# function gauss_sample(μ::Vector, K::Matrix) 

    # adding rounding ... 
    K = round.( K, digits = 10 )
    
    # cholesky decomposition, get lower triangular decomp 
    C = cholesky(K) ; 
    L = C.L 

    # draw random samples 
    u = randn(length(μ)) 

    # f ~ N(mu, K(x, x)) 
    f = μ + L*u

    return f 

end 


## ============================================ ##
# sample from given mean and covariance 

export k_SE 
function k_SE( σ_f, l, xp, xq )

    K = σ_f^2 * exp.( -1/( 2*l^2 ) * sq_dist(xp, xq) )     

    # deal with det(Ky) = 0 
    # if det(K) == 0 
    #     K *= length(xp)
    # end  

    return K 

end 


## ============================================ ##
# sample from given mean and covariance 

export k_periodic
function k_periodic( σ_f, l, p, xp, xq )

    K = σ_f^2 * exp.( -2/( l^2 ) * sin.( π/p * sq_dist(xp, xq)) )     

    # deal with det(Ky) = 0 
    # if det(K) == 0 
    #     K *= length(xp)
    # end  

    return K 

end 


## ============================================ ##
# define square distance function 

export sq_dist 
function sq_dist(a, b) 
# function sq_dist(a::Vector, b::Vector) 

    r = length(a) 
    p = length(b) 

    # iterate 
    C = zeros(r,p) 
    for i = 1:r 
        for j = 1:p 
            C[i,j] = norm( a[i] - b[j] )^2 
        end 
    end 

    return C 

end 


## ============================================ ##
# marginal log-likelihood for Gaussian Processes 

export log_p 
function log_p( σ_f, l, σ_n, x, y, μ )
    
    # training kernel function 
    Ky = k_SE(σ_f, l, x, x) 
    Ky += σ_n^2 * I 
    
    # while det(Ky) == 0 
    #     println( "det(Ky) = 0" )
    #     Ky += σ_n * I 
    # end 

    # term  = 1/2 * ( y - μ )' * inv( Ky ) * ( y - μ ) 
    term  = 1/2 * ( y - μ )' * ( Ky \ ( y - μ ) ) 
    term += 1/2 * log( det( Ky ) ) 

    return term 

end 


## ============================================ ##
# posterior distribution 

export post_dist
function post_dist( x_train, y_train, x_test, σ_f, l, σ_n )

    # x  = training data  
    # xs = test data 
    # joint distribution 
    #   [ y  ]     (    [ K(x,x) + σ_n^2*I  K(x,xs)  ] ) 
    #   [ fs ] ~ N ( 0, [ K(xs,x)           K(xs,xs) ] ) 

    # covariance from training data 
    K    = k_SE(σ_f, l, x_train, x_train)  
    Ks   = k_SE(σ_f, l, x_train, x_test)  
    Kss  = k_SE(σ_f, l, x_test, x_test) 

    # conditional distribution 
    # mu_cond    = K(Xs,X)*inv(K(X,X))*y
    # sigma_cond = K(Xs,Xs) - K(Xs,X)*inv(K(X,X))*K(X,Xs) 

    # fs | (Xs, X, y) ~ N ( mu_cond, sigma_cond ) 
    # μ_post = Ks' * K^-1 * y_train 
    # Σ_post = Kss - (Ks' * K^-1 * Ks)  

    C = cholesky(K + σ_n^2 * I) 
    α = C.U \ ( C.L \ y_train ) 
    v = C.L \ Ks 
    μ_post = Ks' * α 
    Σ_post = Kss - v'*v 

    return μ_post, Σ_post

end 


## ============================================ ##
# hp optimization (June) --> post mean  

export post_dist_hp_opt 
function post_dist_hp_opt( x_train, y_train, y_mean, x_test, plot_option = false )

    # IC 
    hp = [ 1.0, 1.0, 0.1 ] 

    # optimization 
    hp_opt(( σ_f, l, σ_n )) = log_p( σ_f, l, σ_n, x_train, y_train, y_mean )
    od       = OnceDifferentiable( hp_opt, hp ; autodiff = :forward ) 
    result   = optimize( od, hp, LBFGS() ) 
    hp       = result.minimizer 

    μ_post, Σ_post = post_dist( x_train, y_train, x_test, hp[1], hp[2], hp[3] ) 

    if plot_option 
        p = scatter( x_train, y_train, label = "train" )
        scatter!( p, x_test, μ_post, label = "post", ls = :dash )
        display(p) 
    end 

    return μ_post, Σ_post, hp 
end 

## ============================================ ##
# posterior mean with GP toolbox 

export post_dist_SE 
function post_dist_SE( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = SE( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 
    
    n_vars   = size(y_train, 2) 
    y_smooth = zeros( size(x_test, 1), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP( x_train', y_train[:,i], mZero, kern, log_noise ) 

        # optimize!( gp, method = LBFGS(linesearch=LineSearches.BackTracking()) ) 
        optimize!( gp, method = BFGS(linesearch=LineSearches.BackTracking()) ) 

        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        μ, σ²   = predict_y( gp, x_test' )  

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    if n_vars == 1 
        hps = hps[1] 
    end 

    return y_smooth, Σ, hps      
end 

export post_dist_M12A
function post_dist_M12A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M32A
function post_dist_M32A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 
        
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M52A
function post_dist_M52A( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Ard( [0.0], 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt.( gp.kernel.iℓ2[1] ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps  
end 


export post_dist_M12I
function post_dist_M12I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat12Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 
    
    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 

        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_M32I
function post_dist_M32I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat32Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n]   
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_M52I
function post_dist_M52I( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Mat52Iso( 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = gp.kernel.ℓ ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n]   
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 


export post_dist_per
function post_dist_per( x_train, y_train, x_test ) 

    # kernel  
    mZero     = MeanZero() ;            # zero mean function 
    kern      = Periodic( 0.0, 0.0, 0.0 ) ;        # squared eponential kernel (hyperparams on log scale) 
    log_noise = log(0.1) ;              # (optional) log std dev of obs noise 

    n_vars   = size(y_train, 2) 
    y_smooth = zeros( length(x_test), n_vars ) 
    Σ        = 0 * y_smooth 
    hps      = [] 
    for i = 1:n_vars 

        # fit GP 
        gp      = GP(x_train, y_train[:,i], mZero, kern, log_noise) 
        optimize!(gp) 
        μ, σ²   = predict_y( gp, x_test )  
        
        # return HPs 
        σ_f = sqrt( gp.kernel.σ2 ) ; l = sqrt( gp.kernel.ℓ2 ) ; σ_n = exp( gp.logNoise.value )  
        hp  = [σ_f, l, σ_n] 
    
        y_smooth[:,i] = μ 
        Σ[:,i]        = σ²
        push!( hps, hp ) 
    
    end 

    return y_smooth, Σ, hps 
end 
