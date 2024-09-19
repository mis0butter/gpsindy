# this is a script for tuning lambda for each dx based on the error from the TRAINING DATA 

using GaussianSINDy 
using LinearAlgebra 
using Plots 

## ============================================ ##

fn = unicycle 

λ = 0.1 
data_train, data_test = ode_train_test( fn )

x_train_noise  = data_train.x_noise 
dx_train_noise = data_train.dx_noise 
u_train        = data_train.u 
t_train        = data_train.t 
x_train_true   = data_train.x_true 
dx_train_true  = data_train.dx_true 

x_test_noise   = data_test.x_noise 
dx_test_noise  = data_test.dx_noise 
u_test         = data_test.u
t_test         = data_test.t    
x_test_true    = data_test.x_true 
dx_test_true   = data_test.dx_true 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_vars( x_train_noise, u_train ) 
poly_order = 4 

Ξ_true       = sindy_stls( x_train_true, dx_train_true, λ, poly_order, u_train ) 
Ξ_true_terms = pretty_coeffs( Ξ_true, x_train_true, u_train ) 


## ============================================ ##
# smooth with GPs 

# first - smooth measurements with Gaussian processes 
x_train_GP  = smooth_gp_posterior( t_train, 0*x_train_noise, t_train, 0*x_train_noise, x_train_noise ) 
dx_train_GP = smooth_gp_posterior( x_train_GP, 0*dx_train_noise, x_train_GP, 0*dx_train_noise, dx_train_noise ) 
x_test_GP   = smooth_gp_posterior( t_test, 0*x_test_noise, t_test, 0*x_test_noise, x_test_noise ) 
dx_test_GP  = smooth_gp_posterior( x_test_GP, 0*dx_test_noise, x_test_GP, 0*dx_test_noise, dx_test_noise ) 

# build function library from smoothed data 
Θx_gp  = pool_data_test( [ x_train_GP u_train ], n_vars, poly_order ) 

# get x0 from smoothed data 
x0_train_GP = x_train_GP[1,:] 
x0_test_GP  = x_test_GP[1,:] 

# SINDy (STLS) 
λ = 0.1 
Θx_noise      = pool_data_test( [ x_train_noise u_train ], n_vars, poly_order ) 
Ξ_sindy_stls  = sindy_stls( x_train_noise, dx_train_noise, λ, poly_order, u_train ) 
dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
dx_sindy      = Θx_noise * Ξ_sindy_stls  
x_train_sindy = integrate_euler( dx_fn_sindy, x0_train_GP, t_train, u_train ) 

# integrate true function 
dx_fn_true    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
# x_train_true  = integrate_euler( dx_fn_true, x0_train_GP, t_train, u_train ) 

# # integrate unicycle (with GP) 
# x_unicycle_train = integrate_euler_unicycle( fn, x0_train_GP, t_train, u_train ) 
# x_unicycle_test  = integrate_euler_unicycle( fn, x0_test_GP, t_test, u_test ) 
# for i = 1 : x_vars 
#     println( "dx_fn_true norm err = ", norm( x_train_true[:,i] - x_train_true[:,i] )  ) 
#     println( "unicycle norm err = ", norm( x_unicycle_train[:,i] - x_train_true[:,i] )  ) 
# end 

## ============================================ ##

λ_vec = [ 1e-6 ] 
while λ_vec[end] < 10 
    push!( λ_vec, 10 * λ_vec[end] ) 
end 
while λ_vec[end] < 500 
    push!( λ_vec, 10 + λ_vec[end] ) 
end 


err_Ξ_vec  = [] 
err_x_vec  = [] 
err_dx_vec = [] 
Ξ_gpsindy_vec = [] 
for j = 1 : x_vars 

    a_dx = Animation() 
    a_x = Animation() 

    # DX PLOT 
    ymin, dy, ymax = min_d_max( dx_train_true[:,j] ) 
    plt_dx = plot( t_train, dx_train_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "dx", 
        ylim   = ( ymin, ymax ), 
        title = string("dx", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) ; frame(a_dx, plt_dx) 
    plot!( plt_dx, t_train, dx_train_noise[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) ; frame(a_dx, plt_dx) 
        plot!( plt_dx, t_train, dx_train_GP[:,j], 
        label = "GP", 
        ls    = :dashdot,   

    ) ; frame(a_dx, plt_dx) 
    plot!( plt_dx, t_train, dx_sindy[:,j], 
        label = "SINDy",
        ls    = :dashdotdot,  
    ) ; frame(a_dx, plt_dx) 

    # X PLOT 
    ymin, dy, ymax = min_d_max( x_train_true[:,j] ) 
    plt_x = plot( t_train, x_train_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "x", 
        ylim   = ( ymin, ymax ), 
        title = string("x", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) 
    plot!( plt_x, t_train, x_train_noise[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) ; frame(a_x, plt_x) 
    plot!( plt_x, t_train, x_train_GP[:,j], 
        label = "GP", 
        ls    = :dashdot,   
    ) ; frame(a_x, plt_x) 
    # plot!( plt_x, t_train, x_unicycle_train[:,j], 
    #     label = "unicycle", 
    #     ls    = :dashdotdot,   
    # ) ; frame(a_x, plt_x) 
    plt_x_sindy = deepcopy(plt_x) 
    plot!( plt_x_sindy, t_train, x_train_sindy[:,j], 
        label = "SINDy",
        ls    = :dot,  
    ) ; frame(a_x, plt_x_sindy) 
    
    err_ξ_vec     = [] 
    err_xj_vec    = [] 
    err_dxj_vec   = [] 
    ξ_gpsindy_vec = []

    # CROSS-VALIDATION 
    for i = 1 : length(λ_vec) 
        # j = 1 
    
        λ = λ_vec[i] 
        println( "i = ", i, ". λ = ", λ ) 

        Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, poly_order, u_train ) 
        dx_gpsindy = Θx_gp * Ξ_gpsindy 

        if sum(Ξ_gpsindy[:,j]) == 0 
            break 
        end 

        Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_train_GP, u_train ) 
        display(Ξ_gpsindy_terms) 

        dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
        x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train_GP, t_train, u_train ) 
        err_ξ_norm      = norm( Ξ_true[:,j] - Ξ_gpsindy[:,j] ) 
        err_xj_norm     = norm( x_gpsindy_train[:,j] - x_train_GP[:,j] ) 
        err_dxj_norm    = norm( dx_gpsindy[:,j] - dx_train_GP[:,j] ) 

        # plot_validation_test( t_train, x_train, x_unicycle_train, x_sindy_train, x_gpsindy_train ) 

        plt_gpsindy_dx = deepcopy(plt_dx) 
        plot!( plt_gpsindy_dx, t_train, dx_gpsindy[:,j], 
            label = string("GPSINDy λ = ", @sprintf "%.1e" λ),
            ls    = :dot,  
        ) ; frame(a_dx, plt_gpsindy_dx) 
        display( plt_gpsindy_dx )

        plt_gpsindy_x = deepcopy( plt_x )
        plot!( plt_gpsindy_x, t_train, x_gpsindy_train[:,j], 
        label = string("GPSINDy λ = ", @sprintf "%.1e" λ),
        ls    = :dot,   
        ) ; frame(a_x, plt_gpsindy_x) 
        display( plt_gpsindy_x )

        # save ξ coefficients 
        ξ = Ξ_gpsindy[:,j] 
        push!( ξ_gpsindy_vec, ξ ) 

        # push error stuff 
        push!( err_ξ_vec, err_ξ_norm ) 
        push!( err_xj_vec, err_xj_norm ) 
        push!( err_dxj_vec, err_dxj_norm ) 

    end 

    g_dx = gif(a_dx, fps = 1.0) 
    display(g_dx) 
    g_x = gif(a_x, fps = 1.0) 
    display(g_x) 
    # display(plt) 

    push!( err_Ξ_vec, err_ξ_vec ) 
    push!( err_x_vec, err_xj_vec ) 
    push!( err_dx_vec, err_dxj_vec ) 
    push!( Ξ_gpsindy_vec, ξ_gpsindy_vec ) 

end 


## ============================================ ##
# save ξ with smallest x error  

Ξ_gpsindy_minerr = 0 * Ξ_sindy_stls
for i = 1 : x_vars 

    err_xi_vec = err_x_vec[i]
    ξi_vec     = Ξ_gpsindy_vec[i] 

    for i = eachindex(err_xi_vec) 
        if isnan(err_xi_vec[i])  
            err_xi_vec[i] = 1e10 
        end 
    end 

    # find index with smallest element 
    min_idx = argmin( err_xi_vec ) 
    ξi      = ξi_vec[min_idx] 
    println( "min_idx = ", min_idx ) 

    Ξ_gpsindy_minerr[:,i] = ξi 

end 

## ============================================ ##

# integrate with ξ with smallest x error for training data 
dx_fn_gpsindy_minerr = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy_minerr ) 
x_gpsindy_train      = integrate_euler( dx_fn_gpsindy_minerr, x0_train_GP, t_train, u_train ) 

plt_x_vec = [] 
for j = 1 : x_vars 
    ymin, dy, ymax = min_d_max( x_train_true[:,j] ) 
    plt_x = plot( t_train, x_train_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "x", 
        ylim   = ( ymin, ymax ), 
        title = string("x", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) 
    plot!( plt_x, t_train, x_train_noise[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) 
    plot!( plt_x, t_train, x_train_GP[:,j], 
        label = "GP", 
        ls    = :dashdot,   
    ) 
    # plot!( plt_x, t_train, x_unicycle_train[:,j], 
    #     label = "unicycle", 
    #     ls    = :dashdotdot,   
    # ) 
    plot!( plt_x, t_train, x_gpsindy_train[:,j], 
    label = string("GPSINDy"),
    ls    = :dot,   
    ) 
    push!( plt_x_vec, plt_x ) 
end 
pfig = plot( plt_x_vec ... , 
    layout = ( x_vars, 1 ), 
    size = (1000, 400*x_vars), 
    plot_title = "training data", 
)

display(pfig) 

## ============================================ ##
# TESTING 

# integrate with ξ with smallest x error for test data 
x_gpsindy_test = integrate_euler( dx_fn_gpsindy_minerr, x0_test_GP, t_test, u_test )

plt_x_vec = [] 
for j = 1 : x_vars 
    ymin, dy, ymax = min_d_max( x_test_true[:,j] ) 
    plt_x = plot( t_test, x_test_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "x", 
        ylim   = ( ymin, ymax ), 
        title = string("x", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) 
    plot!( plt_x, t_test, x_test_noise[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) 
    plot!( plt_x, t_test, x_test_GP[:,j], 
        label = "GP", 
        ls    = :dashdot,   
    ) 
    # plot!( plt_x, t_test, x_unicycle_test[:,j], 
    #     label = "unicycle", 
    #     ls    = :dashdotdot,   
    # ) 
    plot!( plt_x, t_test, x_gpsindy_test[:,j], 
    label = string("GPSINDy"),
    ls    = :dot,   
    ) 
    push!( plt_x_vec, plt_x ) 
end 
pfig = plot( plt_x_vec ... , 
    layout = ( x_vars, 1 ), 
    size = (1000, 400*x_vars), 
    plot_title = "testing data", 
)
