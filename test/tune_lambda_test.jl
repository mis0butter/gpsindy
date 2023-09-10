# this is a script for tuning lambda for each dx based on the error from the TRAINING DATA 

using GaussianSINDy 
using LinearAlgebra 
using Plots 

## ============================================ ##

fn = unicycle 

λ = 0.1 
data_train, data_test = ode_train_test( fn )

x_train  = data_train.x_noise 
dx_train = data_train.dx_noise 
u_train  = data_train.u 
t_train  = data_train.t 

x_true   = data_train.x_true 
dx_true  = data_train.dx_true 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x_train, u_train ) 
poly_order = 4 

Ξ_true       = sindy_stls( x_true, dx_true, λ, poly_order, u_train ) 
Ξ_true_terms = pretty_coeffs( Ξ_true, x_true, u_train ) 


## ============================================ ##
# smooth with GPs 

# first - smooth measurements with Gaussian processes 
x_GP_train  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
dx_GP_train = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

# get x0 from smoothed data 
x0 = x_GP_train[1,:] 

# SINDy (STLS) 
λ = 0.1 
Θx_sindy      = pool_data_test( [ x_train u_train ], n_vars, poly_order ) 
Ξ_sindy_stls  = sindy_stls( x_train, dx_train, λ, poly_order, u_train ) 
dx_fn_sindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy_stls ) 
dx_sindy      = Θx_sindy * Ξ_sindy_stls  
x_sindy_train = integrate_euler( dx_fn_sindy, x0, t_train, u_train ) 

# integrate unicycle (with GP) 
x_unicycle_train = integrate_euler_unicycle( fn, x0, t_train, u_train ) 
for i = 1 : x_vars 
    println( "norm err = ", norm( x_unicycle_train[:,i] - x_true[:,i] )  ) 
end 

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
    
    err_ξ_vec   = [] 
    err_xj_vec  = [] 
    err_dxj_vec = [] 

    a_dx = Animation() 
    a_x = Animation() 

    # DX PLOT 
    ymin, dy, ymax = min_d_max( dx_true[:,j] ) 
    plt_dx = plot( t_train, dx_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "dx", 
        ylim   = ( ymin, ymax ), 
        title = string("dx", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) ; frame(a_dx, plt_dx) 
    plot!( plt_dx, t_train, dx_train[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) ; frame(a_dx, plt_dx) 
        plot!( plt_dx, t_train, dx_GP_train[:,j], 
        label = "GP", 
        ls    = :dashdot,   

    ) ; frame(a_dx, plt_dx) 
    plot!( plt_dx, t_train, dx_sindy[:,j], 
        label = "SINDy",
        ls    = :dashdotdot,  
    ) ; frame(a_dx, plt_dx) 

    # X PLOT 
    ymin, dy, ymax = min_d_max( x_true[:,j] ) 
    plt_x = plot( t_train, x_true[:,j], 
        label = "true", 
        xlabel = "t (s)", 
        ylabel = "x", 
        ylim   = ( ymin, ymax ), 
        title = string("x", j, ": true vs GP vs SINDy vs GPSINDy"), 
        legend = :outerright,
        size = (1000, 400), 
    ) 
    plot!( plt_x, t_train, x_train[:,j], 
        label = "noise", 
        ls    = :dash,  
    ) ; frame(a_x, plt_x) 
    plot!( plt_x, t_train, x_GP_train[:,j], 
        label = "GP", 
        ls    = :dashdot,   
    ) ; frame(a_x, plt_x) 
    plot!( plt_x, t_train, x_unicycle_train[:,j], 
        label = "unicycle", 
        ls    = :dashdotdot,   
    ) ; frame(a_x, plt_x) 
    plt_x_sindy = deepcopy(plt_x) 
    plot!( plt_x_sindy, t_train, x_sindy_train[:,j], 
        label = "SINDy",
        ls    = :dot,  
    ) ; frame(a_x, plt_x_sindy) 

    for i = 1 : length(λ_vec) 
        # j = 1 
    
        λ = λ_vec[i] 
        println( "i = ", i, ". λ = ", λ ) 

        Θx_gpsindy = pool_data_test( [ x_GP_train u_train ], n_vars, poly_order ) 
        Ξ_gpsindy  = sindy_lasso( x_GP_train, dx_GP_train, λ, poly_order, u_train ) 
        dx_gpsindy = Θx_gpsindy * Ξ_gpsindy 

        if sum(Ξ_gpsindy[:,j]) == 0 
            break 
        end 

        Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 
        display(Ξ_gpsindy_terms) 

        dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
        x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0, t_train, u_train ) 
        err_ξ_norm      = norm( Ξ_true[:,j] - Ξ_gpsindy[:,j] ) 
        err_xj_norm     = norm( x_gpsindy_train[:,j] - x_GP_train[:,j] ) 
        err_dxj_norm    = norm( dx_gpsindy[:,j] - dx_GP_train[:,j] ) 

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

end 

