using GaussianSINDy 


## ============================================ ##
# truth 

fn = unicycle 
data_train, data_test = ode_train_test( fn ) 

λ = 0.1 
Ξ_true = SINDy_test( data_train.x_true, data_train.dx_true, λ, data_train.u ) 
Ξ_true_terms = pretty_coeffs(Ξ_true, data_train.x_true, data_train.u) 

Ξ_sindy = SINDy_test( data_train.x_noise, data_train.dx_noise, λ, data_train.u ) 
Ξ_sindy_terms = pretty_coeffs(Ξ_sindy, data_train.x_noise, data_train.u) 

# GPSINDy 
x_GP  = gp_post( data_train.t, 0*data_train.x_noise, data_train.t, 0*data_train.x_noise, data_train.x_noise ) 
dx_GP = gp_post( x_GP, 0*data_train.dx_noise, x_GP, 0*data_train.dx_noise, data_train.dx_noise ) 
Ξ_gpsindy       = SINDy_test( x_GP, dx_GP, λ, data_train.u ) 
Ξ_gpsindy_terms = pretty_coeffs(Ξ_gpsindy, x_GP, data_train.u) 


# # ----------------------- #
# # Define the 2-layer MLP
# dx_noise_nn_x1 = train_nn_predict(data_train.x_noise, data_train.dx_noise[:, 1], 100, 2)
# dx_noise_nn_x2 = train_nn_predict(data_train.x_noise, data_train.dx_noise[:, 2], 100, 2)

# # Concanate the two outputs to make a Matrix
# dx_noise_nn = hcat(dx_noise_nn_x1, dx_noise_nn_x2)

# Ξ_nn       = SINDy_test( data_train.x_noise, dx_noise_nn, λ, data_train.u ) 
# Ξ_nn_terms = pretty_coeffs(Ξ_nn, data_train.x_noise, data_train.u) 


## ============================================ ##
# validate 

x_vars = size( data_train.x_true, 2 ) 
u_vars = size( data_train.u, 2 ) 
poly_order = x_vars 

dx_fn_true       = build_dx_fn( poly_order, x_vars, u_vars, Ξ_true ) 
dx_fn_sindy      = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
dx_fn_gpsindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 

xu0 = data_train.x_true[1,:] 
push!( xu0, data_train.u[1,1] ) 
push!( xu0, data_train.u[1,2] ) 
dx0_test = dx_fn_sindy( xu0, 0, 0 ) 

x_true_test    = integrate_euler( dx_fn_true, data_test.x_true[1,:], data_test.t, data_test.u ) 
x_sindy_test   = integrate_euler( dx_fn_sindy, data_test.x_true[1,:], data_test.t, data_test.u ) 
x_gpsindy_test = integrate_euler( dx_fn_gpsindy, data_test.x_true[1,:], data_test.t, data_test.u ) 
t_test = data_test.t 

## ============================================ ##
# plot 

using Plots 
using Latexify

xmin, dx, xmax = min_d_max(t_test)

p_vec = [] 
for i = 1 : x_vars 

    # ymin, dy, ymax = min_d_max([ x_true_test[:, i]; x_gpsindy_test[:,i] ])
    ymin = -9 
    ymax = 2 
    dy   = 3 

    p = plot( t_test, x_true_test[:,i], 
        c      = :gray, 
        label  = "test", 
        xlabel = "Time (s)", 
        xticks = xmin:dx:xmax,
        yticks = ymin:dy:ymax,
        ylim   = (ymin, ymax), 
        title  = string(latexify("x_$(i)")),
    ) 
    plot!( p, t_test, x_sindy_test[:,i], 
        c       = :red, 
        label   = "SINDy", 
        xticks  = xmin:dx:xmax,
        yticks  = ymin:dy:ymax,
        ls      = :dash, 
        title   = string(latexify("x_$(i)")),
    ) 
    plot!( p, t_test, x_gpsindy_test[:,i], 
        c       = :blue, 
        label   = "GPSINDy", 
        xticks  = xmin:dx:xmax,
        yticks = ymin:dy:ymax,
        ls      = :dashdot, 
        title   = string(latexify("x_$(i)")),
    )
    push!( p_vec, p ) 

end 

p = deepcopy( p_vec[end] ) 
plot!( p, 
    legend = ( -0.1, 0.6 ), 
    framestyle = :none, 
    title = "",      
)  
push!( p_vec, p ) 

pfig = plot(  p_vec ... , 
    layout = grid(1, x_vars + 1, widths=[0.2, 0.2, 0.2, 0.2, 0.25]), 
    size   = [ x_vars * 400 250 ],         
    margin = 5Plots.mm,
    bottom_margin = 14Plots.mm,
)

display(pfig) 

# dt = data_train.t[2] - data_train.t[1] 
# t_sindy_val,   x_sindy_val   = validate_data( data_test.t, [ data_test.x_noise data_test.u ], dx_sindy_fn, dt ) 
# t_gpsindy_val, x_gpsindy_val = validate_data( data_test.t, x_GP_train, dx_gpsindy_fn, dt ) 








