using GaussianSINDy
using CSV 
using DataFrames 

# ----------------------- #
# load data 

csv_file = "test/data/jake_robot_data.csv" 

# wrap in data frame --> Matrix 
df   = CSV.read(csv_file, DataFrame) 
data = Matrix(df) 

# extract variables 
t = data[:,1] 
x = data[:,2:end-2]
u = data[:,end-1:end] 

x_vars = size(x, 2)
u_vars = size(u, 2) 
poly_order = x_vars 
n_vars = x_vars + u_vars 

# use forward finite differencing 
dx_fd = fdiff(t, x, 1) 
# dx_true = dx_true_fn

# massage data, generate rollovers  
rollover_up_ind = findall( x -> x > 100, dx_fd[:,4] ) 
rollover_dn_ind = findall( x -> x < -100, dx_fd[:,4] ) 
for i = 1 : length(rollover_up_ind) 

    i0   = rollover_up_ind[i] + 1 
    ifin = rollover_dn_ind[i] 
    rollover_rng = x[ i0 : ifin , 4 ]
    dθ = π .- rollover_rng 
    θ  = -π .- dθ 
    x[ i0 : ifin , 4 ] = θ

end 

# use central finite differencing now  
dx_fd = fdiff(t, x, 2) 
# dx_true = dx_true_fn

# ----------------------- #
# split into training and test data 
test_fraction = 0.2 
portion       = 5 
u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
x_train,  x_test  = split_train_test( x, test_fraction, portion ) 
dx_train, dx_test = split_train_test( dx_fd, test_fraction, portion ) 


## ============================================ ##
# SINDy vs. GPSINDy 

# x_GP,  Σ_xGP,  hp = post_dist_SE( t_train, x_train, t_train )              # step -1 
# dx_GP, Σ_dxGP, hp = post_dist_SE( x_GP, dx_train, x_GP )    # step 0 

x_GP_train  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
dx_GP_train = gp_post( x_GP_train, 0*dx_train, x_GP_train, 0*dx_train, dx_train ) 

x_GP_test   = gp_post( t_test, 0*x_test, t_test, 0*x_test, x_test ) 
dx_GP_test  = gp_post( x_GP_test, 0*dx_test, x_GP_test, 0*dx_test, dx_test ) 

# ----------------------- #

λ = 1e-3 
# Ξ = SINDy_c_test( x, u, dx_fd, λ ) 
Ξ_sindy       = SINDy_test( x_train, dx_train, λ, u_train ) 
Ξ_sindy_terms = pretty_coeffs( Ξ_sindy, x_train, u_train ) 

# Ξ_gpsindy       = SINDy_test( x_GP_train, dx_GP_train, λ, u_train ) 
Θx_gpsindy = pool_data_test( [ x_GP_train u_train ], n_vars, poly_order ) 
Ξ_gpsindy  = SINDy_test( x_GP_train, dx_GP_train, λ, u_train ) 
Ξ_gpsindy_terms = pretty_coeffs( Ξ_gpsindy, x_GP_train, u_train ) 

# round 2 of gpsindy 
dx_mean  = Θx_gpsindy * Ξ_gpsindy 
dx_post  = gp_post( x_GP_train, dx_mean, x_GP_train, dx_mean, dx_train ) 
Θx_gpsindy   = pool_data_test( [ x_GP_train u_train ], n_vars, poly_order )
Ξ_gpsindy_x2 = SINDy_test( x_GP_train, dx_post, λ, u_train )
Ξ_gpsindy_x2_terms = pretty_coeffs( Ξ_gpsindy_x2, x_GP_train, u_train ) 


## ============================================ ##
# plot smoothed data 

p_nvars = [] 
for i = 1 : x_vars 
    plt = plot( legend = :outerright, title = string("dx", i) )
        scatter!( plt, t_train, dx_train[:,i], label = "FD" ) 
        plot!( plt, t_train, dx_GP_train[:,i], label = "GP" ) 
    push!( p_nvars, plt ) 
end 
p_nvars = plot( p_nvars ... ,  
    layout = (x_vars, 1), 
    size   = [600 1200], 
    plot_title = "FD vs GP training data"
) 
display(p_nvars) 


## ============================================ ##

using Plots 

# SINDy alone 
Θx = pool_data_test( [x_train u_train], n_vars, poly_order) 
# Ξ_sindy = sparsify_dynamics_test( Θx, dx_fd, λ, x_vars ) 
dx_sindy = Θx * Ξ_sindy 

# GPSINDy 
Θx = pool_data_test( [x_GP_train u_train], n_vars, poly_order) 
# Ξ_gpsindy = sparsify_dynamics_test( Θx, dx_GP, λ, x_vars ) 
dx_gpsindy = Θx * Ξ_gpsindy 

plt = plot( title = "dx: meas vs. sindy", legend = :outerright )
scatter!( plt, t_train, dx_train[:,1], c = :black, ms = 3, label = "meas (finite diff)" )
plot!( plt, t_train, dx_GP_train[:,1], c = :blue, label = "GP" )
plot!( plt, t_train, dx_sindy[:,1], c = :red, ls = :dash, label = "SINDy" )   
plot!( plt, t_train, dx_gpsindy[:,1], c = :green, ls = :dashdot, label = "GPSINDy" )   


## ============================================ ##
# validate 

x_vars = size(x_train, 2)
u_vars = size(u_train, 2) 
poly_order = x_vars 

dx_fn_sindy      = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy ) 
dx_fn_gpsindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
dx_fn_gpsindy_x2 = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy_x2 ) 

xu0 = x_test[1,:] 
push!( xu0, u_test[1,1] ) 
push!( xu0, u_test[1,2] ) 
dx0_test = dx_fn_sindy( xu0, 0, 0 ) 

x0 = x_test[1,:] 
x_unicycle_test   = integrate_euler_unicycle(unicycle_realistic, x0, t_test, u_test) 
x_sindy_test      = integrate_euler( dx_fn_sindy, x0, t_test, u_test ) 
x_gpsindy_test    = integrate_euler( dx_fn_gpsindy, x0, t_test, u_test ) 
x_gpsindy_x2_test = integrate_euler( dx_fn_gpsindy_x2, x0, t_test, u_test ) 

# plot_states(t_train, x_train_noise, t_test, x_test_noise, t_test, x_sindy_val, t_test, x_gpsindy_val, t_test, x_gpsindy_x2_val, t_test, x_nn_val)
# plot_test_data(t_test, x_test_noise, t_test, x_sindy_val, t_test, x_gpsindy_val, t_test, x_gpsindy_x2_val, t_test, x_nn_val) 

## ============================================ ##
# plot 

using Plots 
using Latexify

xmin, dx, xmax = min_d_max(t_test)

p_vec = [] 
for i = 1 : x_vars 

    # ymin, dy, ymax = min_d_max([ x_true_test[:, i]; x_gpsindy_test[:,i] ])
    ymin = -5 
    ymax = 4 
    dy   = 3 

    p = plot( t_test, x_test[:,i], 
        c       = :gray, 
        label   = "test", 
        legend  = :outerright, 
        xlabel  = "Time (s)", 
        xticks  = xmin:dx:xmax,
        yticks  = ymin:dy:ymax,
        ylim    = (ymin, ymax), 
        title   = string(latexify("x_$(i)")),
    ) 
    plot!( p, t_test, x_unicycle_test[:,i], 
        c       = :green, 
        label   = "unicycle", 
        xticks  = xmin:dx:xmax,
        yticks  = ymin:dy:ymax,
        ls      = :dash, 
    ) 
    plot!( p, t_test, x_sindy_test[:,i], 
        c       = :red, 
        label   = "SINDy", 
        xticks  = xmin:dx:xmax,
        yticks  = ymin:dy:ymax,
        ls      = :dashdot, 
    ) 
    plot!( p, t_test, x_gpsindy_test[:,i], 
        c       = :blue, 
        label   = "GPSINDy", 
        xticks  = xmin:dx:xmax,
        yticks = ymin:dy:ymax,
        ls      = :dot, 
    )
    push!( p_vec, p ) 

end 

# p = deepcopy( p_vec[end] ) 
# plot!( p, 
#     legend = ( -0.1, 0.6 ), 
#     framestyle = :none, 
#     title = "",      
# )  
# push!( p_vec, p ) 

pfig = plot(  p_vec ... , 
    layout = grid( x_vars, 1 ), 
    size   = [ 600 x_vars * 400 ],         
    margin = 5Plots.mm,
    bottom_margin = 14Plots.mm,
)

display(pfig) 
