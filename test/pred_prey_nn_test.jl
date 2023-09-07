using GaussianSINDy
using Flux
# using LineSearches 


## ============================================ ##

# choose ODE, plot states --> measurements 
fn = predator_prey

# constants 
λ = 0.1

# # set up noise vec 
# noise_vec = []
# noise_vec_iter = 0.0:0.01:0.3
# for i in noise_vec_iter
#     for j = 1:10
#         push!(noise_vec, i)
#     end
# end
# noise_vec = collect( 0 : 0.05 : 0.2 ) 
noise = 0.2

# ----------------------- # 
# start MC loop 

# Ξ_vec = []
# Ξ_hist = Ξ_struct([], [], [], [])
# Ξ_err_hist = Ξ_err_struct([], [], [])
# for noise = noise_vec 
#     Ξ_hist, Ξ_err_hist = gpsindy_x2( fn, noise, λ, Ξ_hist, Ξ_err_hist ) 
# end 

# ----------------------- # 

# generate true states 
x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2)

# truth coeffs 
n_vars = size(x_true, 2);
poly_order = n_vars;
Ξ_true = SINDy_test(x_true, dx_true, λ)

# add noise 
println("noise = ", noise)
x_noise = x_true + noise * randn(size(x_true, 1), size(x_true, 2))
dx_noise = dx_true + noise * randn(size(dx_true, 1), size(dx_true, 2))

# split into training and test data 
test_fraction = 0.2
portion = 5
t_train, t_test               = split_train_test(t, test_fraction, portion)
x_train_true, x_test_true     = split_train_test(x_true, test_fraction, portion)
dx_train_true, dx_test_true   = split_train_test(dx_true, test_fraction, portion)
x_train_noise, x_test_noise   = split_train_test(x_noise, test_fraction, portion)
dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion)

# ----------------------- # 
# standardize  
x_stand_noise = stand_data(t_train, x_train_noise)
x_stand_true = stand_data(t_train, x_train_true)
dx_stand_true = dx_true_fn(t_train, x_stand_true, p, fn)
dx_stand_noise = dx_stand_true + noise * randn(size(dx_stand_true, 1), size(dx_stand_true, 2))

# set training data for GPSINDy 
x_train = x_stand_noise 
dx_train = dx_stand_noise

## ============================================ ##
# SINDy vs. GPSINDy vs. GPSINDy_x2 

# SINDy by itself 
Θx_sindy = pool_data_test(x_train, n_vars, poly_order)
Ξ_sindy = SINDy_test(x_train, dx_train, λ)

# ----------------------- #
# GPSINDy (first) 

# step -1 : smooth x measurements with t (temporal)  
x_train_GP = gp_post(t_train, 0 * x_train, t_train, 0 * x_train, x_train)

# step 0 : smooth dx measurements with x_GP (non-temporal) 
dx_train_GP = gp_post(x_train_GP, 0 * dx_train, x_train_GP, 0 * dx_train, dx_train)

# SINDy 
Θx_gpsindy = pool_data_test(x_train_GP, n_vars, poly_order)
Ξ_gpsindy = SINDy_test(x_train_GP, dx_train_GP, λ)

# ----------------------- #
# GPSINDy (second) 

# step 2: GP 
dx_mean  = Θx_gpsindy * Ξ_gpsindy
dx_train = dx_stand_noise
dx_post  = gp_post(x_train_GP, dx_mean, x_train_GP, dx_mean, dx_train)

# step 3: SINDy 
Θx_gpsindy = pool_data_test(x_train_GP, n_vars, poly_order)
Ξ_gpsindy_x2 = SINDy_test(x_train_GP, dx_post, λ)


## ============================================ ##
# Train NN on the data

# ----------------------- #
# Define the 2-layer MLP
dx_noise_nn_x1 = train_nn_predict(x_train_noise, dx_train_noise[:, 1], 100, 2)
dx_noise_nn_x2 = train_nn_predict(x_train_noise, dx_train_noise[:, 2], 100, 2)

# Concanate the two outputs to make a Matrix
dx_noise_nn = hcat(dx_noise_nn_x1, dx_noise_nn_x2)

Θx_nn = pool_data_test(x_train_noise, n_vars, poly_order)
Ξ_nn = SINDy_test(x_train_noise, dx_noise_nn, λ)

## ============================================ ##
# validate data 

# function build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

x_vars = size(x_true, 2)
u_vars = 0
dx_sindy_fn      = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy)
dx_gpsindy_fn    = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy)
dx_gpsindy_x2_fn = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy_x2)
dx_nn_fn         = build_dx_fn(poly_order, x_vars, u_vars, Ξ_nn)

t_sindy_val, x_sindy_val           = validate_data(t_test, x_test_noise, dx_sindy_fn, dt)
# t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, fn, dt) 
t_gpsindy_val, x_gpsindy_val       = validate_data(t_test, x_test_noise, dx_gpsindy_fn, dt)
t_gpsindy_x2_val, x_gpsindy_x2_val = validate_data(t_test, x_test_noise, dx_gpsindy_x2_fn, dt)
t_nn_val, x_nn_val                 = validate_data(t_test, x_test_noise, dx_nn_fn, dt)

# plot!! 
plot_states(t_train, x_train_noise, t_test, x_test_noise, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)
plot_test_data(t_test, x_test_noise, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)


## ============================================ ##
# plot quartiles 

# Ξ_sindy_err = Ξ_err_hist.sindy;
# Ξ_gpsindy_err = Ξ_err_hist.gpsindy;
# Ξ_gpsindy_x2_err = Ξ_err_hist.gpsindy_x2;
# plot_med_quarts_gpsindy_x2(Ξ_sindy_err, Ξ_gpsindy_err, Ξ_gpsindy_x2_err, noise_vec)


## ============================================ ##
## ============================================ ##



# x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2)

# # truth coeffs 
# n_vars = size(x_true, 2);
# poly_order = n_vars;
# Ξ_true = SINDy_test(x_true, dx_true, λ)

# ## ============================================ ##
# # CASE 7

# # add noise 
# println("noise = ", noise)
# x_true = stand_data(t, x_true)
# dx_true = dx_true_fn(t, x_true, p, fn)
# Ξ_true = SINDy_test(x_true, dx_true, λ)

# x_noise = x_true + noise * randn(size(x_true, 1), size(x_true, 2))
# dx_noise = dx_true + noise * randn(size(dx_true, 1), size(dx_true, 2))

# Θx_sindy = pool_data_test(x_noise, n_vars, poly_order)
# Ξ_sindy = SINDy_test(x_noise, dx_noise, λ)

# # smooth measurements 
# x_GP, Σ_xsmooth, hp = post_dist_SE(t, x_noise, t)
# dx_GP, Σ_dxsmooth, hp = post_dist_SE(x_GP, dx_noise, x_GP)

# Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order)
# Ξ_gpsindy = SINDy_test(x_GP, dx_GP, λ)

# # ----------------------- #

# # step 2 
# dx_mean = Θx_gpsindy * Ξ_gpsindy

# x_vec = []
# for i = 1:size(x_noise, 1)
#     push!(x_vec, x_noise[i, :])
# end
# dx_post = 0 * dx_noise

# # optimize hyperparameters 
# # i = 1 
# for i = 1:size(x_true, 2)

#     # kernel  
#     mZero = MeanZero()            # zero mean function 
#     kern = SE(0.0, 0.0)        # squared eponential kernel (hyperparams on log scale) 
#     log_noise = log(0.1)              # (optional) log std dev of obs 

#     y_train = dx_noise[:, i] - dx_mean[:, i]
#     gp = GP(x_GP', y_train, mZero, kern, log_noise)

#     optimize!(gp, method=LBFGS(linesearch=LineSearches.BackTracking()))

#     # return HPs 
#     σ_f = sqrt(gp.kernel.σ2)
#     l = sqrt.(gp.kernel.ℓ2)
#     σ_n = exp(gp.logNoise.value)
#     hp = [σ_f, l, σ_n]

#     K = k_SE(σ_f, l, x_vec, x_vec)
#     dx_post[:, i] = dx_mean[:, i] + K * ((K + σ_n^2 * I) \ (dx_noise[:, i] - dx_mean[:, i]))

# end

# Θx_gpsindy = pool_data_test(x_GP, n_vars, poly_order)
# Ξ_gpsindy_gpsindy = SINDy_test(x_GP, dx_post, λ)

# println("Ξ_true - Ξ_sindy = ", norm(Ξ_true - Ξ_sindy))
# println("Ξ_true - Ξ_gpsindy = ", norm(Ξ_true - Ξ_gpsindy))
# println("Ξ_true - Ξ_gpsindy_gpsindy = ", norm(Ξ_true - Ξ_gpsindy_gpsindy))


# ## ============================================ ##
# # plot 

# p_states = []
# for i = 1:n_vars
#     plt = plot(t, x_true[:, i], label="true", c=:green)
#     scatter!(plt, x_noise, t[:, i], label="noise", c=:black, ms=3)
#     scatter!(plt, t, x_GP[:, i], label="smooth", c=:red, ms=1, markerstrokewidth=0)
#     plot!(plt, legend=:outerright, title=string("state ", i))
#     push!(p_states, plt)
# end
# p_states = plot(p_states...,
#     layout=(2, 1),
#     size=[800 300]
# )
# display(p_states)
# 