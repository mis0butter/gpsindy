using GaussianSINDy
using Flux
# using LineSearches 

# choose ODE, plot states --> measurements 
fn = predator_prey

# constants 
λ = 0.1

## ============================================ ## 

# set up noise vec 
noise_vec      = []
noise_vec_iter = 0.05 : 0.05 : 0.25 
for i in noise_vec_iter
    for j = 1:2 
        push!(noise_vec, i)
    end
end 
# noise_vec = [ 0.01 ] 

# ----------------------- # 
# start MC loop 

Ξ_vec = []
Ξ_hist = Ξ_struct([], [], [], [], []) 
Ξ_err_hist = Ξ_err_struct([], [], [], [])
for noise = noise_vec 
    Ξ_hist, Ξ_err_hist = sindy_nn_gpsindy( fn, noise, λ, Ξ_hist, Ξ_err_hist, 0 ) 
end 

## ============================================ ##
# plot quartiles 

Ξ_sindy_stls_err = Ξ_err_hist.sindy_stls
Ξ_sindy_lasso_err = Ξ_err_hist.sindy_lasso 

Ξ_gpsindy_err    = Ξ_err_hist.gpsindy 
Ξ_nn_err         = Ξ_err_hist.nn 
plot_med_quarts_gpsindy_x2(Ξ_sindy_lasso_err, Ξ_gpsindy_err, Ξ_nn_err, noise_vec)


# ## ============================================ ##
# # save data 

# using JLD2

# timestamp = Dates.format(now(), "YYYYmmdd-HHMMSS")
# dir_name  = joinpath(@__DIR__, "outputs", "runs_$timestamp")
# @assert !ispath(dir_name) "Somebody else already created the directory"
# if !ispath(dir_name)
#     mkdir(dir_name)
# end 

# # save 
# jldsave(string( dir_name, "\\Ξ_err_hist.jld2" ); Ξ_err_hist)


# ## ============================================ ##
# ## ============================================ ##
# # single test case

# # generate true states 
# x0, dt, t, x_true, dx_true, dx_fd, p = ode_states(fn, 0, 2)
# noise = 0.1 

# # truth coeffs 
# n_vars = size(x_true, 2);
# poly_order = n_vars;
# Ξ_true = sindy_stls(x_true, dx_true, λ)

# # add noise 
# println("noise = ", noise)
# x_noise = x_true + noise * randn(size(x_true, 1), size(x_true, 2))
# dx_noise = dx_true + noise * randn(size(dx_true, 1), size(dx_true, 2))

# # split into training and test data 
# test_fraction = 0.2
# portion = 5
# t_train, t_test               = split_train_test(t, test_fraction, portion)
# x_train_true, x_test_true     = split_train_test(x_true, test_fraction, portion)
# dx_train_true, dx_test_true   = split_train_test(dx_true, test_fraction, portion)
# x_train_noise, x_test_noise   = split_train_test(x_noise, test_fraction, portion)
# dx_train_noise, dx_test_noise = split_train_test(dx_noise, test_fraction, portion)

# # ----------------------- # 
# # standardize  
# x_stand_noise = stand_data(t_train, x_train_noise)
# x_stand_true = stand_data(t_train, x_train_true)
# dx_stand_true = dx_true_fn(t_train, x_stand_true, p, fn)
# dx_stand_noise = dx_stand_true + noise * randn(size(dx_stand_true, 1), size(dx_stand_true, 2))

# # set training data for GPSINDy 
# x_train = x_stand_noise 
# dx_train = dx_stand_noise


# ## ============================================ ##
# # SINDy vs. GPSINDy vs. GPSINDy_x2 

# # SINDy by itself 
# Θx_sindy = pool_data_test(x_train, n_vars, poly_order)
# Ξ_sindy = sindy_stls(x_train, dx_train, λ)

# # ----------------------- #
# # GPSINDy (first) 

# # step -1 : smooth x measurements with t (temporal)  
# x_train_GP = gp_post(t_train, 0 * x_train, t_train, 0 * x_train, x_train)

# # step 0 : smooth dx measurements with x_GP (non-temporal) 
# dx_train_GP = gp_post(x_train_GP, 0 * dx_train, x_train_GP, 0 * dx_train, dx_train)

# # SINDy 
# Θx_gpsindy = pool_data_test(x_train_GP, n_vars, poly_order)
# Ξ_gpsindy = sindy_stls(x_train_GP, dx_train_GP, λ)

# # ----------------------- #
# # GPSINDy (second) 

# # step 2: GP 
# dx_mean  = Θx_gpsindy * Ξ_gpsindy
# dx_train = dx_stand_noise
# dx_post  = gp_post(x_train_GP, dx_mean, x_train_GP, dx_mean, dx_train)

# # step 3: SINDy 
# Θx_gpsindy = pool_data_test(x_train_GP, n_vars, poly_order)
# Ξ_gpsindy_x2 = sindy_stls(x_train_GP, dx_post, λ)


# ## ============================================ ##
# # Train NN on the data

# # ----------------------- #
# # Define the 2-layer MLP
# dx_noise_nn_x1 = train_nn_predict(x_train_noise, dx_train_noise[:, 1], 100, 2)
# dx_noise_nn_x2 = train_nn_predict(x_train_noise, dx_train_noise[:, 2], 100, 2)

# # Concanate the two outputs to make a Matrix
# dx_noise_nn = hcat(dx_noise_nn_x1, dx_noise_nn_x2)

# Θx_nn = pool_data_test(x_train_noise, n_vars, poly_order)
# Ξ_nn = sindy_stls(x_train_noise, dx_noise_nn, λ)

# ## ============================================ ##
# # validate data 

# # function build_dx_fn(poly_order, x_vars, u_vars, z_fd) 

# x_vars = size(x_true, 2)
# u_vars = 0
# dx_sindy_fn      = build_dx_fn(poly_order, x_vars, u_vars, Ξ_sindy)
# dx_gpsindy_fn    = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy)
# dx_gpsindy_x2_fn = build_dx_fn(poly_order, x_vars, u_vars, Ξ_gpsindy_x2)
# dx_nn_fn         = build_dx_fn(poly_order, x_vars, u_vars, Ξ_nn)

# t_sindy_val, x_sindy_val           = validate_data(t_test, x_test_noise, dx_sindy_fn, dt)
# # t_sindy_val,      x_sindy_val      = validate_data(t_test, x_test, fn, dt) 
# t_gpsindy_val, x_gpsindy_val       = validate_data(t_test, x_test_noise, dx_gpsindy_fn, dt)
# t_gpsindy_x2_val, x_gpsindy_x2_val = validate_data(t_test, x_test_noise, dx_gpsindy_x2_fn, dt)
# t_nn_val, x_nn_val                 = validate_data(t_test, x_test_noise, dx_nn_fn, dt)

# # plot!! 
# plot_states(t_train, x_train_noise, t_test, x_test_noise, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)
# plot_test_data(t_test, x_test_noise, t_sindy_val, x_sindy_val, t_gpsindy_val, x_gpsindy_val, t_gpsindy_x2_val, x_gpsindy_x2_val, t_nn_val, x_nn_val)

