using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using DifferentialEquations


## ============================================ ## 
# # rollout_1.csv  

freq_hz = 5 
noise   = 0  

# rollout_1.csv  
csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/5hz_noise_0/rollout_1.csv"

# tuning parameters  
interpolate_gp = true 

σn      = 0.02 
opt_σn  = false 


df_min_err_sindy, df_min_err_gpsindy, f_train, f_test = cross_validate_csv_path_file( csv_path_file, σn, opt_σn, freq_hz, noise, interpolate_gp ) 

f_train 


## ============================================ ## 
## ============================================ ## 
#  the set-up 

# generate data from an ode 
# Define the ODE system (Lorenz system as an example)
function lorenz!(du, u, p, t)
    σ, ρ, β = p
    du[1] = σ * (u[2] - u[1])
    du[2] = u[1] * (ρ - u[3]) - u[2]
    du[3] = u[1] * u[2] - β * u[3]
end

# Set up parameters 
tspan   = (0.0, 10.0) 
x0      = [1.0, 0.0, 0.0] 
p       = [10.0, 28.0, 8/3] 
dt      = 0.02 

# Generate clean data
prob = ODEProblem(lorenz!, x0, tspan, p)
sol  = solve(prob, Tsit5(), saveat = dt) 

# Extract time and state variables 
t = sol.t
x = Array(sol)' 

# Calculate derivatives
dx = similar(x) 
for i in 1:size(x, 1) 
    du = similar(x0) 
    lorenz!(du, x[i, :], p, t[i]) 
    dx[i, :] = du 
end 

## ============================================ ## 


# Add noise to create noisy data
noise_level = 0.1
x_noise     = x  .+ noise_level * randn(size(x)) 
dx_noise    = dx .+ noise_level * randn(size(dx)) 

# create zero control input 
u = zeros(size(x)) 

# split into training and test data 
test_fraction = 0.3 
portion       = "last" 

# split t and u 
t_train,  t_test  = split_train_test( t, test_fraction, portion ) 
u_train,  u_test  = split_train_test( u, test_fraction, portion ) 
t_train = t_train[:] ; t_test = t_test[:] 

# split x and dx (true) 
x_train_true, x_test_true   = split_train_test( x, test_fraction, portion ) 
dx_train_true, dx_test_true = split_train_test( dx, test_fraction, portion )     

# split x and dx (noise)
x_train_noise,  x_test_noise  = split_train_test( x_noise, test_fraction, portion ) 
dx_train_noise, dx_test_noise = split_train_test( dx_noise, test_fraction, portion ) 

# data structs have fields 
#   t 
#   u 
#   x_true 
#   dx_true 
#   x_noise 
#   dx_noise 

# Create data structures
data_train = ( t = t_train, x_true = x_train_true, dx_true = dx_train_true, x_noise = x_train_noise, dx_noise = dx_train_noise, u = u_train ) 

data_test  = ( t = t_test, x_true = x_test_true, dx_true = dx_test_true, x_noise = x_test_noise, dx_noise = dx_test_noise, u = u_test )      

## ============================================ ## 

# interpolation factor 
interp_factor = 2 
t_interp = interpolate_time( data_train.t, interp_factor )  

x_col, x_row = size( data_train.x_noise ) 
u_col, u_row = size( data_train.u ) 

x_train_GP  = smooth_gp_posterior( t_interp, zeros( interp_factor * x_col, x_row ), data_train.t, 0*data_train.x_noise, data_train.x_noise, σ_n, opt_σn ) 



## ============================================ ## 
# the interpolation 

# let's double the points 
t_train_x2 = t_double_fn( data_train.t ) 

x_col, x_row = size( data_train.x_noise ) 
u_col, u_row = size( data_train.u ) 

# first - smooth training data with Gaussian processes 
x_train_GP  = smooth_gp_posterior( t_train_x2, zeros( 2 * x_col, x_row ), data_train.t, 0*data_train.x_noise, data_train.x_noise, σ_n, opt_σn ) 
dx_train_GP = smooth_gp_posterior( x_train_GP, zeros( 2 * x_col, x_row ), data_train.x_noise, 0*data_train.dx_noise, data_train.dx_noise, σ_n, opt_σn  ) 
# u_train_x2  = smooth_gp_posterior( t_train_x2, zeros( 2 * u_col, u_row ), data_train.t, 0*data_train.u, data_train.u, σ_n, opt_σn  ) 
u_train_x2 = interp_dbl_fn( data_train.u ) 

# smooth testing data 
x_test_GP   = smooth_gp_posterior( data_test.t, 0*data_test.x_noise, data_test.t, 0*data_test.x_noise, data_test.x_noise ) 
dx_test_GP  = smooth_gp_posterior( x_test_GP, 0*data_test.dx_noise, x_test_GP, 0*data_test.dx_noise, data_test.dx_noise ) 





