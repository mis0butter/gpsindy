
using GaussianSINDy
using LinearAlgebra
using Statistics
using CairoMakie
using Printf
using CSV, DataFrames 

using DifferentialEquations


## ============================================ ## 
## load data   
## ============================================ ## 

# rollout_1.csv  
csv_path_file = "test/data/jake_car_csvs_ctrlshift_no_trans/5hz_noise_0/rollout_1.csv"

data_train, data_test = make_data_structs( csv_path_file ) 

# plot the data 
fig = plot_car_x_dx( data_train.t, data_train.x_noise, data_train.dx_noise ) 


## ============================================ ## 
## smooth and interpolate the data using GPs 
## ============================================ ## 

# optimize noise hyperparameter 
σn     = 0.02 
opt_σn = true   

x_col, x_row = size( data_train.x_noise ) 
u_col, u_row = size( data_train.u ) 

# interpolation factor 
interp_factor  = 10 
t_train_interp = interpolate_array( data_train.t, interp_factor ) 
u_train_interp = interpolate_array( data_train.u, interp_factor )  

x_train_GP  = smooth_gp_posterior( data_train.t, zeros( x_col, x_row ), data_train.t, 0 * data_train.x_noise, data_train.x_noise, σn, opt_σn ) 
dx_train_GP = smooth_gp_posterior( x_train_GP, zeros( x_col, x_row ), data_train.x_noise, 0 * data_train.dx_noise, data_train.dx_noise, σn, opt_σn  ) 

x_train_gp_interp  = smooth_gp_posterior( t_train_interp, zeros( interp_factor * x_col, x_row ), data_train.t, 0 * data_train.x_noise, data_train.x_noise, σn, opt_σn ) 
dx_train_gp_interp = smooth_gp_posterior( x_train_gp_interp, zeros( interp_factor * x_col, x_row ), data_train.x_noise, 0 * data_train.dx_noise, data_train.dx_noise, σn, opt_σn ) 

data_train_gp_interp = data_struct( t_train_interp, u_train_interp, [], [], x_train_gp_interp, dx_train_gp_interp ) 
data_train_gp = data_struct( data_train.t, data_train.u, [], [], x_train_GP, dx_train_GP )  

data_train_gp_interp = ( 
    t  = t_train_interp, 
    u  = u_train_interp, 
    x  = x_train_gp_interp, 
    dx = dx_train_gp_interp 
) 
data_train_gp = ( 
    t  = data_train.t, 
    u  = data_train.u, 
    x  = x_train_GP, 
    dx = dx_train_GP 
)  


## ============================================ ## 
## plot the data 
## ============================================ ##  

# plot x 
fig = Figure() 
fig = Figure( size = (1000, 1000) )  
for i in 1:size(data_train.x_noise, 2)

    ax = Axis(fig[i, 1], title = "x$i")
    scatter!(ax, data_train.t, data_train.x_noise[:, i], label = "x$i (train)")
    scatter!(ax, data_test.t, data_test.x_noise[:, i], label = "x$i (test)")
    scatter!(ax, data_train.t, x_train_GP[:, i], label = "x$i (GP)", color = :red)
    scatter!(ax, t_train_interp, x_train_gp_interp[:, i], label = "x$i (GP interp)", color = :green, markersize = 4)

    axislegend(ax)
end 

# plot control inputs
for i in 1:size(data_train.u, 2)
    ax = Axis(fig[i, 2], title = "u$i")
    scatter!(ax, data_train.t, data_train.u[:, i], label = "u$i (original)")
    scatter!(ax, t_train_interp, u_train_interp[:, i], label = "u$i (interpolated)", color = :orange, markersize = 4)
    axislegend(ax)
end

# Adjust the layout
for i in 1:4
    rowsize!(fig.layout, i, Relative(0.25))
end
colsize!(fig.layout, 1, Relative(0.5))
colsize!(fig.layout, 2, Relative(0.5))

fig 


## ============================================ ## 
## discover ODEs 
## ============================================ ## 

# x_gpsindy_train, x_gpsindy_test = integrate_gpsindy_interp( x_train_GP, dx_train_GP, t_train_interp, u_train_interp, λ, data_train, data_test )  
    
# get x0 from noisy and smoothed data 
x0_train = data_train.x_noise[1,:]  
x0_test  = data_test.x_noise[1,:]  

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( data_train.x_noise, data_train.u ) 

# ----------------------- #
# sindy-lasso ! 
Ξ_gpsindy = sindy_lasso( x_train_GP, dx_train_GP, λ, u_train_x2 ) 

# integrate discovered dynamics 
dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
# x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, data_train.t, data_train.u ) 
x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train, t_interp, u_interp ) 
x_gpsindy_test  = integrate_euler( dx_fn_gpsindy, x0_test, data_test.t, data_test.u )


## ============================================ ## 
# check downsampling 

# Example usage:
t_downsampled, x_train_gp_downsampled = downsample_to_original(data_train.t, t_train_interp, x_train_gp_interp)

# Verify the shapes
println("Shape of original t: ", size(data_train.t))
println("Shape of downsampled t: ", size(t_downsampled))
println("Shape of original x: ", size(x_train_gp_interp))
println("Shape of downsampled x: ", size(x_train_gp_downsampled)) 



