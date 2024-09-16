using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using BenchmarkTools 
using CairoMakie 

Infiltrator.clear_disabled!() 
Infiltrator.toggle_async_check(false) 

## ============================================ ##
# get states and inputs 

path = "test/data/cyrus_quadcopter_csvs_sparse/" 
# state vars: px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz 
csv_files_vec = readdir( path ) 

# for i_csv in eachindex(csv_files_vec) 
i_csv = 2 

    csv_file = string( path, csv_files_vec[i_csv] ) 
    println( "csv_file = ", csv_files_vec[i_csv] ) 
    df   = CSV.read( csv_file, DataFrame ) 
    # header = [ 
    #     "t", 
    #     "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",   # states 
    #     "c1", "c2", "c3", "c4", "c5", "c6"                                              # commands  
    #     ] 
    data = Matrix(df) 
    t    = data[:,1] 
    x    = data[:,2:14] 
    u    = data[:,15:end] 
# end 

# all points 
N  = size(x, 1) 

# get derivatives 
x, dx_fd = unroll( t, x ) 

# ----------------------- #
# plot entire trajectory 

fig_entire_traj = plot_axes3d( ) 
fig_entire_traj = plot_line3d( x[:,1], x[:,2], x[:,3], fig_entire_traj ) 
fig_entire_traj = add_title3d( fig_entire_traj, "Entire Trajectory" ) 


## ============================================ ##
# split into training and testing 

N_train = Int( round( N * 0.8 ) ) 
N_train = 100 

# split into training and test data 
t_train, t_test, x_train, x_test, dx_train, dx_test, u_train, u_test = split_train_test_Npoints( t, x, dx_fd, u, N_train ) 

# get sizes 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x, u )

# plot training and testing data 
fig_train_test = plot_axes3d( ) 
fig_train_test = plot_train_test( x_train, x_test, N_train, fig_train_test ) 


## ============================================ ##
# try gp stuff 

# first - smooth measurements with Gaussian processes 

# normal SINDy 
λ = 0.1 
Ξ_lasso    = sindy_lasso( x_train, dx_train, λ, u_train ) 

# GPSINDy 
x_train_GP  = gp_post( t_train, 0*x_train, t_train, 0*x_train, x_train ) 
dx_train_GP = gp_post( x_train_GP, 0*dx_train, x_train_GP, 0*dx_train, dx_train ) 
Ξ_GP_lasso  = sindy_lasso( x_train_GP, dx_train_GP, λ, u_train ) 

# ----------------------- # 
# now test on training data 

x0_train_GP = x_train_GP[1,:] 

# build dx fn 
dx_fn_sindy     = build_dx_fn( poly_order, x_vars, u_vars, Ξ_lasso ) 
dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_GP_lasso ) 
x_train_sindy   = integrate_euler( dx_fn_sindy, x0_train_GP, t_train, u_train )  
x_train_gpsindy = integrate_euler( dx_fn_gpsindy, x0_train_GP, t_train, u_train )  

# create figure 
fig_train_pred  = plot_axes3d( ) 
fig_train_pred  = plot_train_pred( x_train, x_train_gpsindy, N_train, fig_train_pred ) 

## ============================================ ## 
# okay ... debug why the GP stuff is not working for q1, w1, w2, w3, and wdot3 

q1_train = x_train[:,7] 
q1_train_GP  = gp_post( t_train, 0*q1_train, t_train, 0*q1_train, q1_train ) 


## ============================================ ##



fig_x  = plot_quad_x_train_GP( t_train, x_train, x_train_GP ) 
fig_dx = plot_quad_dx_train_GP( t_train, dx_train, dx_train_GP ) 

fig_train_GP 
    # lines!(ax2, t_train, x_train_GP[:,1], x_train_GP[:,2], x_train_GP[:,3], linewidth = 1, linestyle = :solid, label = "GP") 


## ============================================ ## 
# cross-validation 

λ_vec = λ_vec_fn() 
Ξ_gpsindy_vec, err_x_sindy, err_dx_sindy = cross_validate_λ( t_train, x_train_GP, dx_train_GP, u_train, λ_vec ) 
Ξ_gpsindy_minerr = Ξ_minerr( Ξ_gpsindy_vec, err_x_sindy ) 


## ============================================ ##
# debug cross-validation 
   
# build function library from smoothed data 
Θx_gp  = pool_data_test( [ x_train_GP u_train ], n_vars, poly_order ) 

j = size(x_train_GP, 2) 
    
# CROSS-VALIDATION 
for i = eachindex(λ_vec) 

    λ = λ_vec[i] 
    println( "x", j, ": λ = ", λ ) 

    # GPSINDy-lasso ! 
    Ξ_gpsindy  = sindy_lasso( x_train_GP, dx_train_GP, λ, u_train ) 
    dx_gpsindy = Θx_gp * Ξ_gpsindy 

    # integrate discovered dynamics 
    dx_fn_gpsindy   = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy ) 
    x_gpsindy_train = integrate_euler( dx_fn_gpsindy, x0_train_GP, t_train, u_train ) 

    # save ξ coefficients 
    ξ = Ξ_gpsindy[:,j] 
    println( "ξ norm = ", norm( ξ ) ) 
    
    # push error stuff 
    err_xj_norm  = norm( x_gpsindy_train[:,j] - x_train_GP[:,j] )   ; println( "x err = ", err_xj_norm ) 
    # err_dxj_norm = norm( dx_gpsindy[:,j] - dx_train_GP[:,j] )       

end 







## ============================================ ##
# get states and inputs (OLD WAY) 

path = "test/data/cyrus_quadcopter_csvs/" 
# state vars: px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz 
csv_files_vec = readdir( path ) 

# for i_csv in eachindex(csv_files_vec) 
i_csv = 1 
    csv_file = string( path, csv_files_vec[i_csv] ) 
    df   = CSV.read(csv_file, DataFrame) 
    x    = Matrix(df) 
i_csv = 2 
    csv_file = string( path, csv_files_vec[i_csv] ) 
    df   = CSV.read(csv_file, DataFrame) 
    u    = Matrix(df) 
# end 

# all points 
N  = size(x, 1) 

# time vector ( probably dt = 0.01 s? )
dt = 0.01 
t  = collect( range(0, step = dt, length = N) ) 

# get derivatives 
x, dx_fd = unroll( t, x ) 

# ----------------------- #
# plot entire trajectory 

fig_entire_traj = plot_line3d( x[:,1], x[:,2], x[:,3] ) 
fig_entire_traj = add_title3d( fig_entire_traj, "Entire Trajectory" ) 






## ============================================ ##
# try sindy stls and lasso 

# try sindy 
λ = 0.1 

# println( "Ξ_stls time " ) 
# @btime Ξ_stls  = sindy_stls( x_train, dx_train, λ, u_train ) 

println( "Ξ_lasso time " ) 

start   = time() 
Ξ_lasso = @btime sindy_lasso( x_train, dx_train, λ, u_train ) 
elapsed = time() - start 
println( "btime elapsed = ", elapsed ) 

start   = time() 
Ξ_lasso = @time sindy_lasso( x_train, dx_train, λ, u_train ) 
elapsed = time() - start 
println( "time elapsed = ", elapsed ) 











## ============================================ ##
## ============================================ ##
# below is all jake's car data stuff 


## ============================================ ##
# single run (good) 

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

x_err_hist  = x_err_struct([], [], [], [])
for i = eachindex(csv_files_vec) 
# for i = [ 4, 5 ]
    # i = 4 
    csv_file = csv_files_vec[i] 
    t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 ) 

    push!( x_err_hist.sindy_lasso, norm( x_test_noise - x_test_sindy )  ) 
    push!( x_err_hist.gpsindy,     norm( x_test_noise - x_test_gpsindy )  ) 
end 

## ============================================ ## 
# single run (good) 


csv_file = "test/data/rollout_4_mod_u.csv"

Ξ_gpsindy = [] 
x_gpsindy = [] 
x_sindy   = [] 
t_train, t_test, x_train_noise, x_test_noise, Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 )

using Plots 
plot( x_test_noise[:,1], x_test_noise[:,2] ) 
plot!( x_test_sindy[:,1], x_test_sindy[:,2] ) 
plot!( x_test_gpsindy[:,1], x_test_gpsindy[:,2] ) 


## ============================================ ##
# save outputs as csv 
header = [ "t", "x1_test", "x2_test", "x3_test", "x4_test", "x1_sindy", "x2_sindy", "x3_sindy", "x4_sindy", "x1_gpsindy", "x2_gpsindy", "x3_gpsindy", "x4_gpsindy" ] 

data   = [ t_test x_test_noise x_test_sindy x_test_gpsindy ]
df     = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single", ".csv"), df, header=header)


## ============================================ ##
data_noise = [ t_test x_test_noise ] 
header     = [ "t", "x1_test", "x2_test", "x3_test", "x4_test" ]
data       = [ t_test x_test_noise ]  
df         = DataFrame( data,  :auto ) 
CSV.write(string("car_hw_single_test_data", ".csv"), df, header=header)

