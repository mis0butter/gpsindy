using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# control input delay  

F_hz      = 25 
csv_path  = "test/data/jake_car_csvs_control_adjust/" 

save_path = replace( csv_path, "_adjust" => string( "_adjust_", F_hz, "hz" ) ) 
if !isdir( save_path ) 
    mkdir( save_path ) 
end 

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

for i = eachindex(csv_files_vec) 
    # i = 1 
    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 
    header   = names(df) 

    # get data 
    t = data[:,1] ; x = data[:,2:5] ; u = data[:,6:7] 

    # save data at 10 Hz (assuming that previous data is 50 Hz) 
    t_Fhz, x_Fhz_mat, u_Fhz_mat = Fhz_data( t, x, u, F_hz ) 

    # save sparse data 
    data_sparse = [ t_Fhz x_Fhz_mat u_Fhz_mat ]
    
    # save data_shift as csv file 
    csv_file_save = string( save_path, replace( csv_file, csv_path => "" ) ) 
    csv_file_save = replace( csv_file_save, "rollout_shift_" => string("rollout_shift_", F_hz, "hz_") )  

    CSV.write( csv_file_save, DataFrame(data_sparse, header) ) 

end 

























## ============================================ ##
# make better plot 

path          = "test/data/jake_car_csvs_control_adjust_5hz/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

i = 4 
csv_file = csv_files_vec[i] 
df       = CSV.read(csv_file, DataFrame) 
data     = Matrix(df) 

# get data 
t = data[:,1] 
x = data[:,2:5] 
u = data[:,6:7] 

# get derivative data 
x, dx = unroll( t, x ) 

# truth coeffs 
x_vars, u_vars, poly_order, n_vars = size_x_n_vars( x, u ) 

λ = 0.1 

x_GP  = smooth_gp_posterior( t, 0 * x, t, 0 * x, x )
dx_GP = smooth_gp_posterior( x, 0 * dx, x, 0 * dx, dx )

# learn coefficients 
Ξ_sindy   = sindy_lasso( x, dx, λ, u )
Ξ_gpsindy = sindy_lasso( x_GP, dx_GP, λ, u )

# build dx fns 
dx_fn_sindy    = build_dx_fn( poly_order, x_vars, u_vars, Ξ_sindy )
dx_fn_gpsindy  = build_dx_fn( poly_order, x_vars, u_vars, Ξ_gpsindy )

# generate predicts 
x0        = x[1,:] 
x_sindy   = integrate_euler( dx_fn_sindy, x0, t, u ) 
x_gpsindy = integrate_euler( dx_fn_gpsindy, x0, t, u ) 


## ============================================ ##

fig = plot_noise_GP_sindy_gpsindy( t, x, x_GP, x_sindy, x_gpsindy, "all data" ) 




