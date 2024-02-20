using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using BenchmarkTools 
using GLMakie 


## ============================================ ##
# save data at 10 Hz (assuming that previous data is 100 Hz) 

function Fhz_data( x, u, F_hz ) 

    N = size(x, 1) 

    x_Fhz_mat = zeros(1,13) 
    u_Fhz_mat = zeros(1,4) 
    for i = 1 : Int(100 / F_hz) : N      # assuming the quadcopter data is already at 100 Hz - so we can just take every 100 / F_hz-th point 
        x_Fhz = x[i,:] 
        u_Fhz = u[i,:] 
        if i == 1 
            x_Fhz_mat = x_Fhz' 
            u_Fhz_mat = u_Fhz' 
        else 
            x_Fhz_mat = vcat( x_Fhz_mat, x_Fhz' ) 
            u_Fhz_mat = vcat( u_Fhz_mat, u_Fhz' ) 
        end 
    end
    
    N_Fhz = size(x_Fhz_mat, 1) 
    t_Fhz = collect( range(0, step = 1 / F_hz, length = N_Fhz) ) 

    return t_Fhz, x_Fhz_mat, u_Fhz_mat 
end 

t_Fhz, x_Fhz_mat, u_Fhz_mat = Fhz_data( x, u, F_hz ) 


## ============================================ ##
# get states and inputs 

function save_Fhz_data( F_hz ) 

    path = "test/data/cyrus_quadcopter_csvs/" 
    # state vars: px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz 
    csv_files_vec = readdir( path ) 
    
    for i_x_csv in 1 : 2 : length(csv_files_vec) 
    
        csv_file = string( path, csv_files_vec[i_x_csv] ) 
        df   = CSV.read(csv_file, DataFrame) 
        x    = Matrix(df) 
    
        i_u_csv = i_x_csv + 1 
        csv_file = string( path, csv_files_vec[i_u_csv] ) 
        df   = CSV.read(csv_file, DataFrame) 
        u    = Matrix(df) 

        # discretize data 
        t_Fhz, x_Fhz_mat, u_Fhz_mat = Fhz_data( x, u, F_hz ) 

        # save data 
        data = [ t_Fhz x_Fhz_mat u_Fhz_mat ] 

        header = [ 
            "t", 
            "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",   # states 
            "c1", "c2", "c3", "c4", "c5", "c6"                                              # commands  
            ] 

        data_df = DataFrame(data, :auto) 
        csv_file_str = replace( csv_files_vec[i_x_csv], ".csv" => "" ) 
        println( csv_file_str ) 
        str = string( "test/data/cyrus_quadcopter_csvs_sparse/", csv_file_str, "-", F_hz, "hz.csv" ) 
        CSV.write( str, data_df ) 
    
        # # plot entire trajectory 
        # fig_entire_traj = plot_line3d( x[:,1], x[:,2], x[:,3] ) 
        # fig_entire_traj = add_title3d( fig_entire_traj, "Entire Trajectory" ) 
    
    end 

end 

# ----------------------- #
# save data 

F_hz = 10   # 10 Hz 
save_Fhz_data( F_hz )


## ============================================ ##
# test saving 


fig_Fhz = plot_line3d( x_Fhz_mat[:,1], x_Fhz_mat[:,2], x_Fhz_mat[:,3] ) 
fig_Fhz = add_title3d( fig_Fhz, "Entire Trajectory (10 Hz)" ) 

# save data 
data = [ t_Fhz x_Fhz_mat u_Fhz_mat ] 

header = [ 
    "t", 
    "px", "py", "pz", "vx", "vy", "vz", "qw", "qx", "qy", "qz", "wx", "wy", "wz",   # states 
    "c1", "c2", "c3", "c4", "c5", "c6"                                              # commands  
    ] 

data_df = DataFrame(data, :auto) 
str = string( "test/data/cyrus_quadcopter_csvs_sparse/quadcopter_", F_hz, "hz.csv" ) 
CSV.write( str, data_df ) 









