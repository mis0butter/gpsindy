using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# control input delay  

path          = "test/data/jake_car_csvs_control_adjust/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    # i = 1 
    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 
    header   = names(df) 

    # get data 
    t = data[:,1] ; x = data[:,2:5] ; u = data[:,6:7] 

    # shift control input 
    u_shift = circshift(u, 6) 
    # u_shift[1:6,2] .= 0 

    # Let's save over the u data for the csv file 
    data_shift = copy(data) 
    data_shift[:,6:7] = u_shift 

    # delete first 6 rows of data_shift 
    data_shift = data_shift[7:end,:] 
    
    # save data_shift as csv file 
    csv_file_shift = replace( csv_file, "rollout_" => "rollout_shift_") 
    CSV.write( csv_file_shift, DataFrame(data_shift, header) ) 

end 


## ============================================ ##
# save data at certain rate 


function Fhz_data( t, x, u, F_hz_des, F_hz_OG = 50 ) 

    N = size(x, 1) 

    x_Fhz_mat = zeros(1,13) 
    u_Fhz_mat = zeros(1,4) 
    t_Fhz_mat = zeros(1) 
    for i = 1 : Int(F_hz_OG / F_hz_des) : N      # assuming the quadcopter data is already at 100 Hz - so we can just take every 100 / F_hz-th point 
        x_Fhz = x[i,:] 
        u_Fhz = u[i,:] 
        t_Fhz = t[i] 
        if i == 1 
            x_Fhz_mat = x_Fhz' 
            u_Fhz_mat = u_Fhz' 
            t_Fhz_mat = t_Fhz 
        else 
            x_Fhz_mat = vcat( x_Fhz_mat, x_Fhz' ) 
            u_Fhz_mat = vcat( u_Fhz_mat, u_Fhz' ) 
            t_Fhz_mat = vcat( t_Fhz_mat, t_Fhz ) 
        end 
    end

    return t_Fhz_mat, x_Fhz_mat, u_Fhz_mat 
end 




## ============================================ ##
# control input delay  

path          = "test/data/jake_car_csvs_control_adjust/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

# for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    i = 1 
    csv_file = csv_files_vec[i] 
    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 
    header   = names(df) 

    # get data 
    t = data[:,1] ; x = data[:,2:5] ; u = data[:,6:7] 

    # save data at 10 Hz (assuming that previous data is 50 Hz) 
    F_hz = 1 
    t_Fhz, x_Fhz_mat, u_Fhz_mat = Fhz_data( t, x, u, F_hz ) 

    # save sparse data 
    data_sparse = [ t_Fhz x_Fhz_mat u_Fhz_mat ]
    
    # save data_shift as csv file 
    csv_file_sparse = replace( csv_file, "rollout_shift_" => string("rollout_shift_", F_hz, "hz_") )  
    CSV.write( csv_file_sparse, DataFrame(data_sparse, header) ) 

# end 







