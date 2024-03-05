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







