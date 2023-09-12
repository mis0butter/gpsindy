using GaussianSINDy 


## ============================================ ##

path          = "test/data/jake_car_csvs/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end

# for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
    i = 4 
    csv_file = csv_files_vec[i] 
    Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 )
# end 


## ============================================ ## 
# setup 

# load data 
# csv_file = "test/data/jake_robot_data.csv" 
csv_file = "test/data/jake_car_csvs/rollout_1.csv" 


Ξ_sindy_stls, x_train_sindy, x_test_sindy, Ξ_gpsindy_minerr, x_train_gpsindy, x_test_gpsindy = cross_validate_gpsindy( csv_file, 1 )



