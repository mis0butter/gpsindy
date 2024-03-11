using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using CairoMakie 


## ============================================ ##
# control input delay  

# add how much noise to all data? 
freq_hz = 50 

csv_path  = string( "test/data/jake_car_csvs_ctrlshift/50hz/" ) 
save_path = string( "test/data/jake_car_csvs_ctrlshift_no_trans/50hz/" )  

if !isdir( save_path ) 
    mkdir( save_path ) 
end


## ============================================ ##

csv_files_vec = readdir( csv_path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( csv_path, csv_files_vec[i] ) 
end 

for i in eachindex(csv_files_vec) 

    csv_file = csv_files_vec[i] 

    df       = CSV.read(csv_file, DataFrame) 
    data     = Matrix(df) 
    
    # remove all rows where first row is < 1 
    data_notrans = data[ data[:,1] .> 1, : ] 
    
    # remove all rows from dataframe where first row is < 1 
    df_notrans = df[ df[:,1] .> 1, : ] 
    
    # save to new csv 
    csv_file_notrans = string( save_path, "rollout_", i, ".csv" ) 
    CSV.write( csv_file_notrans, df_notrans ) 
        
end 










