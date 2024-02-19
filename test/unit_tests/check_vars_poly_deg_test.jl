using GaussianSINDy 

## ============================================ ##
# function to check the number of variables in the polynomial expansion 

ind = 0 
n_vars = 5 
poly_order = 4 

    # poly order 4 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                for l = k  : n_vars 
                    ind += 1            
                end 
            end 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 

## ============================================ ##

ind = 0 
n_vars = 10 
poly_order = 3 

    # poly order 3 
    for i = 1 : n_vars 
        for j = i : n_vars 
            for k = j : n_vars 
                ind += 1 
            end 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 

## ============================================ ##

ind = 0 
n_vars = 19  
poly_order = 2 

    # poly order 2 
    for i = 1 : n_vars 
        for j = i : n_vars 
            ind += 1 
        end 
    end 

fn_check = check_vars_poly_deg( n_vars, poly_order )   

println( "fn_check = ", fn_check ) 
println( "ind = ", ind ) 

