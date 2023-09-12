## structs 

struct Hist 
    objval 
    fval 
    gval 
    hp 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

struct hist_lasso_struct  
    objval 
    r_norm 
    s_norm 
    eps_pri 
    eps_dual 
end 

struct Ξ_struct 
    truth 
    sindy_stls 
    sindy_lasso 
    nn 
    gpsindy 
end

struct Ξ_err_struct 
    sindy_stls 
    sindy_lasso 
    nn 
    gpsindy 
end 

struct x_struct 
    truth 
    sindy_stls 
    sindy_lasso 
    nn 
    gpsindy 
end

struct x_err_struct 
    sindy_stls 
    sindy_lasso 
    nn 
    gpsindy 
end 

struct data_struct
    t 
    u 
    x_true 
    dx_true 
    x_noise 
    dx_noise 
end 

export Hist, hist_lasso_struct, Ξ_struct, Ξ_err_struct, data_struct, x_struct, x_err_struct

