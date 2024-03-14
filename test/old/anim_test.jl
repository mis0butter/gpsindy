using Plots 


## ============================================ ## 

x = 0 : 0.01 : 2*pi 
y = sin.(x) 

## ============================================ ##
# plot 

a = Animation() 

x_min = floor(minimum( minimum(x) )) 
x_max = ceil(maximum( maximum(x) )) 

y_min = floor(minimum( minimum(y) )) 
y_max = ceil(maximum( maximum(y) )) 

for i = 1 : size(x, 1)

    p = plot( 
        title = "Animation", 
        xlim  = ( x_min, x_max ) , 
        ylim  = ( y_min, y_max ) ,  
        axis  = ( [], false ) , 
     ) 

    plot!( p, x[1:i], y[1:i] )
     
    frame(a, p)

end 

g = gif(a, fps = 20.0)
display(g)  



