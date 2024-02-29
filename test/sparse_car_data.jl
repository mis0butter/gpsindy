using GaussianSINDy 
using LinearAlgebra 
using CSV, DataFrames 
using GLMakie 


## ============================================ ##
# control input delay  

path          = "test/data/jake_car_csvs_sparse/" 
csv_files_vec = readdir( path ) 
for i in eachindex(csv_files_vec)  
    csv_files_vec[i] = string( path, csv_files_vec[i] ) 
end 

# for i = eachindex(csv_files_vec) 
# for i = [ 4 ]
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
    ddx = fdiff(t, dx, 1) 
    du  = fdiff(t, u, 1) 
    ddu = fdiff(t, du, 1) 

    # smooth so that it's not so noisy ... 
    
    x_GP  = gp_post( t, 0*x, t, 0*x, x ) 
    dx_GP = gp_post( x, 0*dx, x, 0*dx, dx ) 


## ============================================ ##
# moving average 

using EasyFit 

# dx_ave = movavg(dx, 10) 

N_window = 2 

dx1 = movavg(dx[:,1], N_window) 
dx2 = movavg(dx[:,2], N_window) 
dx3 = movavg(dx[:,3], N_window) 
dx4 = movavg(dx[:,4], N_window) 

dx_ave = [ dx1.x dx2.x dx3.x dx4.x ] 

du1 = movavg(du[:,1], N_window) 
du2 = movavg(du[:,2], N_window) 

du_ave = [ du1.x du2.x ] 

f = Figure() 
    Axis( f[1,1] ) 
        lines!( f[1,1], t, dx[:,4], label = "dx" ) 
        lines!( f[1,1], t, dx4.x, label = "dx_ave" ) 
        axislegend() 

f 


## ============================================ ##
# I don't like moving average ... try lowpass 

using DSP 

dx_low = Lowpass( dx ) 
dx_low = dx_low.w 

fs = 50 
# t = 0:1/fs:1
# comment out random noise for testing
# x = @. sin(2*pi*50*t) + 2*sin(2*pi*250*t)# + randn() / 10

responsetype = Bandpass( 1, 20; fs = 2000 )
designmethod = Butterworth(4)
dx_filt = filt(digitalfilter(responsetype, designmethod), dx)

f = Figure() 
    Axis( f[1,1] ) 
        lines!( f[1,1], t, dx[:,4] )
        lines!( f[1,1], t, dx_filt[:,4] )  

f 

## ============================================ ##
# make fig - manually find spikes in control input delay ... 

i0 = 740  
i1 = 770   

t1 = 14.98 
t2 = 14.98 + 0.12 

fig = Figure() 
    Axis( fig[1,1], xlabel = "t" )
        # lines!( fig[1,1], t, du[:,1], label = "du1" )  
        lines!( fig[1,1], t[i0:i1], u[i0:i1,2], label = "u2" ) 
        lines!( fig[1,1], t[i0:i1], x[i0:i1,4], label = "x4" )  
        lines!( fig[1,1], t[i0:i1], x_GP[i0:i1,4], label = "x4_GP" )  
        vlines!( fig[1,1], [t1, t2], color = :red)
        axislegend() 
    Axis( fig[2,1], xlabel = "t" )
        # lines!( fig[1,1], t, du[:,1], label = "du1" )  
        lines!( fig[2,1], t[i0:i1], du[i0:i1,2], label = "du2" ) 
        lines!( fig[2,1], t[i0:i1], du_ave[i0:i1,2], label = "du2_ave" ) 
        lines!( fig[2,1], t[i0:i1], dx[i0:i1,4], label = "dx4" )  
        lines!( fig[2,1], t[i0:i1], dx_ave[i0:i1,4], label = "dx4_ave" )  
        vlines!( fig[2,1], [t1, t2], color = :red)
        axislegend() 
    Axis( fig[1,2], xlabel = "t" )
        lines!( fig[1,2], t[i0:i1], ddu[i0:i1,2], label = "ddu2" ) 
        vlines!( fig[1,2], [t1, t2], color = :red)
        axislegend() 
    Axis( fig[2,2], xlabel = "t" ) 
        lines!( fig[2,2], t[i0:i1], ddx[i0:i1,4], label = "ddx4" )  
        vlines!( fig[2,2], [t1, t2], color = :red)
        axislegend() 

fig 
# end 

# looks like there is control input delay of 9.54 , 9.66 maybe 
# delay of 0.12 seconds? 

