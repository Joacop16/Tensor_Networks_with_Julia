using ITensors
using CUDA

#monitoring multiples gpu functions:

function memory_info_all_gpus(print_info = true)
    
    percentages = []

    scale = 1/(1024^3) #converty bytes to GB
    for (i, dev) in enumerate(CUDA.NVML.devices())

        name = CUDA.NVML.name(dev) 
        mem_info = CUDA.NVML.memory_info(dev)
        total = round(mem_info.total*scale, sigdigits=4)
        used = round(mem_info.used*scale, sigdigits=4)
        free = round(mem_info.free*scale, sigdigits=4)
        percentage= round(used*100/total, sigdigits=4)
        
        print_info ? println("$name #$i memory usage: $percentage % ($used GB/ $total GB)" ) : nothing
        
        append!(percentages, percentage)
    end
    
    return percentages
end

function clean_all_gpus(Deep_cleaning = false)
    for i=reverse(0:length(CUDA.devices()) - 1)
        global current_gpu = i
        CUDA.device!(current_gpu)
        Deep_cleaning ? GC.gc(true) : nothing #This could be very slow.
        CUDA.reclaim()
    end
end

#DMRG functions

function Create_H_MPO(t,U, N, sites = []) 

    if length(sites) == 0
        sites = siteinds("Electron",N)    
    end
    
    os = OpSum() 
    for j=1:N-1 
        os += -t,"Cdagup",j,"Cup",j+1 
        os += -t,"Cdagup",j+1,"Cup",j
        os += -t,"Cdagdn",j,"Cdn",j+1 
        os += -t,"Cdagdn",j+1,"Cdn",j
    end 

    for j=1:N
        os += U,"Nup * Ndn",j
    end
    # Convert these terms to an MPO 
    H = MPO(os,sites)
    return H, sites
end

#Custom observer to measure the use of GPU:

mutable struct DemoObserver <: AbstractObserver
    energy_tol::Float64
    last_energy::Float64

    DemoObserver(energy_tol=0.0) = new(energy_tol,1000.0)
end

function ITensors.checkdone!(o::DemoObserver;kwargs...)
    
    CUDA.reclaim()
    memory_info_all_gpus() #Print GPU percentage of use. 
        
    sw = kwargs[:sweep]
    energy = kwargs[:energy]
    if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
    println("Stopping DMRG after sweep $sw")
    return true
    end
    # Otherwise, update last_energy and keep going
    o.last_energy = energy
    return false
end

#First Run
N = 4
t = 1
U = 1

nsweeps = 100
maxdim = [1500] #maxdim - integer or array of integers specifying the maximum size allowed for the bond dimension or rank of the MPS being optimized
cutoff = [1E-10] #maxdim - integer or array of integers specifying the maximum size allowed for the bond dimension or rank of the MPS being optimized
# DMRG_observer = DMRGObserver(;energy_tol=10e-8, minsweeps=10, energy_type=Float64)
DMRG_observer = DemoObserver(10e-8)

H, sites = Create_H_MPO(t,U, N) 
Initial_Guess = randomMPS(sites);

H = NDTensors.cu(H)
Initial_Guess = NDTensors.cu(Initial_Guess);

@time energy_ground_state, psi_ground_state = dmrg(H,Initial_Guess; nsweeps, maxdim, cutoff, observer = DMRG_observer, outputlevel = 0) 

clean_all_gpus(true)
memory_info_all_gpus()

#Real Run
N = 140
t = 1
U = 1

nsweeps = 100
maxdim = [1500] #maxdim - integer or array of integers specifying the maximum size allowed for the bond dimension or rank of the MPS being optimized
cutoff = [1E-10] #maxdim - integer or array of integers specifying the maximum size allowed for the bond dimension or rank of the MPS being optimized
# DMRG_observer = DMRGObserver(;energy_tol=10e-8, minsweeps=10, energy_type=Float64)
DMRG_observer = DemoObserver(10e-8)

H, sites = Create_H_MPO(t,U, N) 
Initial_Guess = randomMPS(sites);

H = NDTensors.cu(H)
Initial_Guess = NDTensors.cu(Initial_Guess);

@time energy_ground_state, psi_ground_state = dmrg(H,Initial_Guess; nsweeps, maxdim, cutoff, observer = DMRG_observer, outputlevel = 1) 
