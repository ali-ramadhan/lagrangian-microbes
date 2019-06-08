using Distributed
@everywhere using DistributedArrays

using Printf
import PyPlot
using BenchmarkTools

using PyCall
@pyimport pickle

#= @benchmark results for random_points_distributed(10000, 70, R32)
 nprocs()  mean time
---------------------
  1         4.035 s
  5         1.764 s
 15           605 ms
 30           313 ms
 40           272 ms
 60           268 ms
=#

# Calculate distance in meters between two points (ϕ1, λ1) and (ϕ2, λ2)
# on the Earth's surface using the haversine formula. ϕ denotes the latitude
# while λ denotes the longitude.
# See: http://www.movable-type.co.uk/scripts/latlong.html
function haversine_distance(ϕ1, λ1, ϕ2, λ2, R)
    Δϕ = ϕ2 - ϕ1
    Δλ = λ2 - λ1
    a = sinpi(Δϕ / 360.0f0)^2 + cospi(ϕ1 / 180.0f0) * cospi(ϕ2 / 180.0f0) * sinpi(Δλ / 360.0f0)^2
    # c = 2.0f0 * atan(√a, √(1-a))
    c = 2.0f0 * asin(min(1.0f0, √a))
    R*c
end

@everywhere function haversine_distance32(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
    c1 = cospi(lat1 / 180.0f0)
    c2 = cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = sinpi(dlat / 360.0f0)
    d2 = sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * asin(min(1.0f0, sqrt(a)))
    return radius * c
end

function pairwise_distance_hist!(lats, lons, dist)
    N = length(lats)
    for i in 1:N-1, j in (i+1):N
        bin = round(Int8, 10.0f0 * log10(haversine_distance32(lats[i], lons[i], lats[j], lons[j], R32)))
        @inbounds dist[bin] += 1
    end
end

@everywhere function pairwise_distance_histogram_1point(i, lats, lons, bins, R)
    N = length(lats)
    sub_hist = zeros(Int, bins)
    
    for j in (i+1):N
        bin = round(Int8, 10.0f0 * log10(max(1, haversine_distance32(lats[i], lons[i], lats[j], lons[j], R))))
        @inbounds sub_hist[bin] += 1
    end
    
    return sub_hist
end

function pairwise_distance_histogram_parallel(lats, lons, bins, R)
    N = length(lats)
    
    hist = zeros(Int, bins)
    # sub_hists = zeros(Int, bins, N)
    
    @DArray [pairwise_distance_histogram_1point(i, lats, lons, bins, R) for i in 1:(N-1)];
    # sum(sub_hists, dims=2)
end

function random_points(N, bins)
    lats = 20 .+ 20 .* rand(Float32, N)
    lons = 20 .+ 20 .* rand(Float32, N)
    
    dist = zeros(bins)
    
    pairwise_distance_hist!(lats, lons, dist)
end

function random_points_distributed(N, bins, R)
    lats = 20 .+ 20 .* rand(Float32, N)
    lons = 20 .+ 20 .* rand(Float32, N)    
    pairwise_distance_histogram_parallel(lats, lons, bins, 6300.0f3)
end

function plot_pairwise_histogram(pickle_fpath, bins)
    R64, R32 = 6371.228e3, 6371.228f3 # average radius of the earth [m]

    ROCK, PAPER, SCISSOR = 1, 2, 3

    ROCK_COLOR = "red"
    PAPER_COLOR = "limegreen"
    SCISSOR_COLOR = "blue"

    f = open(pickle_fpath, "r")
    microbes = pickle.load(f)

    mlons, mlats, species = microbes[:, 1], microbes[:, 2], microbes[:, 3]

    mlons = convert(Array{Float32}, mlons)
    mlats = convert(Array{Float32}, mlats)
    species = convert(Array{Int8}, species)

    rock_mlons = mlons[species .== ROCK];
    rock_mlats = mlats[species .== ROCK];
    paper_mlons = mlons[species .== PAPER];
    paper_mlats = mlats[species .== PAPER];
    scissor_mlons = mlons[species .== SCISSOR];
    scissor_mlats = mlats[species .== SCISSOR];

    N = length(mlons)
    Nr = length(rock_mlons)
    Np = length(paper_mlons)
    Ns = length(scissor_mlons)

    @assert Nr + Np + Ns == N
    @show N, Nr, Np, Ns;

    # colors = Array{String,1}(undef, N); colors .= "";
    # colors[species .== ROCK] .= ROCK_COLOR;
    # colors[species .== PAPER] .= PAPER_COLOR;
    # colors[species .== SCISSOR] .= SCISSOR_COLOR;
    # @assert sum(colors .== "") == 0;

    @printf("Found %d workers.\n", nprocs())

    @printf("Computing rock distance histogram (Nr=%d)...", Nr)
    @time rock_distance_subhists = pairwise_distance_histogram_parallel(rock_mlats, rock_mlons, bins, R32)
    
    @printf("Computing paper distance histogram (Np=%d)...", Np)
    @time paper_distance_subhists = pairwise_distance_histogram_parallel(paper_mlats, paper_mlons, bins, R32)
    
    @printf("Computing scissor distance histogram (Ns=%d)...", Ns)
    @time scissor_distance_subhists = pairwise_distance_histogram_parallel(scissor_mlats, scissor_mlons, bins, R32)
    
    # @printf("Computing pairwise distance histogram (N=%d)...", N)
    # @time all_distance_histp = pairwise_distance_histogram_parallel(mlats, mlons, bins)

    rock_pdh = sum(rock_distance_subhists)
    paper_pdh = sum(paper_distance_subhists)
    scissor_pdh = sum(scissor_distance_subhists)

    max_initial_distance = haversine_distance32(25f0, -145f0, 35f0, -155f0, R32)
    interaction_distance = haversine_distance32(30f0, -150f0, 30f0, -150.01f0, R32)
    PyPlot.axvline(x=max_initial_distance, linestyle=":")
    PyPlot.axvline(x=interaction_distance, linestyle=":")

    length_m = 10 .^ (collect(0:bins-1) ./ 10)

    PyPlot.loglog(length_m, rock_pdh, label="Rocks")
    PyPlot.loglog(length_m, paper_pdh, label="Papers")
    PyPlot.loglog(length_m, scissor_pdh, label="Scissors")

    PyPlot.xlabel("Distance (m)");
    PyPlot.ylabel("Counts");
    PyPlot.xlim([1, 2e7])
    PyPlot.ylim([1, 4e9])
    
    PyPlot.legend()

    png_fpath = pickle_fpath * ".png"
    @printf("Saving pairwise distance histogram figure: %s\n", png_fpath)
    PyPlot.savefig(png_fpath, dpi=300, format="png", transparent=false)
    PyPlot.close("all")
end

# rps_fpath(p,h) = "/home/alir/nobackup/lagrangian_microbe_output/small_patch_490kp_p0.9_interactions/rps_microbe_species_p" * lpad(string(p), 4, "0") * "_h" * lpad(string(h), 3, "0") * ".pickle"
# rps_fpath(p,h) = "/home/alir/nobackup/lagrangian_microbe_output/small_patch_490kp_p0.55_interactions/rps_microbe_species_p" * lpad(string(p), 4, "0") * "_h" * lpad(string(h), 3, "0") * ".pickle"
rps_fpath(p,h) = "/home/alir/nobackup/lagrangian_microbe_output/small_patch_490kp_pRS0.51_interactions/rps_microbe_species_p" * lpad(string(p), 4, "0") * "_h" * lpad(string(h), 3, "0") * ".pickle"

function plot_multiple_pairwise_histogram()
    for p in 1:6
        plot_pairwise_histogram(rps_fpath(6*p, 0), 70)
    end
end
