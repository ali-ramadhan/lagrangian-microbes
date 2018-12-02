using Distributed
@everywhere using DistributedArrays

import PyPlot

using BenchmarkTools

using PyCall
@pyimport pickle


R64, R32 = 6371.228e3, 6371.228f3 # average radius of the earth [m]

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

@everywhere function pairwise_distance_histogram_1point(i, lats, lons, bins)
    N = length(lats)
    sub_hist = zeros(Int, bins)
    
    for j in (i+1):N
        bin = round(Int8, 10.0f0 * log10(haversine_distance32(lats[i], lons[i], lats[j], lons[j], R32)))
        @inbounds sub_hist[bin] += 1
    end
    
    return sub_hist
end

function pairwise_distance_histogram_parallel(lats, lons, bins)
    N = length(lats)
    
    hist = zeros(Int, bins)
    # sub_hists = zeros(Int, bins, N)
    
    @DArray [pairwise_distance_histogram_1point(i, lats, lons, bins) for i in 1:(N-1)];
    # sum(sub_hists, dims=2)
end

function random_points(N, bins)
    lats = 20 .+ 20 .* rand(N)
    lons = 20 .+ 20 .* rand(N)
    
    dist = zeros(bins)
    
    pairwise_distance_hist!(lats, lons, dist)
end