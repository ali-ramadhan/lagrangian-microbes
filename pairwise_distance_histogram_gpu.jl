function haversine_gpu(lat1::Float32, lon1::Float32, lat2::Float32, lon2::Float32, radius::Float32)
    c1 = CUDAnative.cospi(lat1 / 180.0f0)
    c2 = CUDAnative.cospi(lat2 / 180.0f0)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    d1 = CUDAnative.sinpi(dlat / 360.0f0)
    d2 = CUDAnative.sinpi(dlon / 360.0f0)
    t = d2 * d2 * c1 * c2
    a = d1 * d1 + t
    c = 2.0f0 * CUDAnative.asin(CUDAnative.min(1.0f0, CUDAnative.sqrt(a)))
    return radius * c
end

function pairwise_dist_kernel(lat::CuDeviceVector{Float32}, lon::CuDeviceVector{Float32},
                              dist_hist::CuDeviceArray{Float32}, n)
    i = (blockIdx().x-1) * blockDim().x + threadIdx().x
    j = (blockIdx().y-1) * blockDim().y + threadIdx().y

    if i <= n && j <= n
        # store to shared memory
        shmem = @cuDynamicSharedMem(Float32, 2*blockDim().x + 2*blockDim().y)
        if threadIdx().y == 1
            shmem[threadIdx().x] = lat[i]
            shmem[blockDim().x + threadIdx().x] = lon[i]
        end
        if threadIdx().x == 1
            shmem[2*blockDim().x + threadIdx().y] = lat[j]
            shmem[2*blockDim().x + blockDim().y + threadIdx().y] = lon[j]
        end
        sync_threads()

        # load from shared memory
        lat_i = shmem[threadIdx().x]
        lon_i = shmem[blockDim().x + threadIdx().x]
        lat_j = shmem[2*blockDim().x + threadIdx().y]
        lon_j = shmem[2*blockDim().x + blockDim().y + threadIdx().y]

        bin = CUDAnative.floor(10.0f0 * CUDAnative.log10(haversine_gpu(lat_i, lon_i, lat_j, lon_j, 6371.228f3)))
        @inbounds dist_hist[threadIdx().x, threadIdx().y] += 1
    end

    return
end

function pairwise_dist_gpu(lat::Vector{Float32}, lon::Vector{Float32})
    # upload
    lat_gpu = CuArrays.CuArray(lat)
    lon_gpu = CuArrays.CuArray(lon)

    n = length(lat)

    # calculate launch configuration
    # NOTE: we want our launch configuration to be as square as possible,
    #       because that minimizes shared memory usage
    dev = device()
    total_threads = min(n, attribute(dev, CUDAdrv.MAX_THREADS_PER_BLOCK))
    threads_x = floor(Int, sqrt(total_threads))
    threads_y = total_threads ÷ threads_x
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, n ./ threads)

    # calculate size of dynamic shared memory
    shmem = 2 * sum(threads) * sizeof(Float32)

    # allocate
    bins = 100
    dist_hist_gpu = CuArrays.CuArray{Float32}(undef, (threads_x, threads_y))

    @show total_threads
    @show threads
    @show blocks
    @show shmem

    @cuda blocks=blocks threads=threads shmem=shmem pairwise_dist_kernel(lat_gpu, lon_gpu, dist_hist_gpu, n)

    return Array(dist_hist_gpu)
end


function main(n = 10000)
    lat = rand(Float32, n) .* 45
    lon = rand(Float32, n) .* -120

    pairwise_dist_gpu(lat, lon)
end
