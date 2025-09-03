
"""
    MultiEpoch(corealg::ADIAlgorithm, epochsizes::Vector{Int}; weights=nothing, method=mean, options...)

A wrapper to combine multiple epochs of data using any of the algorithms in HCI.jl. Each epoch is processed independently using the specified `corealg`, and the results are then combined using the specified `method`.

This wrapper is of type `ADIAlgorithm`, so it can be used wherever an ADI algorithm is expected, such as `contrast_curve` or any of the other metrics.

When using `contrast_curve` with `MultiEpoch`, you can't specify a template PSF for each epoch. A workaround is using the averaged PSF of all epochs. This isn't ideal, but should be a decent approximation for epochs with similar PSFs.

Use the `corealg` algorithm to process multiple epochs of data, each of size given in `epochsizes`. `epochsizes` should be an array of integers corresponding to the number of frames in each epoch. You can also weight each epoch using the `weights` argument, which should be an array of the same length as `epochsizes`.

"""

@concrete struct MultiEpoch <: ADIAlgorithm
    corealg::ADIAlgorithm
    epochsizes 
    weights
    method
    opts
end

MultiEpoch(corealg, epochsizes, weights, method, options...) = MultiEpoch(corealg, epochsizes, weights, method, options)
MultiEpoch(corealg, epochsizes; weights=nothing, method=mean, options...) = MultiEpoch(corealg, epochsizes, weights, method, options)

@inline function chunk_array(sizes, cube)
    dims = ndims(cube)
    ends = cumsum(sizes)
    if dims == 1
        return [@view cube[(i == 1 ? 1 : ends[i-1] + 1):ends[i]] for i in eachindex(ends)]
    elseif dims == 2
        return [@view cube[:, (i == 1 ? 1 : ends[i-1] + 1):ends[i]] for i in eachindex(ends)]
    elseif dims == 3
        return [@view cube[:,:, (i == 1 ? 1 : ends[i-1] + 1):ends[i]] for i in eachindex(ends)]
    else
        error("Unsupported number of dimensions: $dims")
    end
end

function process(alg::MultiEpoch, cube::AbstractArray{T,3}, angles; kwargs...) where T
    nepochs = length(alg.epochsizes)

    cube_chunks = chunk_array(alg.epochsizes, cube)
    angle_chunks = chunk_array(alg.epochsizes, angles)

    res = zeros(size(cube, 1), size(cube, 2), nepochs)
    for i in 1:nepochs
        res[:,:,i] = alg.corealg(cube_chunks[i], angle_chunks[i])
    end

    # weighted average
    if !isnothing(alg.weights)
        weights = alg.weights
        weights = weights ./ sum(weights)
        for i in 1:nepochs
            res[:,:,i] = res[:,:,i] * weights[i]
        end
        return sum(res; dims=3)[:,:,1]
    else
        return alg.method(res; dims=3)[:,:,1]
    end

end