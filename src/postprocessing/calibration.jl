"""
    TransferFunction(frequencies, values; delay=0)

Complex system response sampled at strictly increasing positive frequencies [Hz].
Interpolation is linear in complex amplitude after removing the explicitly supplied
delay [s]; evaluation restores `exp(2pi*im*f*delay)`. Thus positive delay moves the
synthesized echo to later time. No phase unwrapping or extrapolation is performed.
The delay must be known or estimated separately; it is not inferred from wrapped phase.
Pass the result as `system_response` to [`time_synthesis`](@ref). Avoid multiplying
transmit/receive responses again if they are already included in this response.
"""
struct TransferFunction
    frequencies::Vector{Float64}
    values::Vector{ComplexF64}
    delay::Float64
    function TransferFunction(frequencies, values; delay::Real = 0.0)
        length(frequencies)==length(values)>=2 ||
            throw(ArgumentError("a transfer function needs at least two matching samples"))
        f=Float64.(collect(frequencies))
        all(x->isfinite(x)&&x>0, f) && all(>(0), diff(f)) ||
            throw(ArgumentError("frequencies must be finite, positive and strictly increasing"))
        all(isfinite, values) && isfinite(delay) ||
            throw(ArgumentError("response and delay must be finite"))
        return new(f, ComplexF64.(values), Float64(delay))
    end
end

function (response::TransferFunction)(f::Real)
    isfinite(f) && first(response.frequencies)<=f<=last(response.frequencies) ||
        throw(ArgumentError("frequency lies outside the calibrated band"))
    i=searchsortedlast(response.frequencies, f)
    f==response.frequencies[i] && return response.values[i]
    f0, f1=response.frequencies[i:(i + 1)]
    a=(f-f0)/(f1-f0)
    y0=response.values[i]*cis(-2pi*f0*response.delay)
    y1=response.values[i + 1]*cis(-2pi*f1*response.delay)
    return ((1-a)*y0+a*y1)*cis(2pi*f*response.delay)
end

"""
    calibrate_response(frequencies, measured, reference; weights=nothing,
        delay=0, reference_floor=0)

Fit the complex multiplicative response `measured = H .* reference` independently
at each frequency [Hz]. Inputs are vectors, or matrices with repeated calibrations
in columns. A reference vector is shared across columns. Optional nonnegative
weights have the same shape as `measured` (a vector supplies one weight per frequency).
Fit by weighted complex least squares, with no phase conjugation of `H`.
Return `(; response::TransferFunction, residual, reference_rms)`; `residual` is the
weighted relative residual at each frequency. Reject unresolved bins whose weighted
reference RMS is at most `reference_floor`, including exact reference nulls.

The measured and modeled reference must describe the same observable and phase
origin. For a sphere use complex `modal(Sphere(...), SolidElastic(...), k)` amplitudes
only after normalizing the measurements to the same spreading/beam convention;
otherwise model the complete transmitter/sphere/receiver signal. Applying the fitted
response to another target requires the same instrument configuration and reference
plane. This fits the instrument response, not an intrinsic target strength correction.
`delay` specifies the known system delay used for interpolation, not a delay subtraction.
"""
function calibrate_response(frequencies, measured::AbstractArray{<:Number},
        reference::AbstractArray{<:Number}; weights = nothing, delay::Real = 0.0,
        reference_floor::Real = 0.0)
    ndims(measured) in (1, 2) && ndims(reference) in (1, 2) ||
        throw(ArgumentError("calibration samples must be vectors or matrices"))
    n=length(frequencies)
    n>=2 && all(f->isfinite(f)&&f>0, frequencies) && all(>(0), diff(frequencies)) ||
        throw(ArgumentError("calibration needs at least two finite, positive, increasing frequencies"))
    size(measured, 1)==size(reference, 1)==n ||
        throw(ArgumentError("calibration samples must match frequencies"))
    size(measured, 2)>0 && size(reference, 2) in (1, size(measured, 2)) ||
        throw(ArgumentError("reference must have one column or match measured columns"))
    isfinite(reference_floor) && reference_floor>=0 ||
        throw(ArgumentError("reference_floor must be finite and nonnegative"))
    all(isfinite, measured) && all(isfinite, reference) ||
        throw(ArgumentError("calibration samples must be finite"))
    data=reshape(measured, n, :)
    model=reshape(reference, n, :)
    w=if weights===nothing
        ones(size(data))
    else
        ndims(weights) in (1, 2) && size(weights, 1)==n &&
        size(weights, 2) in (1, size(data, 2)) ||
            throw(ArgumentError("weights must match frequencies and calibration columns"))
        reshape(weights, n, :) .* ones(size(data))
    end
    all(x->x isa Real && isfinite(x)&&x>=0, w) ||
        throw(ArgumentError("calibration weights must be finite and nonnegative"))
    gains=zeros(ComplexF64, n)
    residual, reference_rms=zeros(n), zeros(n)
    for i in 1:n
        total_weight=sum(view(w, i, :))
        total_weight>0 || throw(ArgumentError("calibration bin $i has no positive weights"))
        predicted=view(model, i, :)
        denominator=sum(w[i, j]*abs2(predicted[mod1(j, length(predicted))])
        for j in axes(data, 2))
        reference_rms[i]=sqrt(denominator/total_weight)
        reference_rms[i]>reference_floor ||
            throw(ArgumentError("calibration reference is unresolved at frequency $(frequencies[i])"))
        gains[i]=sum(w[i, j]*conj(predicted[mod1(j, length(predicted))])*data[i, j]
        for j in axes(data, 2))/denominator
        error=sum(w[i, j]*abs2(data[i, j]-gains[i]*predicted[mod1(j, length(predicted))])
        for j in axes(data, 2))
        scale=sum(w[i, j]*abs2(data[i, j]) for j in axes(data, 2))
        residual[i]=iszero(scale) ? sqrt(error) : sqrt(error/scale)
    end
    return (;
        response = TransferFunction(frequencies, gains; delay), residual, reference_rms)
end

function time_synthesis(sweep::FrequencySweep, pulse, args...; kwargs...)
    sweep.amplitudes isa AbstractVector || throw(ArgumentError(
        "sweep must retain one complex response; select a component column explicitly otherwise"))
    return time_synthesis(sweep.frequencies, sweep.amplitudes, pulse, args...; kwargs...)
end
