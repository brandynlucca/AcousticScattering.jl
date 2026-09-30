# Experimental, unexported tank instrument helpers. No stable API commitment.
# Keep out of the exported/public API while the physical model is under development.

"""
    linear_chirp(f_start, f_stop, duration; amplitude=1, phase=0,
        start_time=0, window=:rectangular, rtol=1e-9)

Spectrum of a real linear-FM voltage pulse. Frequencies are Hz, duration/start time
seconds, phase radians and amplitude peak volts. Local-time phase is
`2pi*(f_start*t + (f_stop-f_start)*t^2/(2duration)) + phase`, so instantaneous
frequency runs from `f_start` to `f_stop` without doubling the sweep slope.
Both real-signal sidebands are retained, with the package's positive-sign forward
Fourier transform. The window is `:rectangular` or `:hann`. Numerical quadrature
is split into intervals resolving the fastest integrand oscillation.
"""
function linear_chirp(f_start::Real, f_stop::Real, duration::Real;
        amplitude::Real=1, phase::Real=0, start_time::Real=0,
        window::Symbol=:rectangular, rtol::Real=1e-9)
    all(x->isfinite(x) && x>0, (f_start,f_stop,duration,rtol)) ||
        throw(ArgumentError("frequencies, duration and rtol must be positive and finite"))
    all(isfinite,(amplitude,phase,start_time)) || throw(ArgumentError("pulse parameters must be finite"))
    window in (:rectangular,:hann) || throw(ArgumentError("unknown pulse window"))
    slope=(f_stop-f_start)/duration
    return f->begin
        isfinite(f) || throw(ArgumentError("frequency must be finite"))
        count=max(1,ceil(Int,(abs(f)+max(f_start,f_stop))*duration/4))
        nodes=range(0.,duration;length=count+1)
        value=quadgk(nodes...;rtol,atol=rtol*abs(amplitude)*duration) do t
            envelope=window===:hann ? (1-cospi(2t/duration))/2 : 1.
            amplitude*envelope*cos(2pi*(f_start*t+slope*t^2/2)+phase)*cis(2pi*f*t)
        end
        cis(2pi*f*start_time)*first(value)
    end
end

"""
    resonant_response(frequency, decay_time; gain=1, delay=0)

Causal second-order bandpass response, unity at the resonance before real `gain`
and nonnegative `delay` [s]. `decay_time` [s] is the pole amplitude-decay constant
in the underdamped regime, with `Q=pi*frequency*decay_time`. Zero decay time
disables the resonator. Separate instances can represent Tx and Rx ring-down.
This lumped response does not identify any particular transducer from its name.
"""
function resonant_response(frequency::Real, decay_time::Real; gain::Real=1, delay::Real=0)
    isfinite(frequency) && frequency>0 || throw(ArgumentError("frequency must be positive and finite"))
    all(x->isfinite(x) && x>=0,(decay_time,delay)) && isfinite(gain) ||
        throw(ArgumentError("decay/delay must be nonnegative and gain finite"))
    return f->begin
        isfinite(f) || throw(ArgumentError("frequency must be finite"))
        omega=2pi*f
        h=iszero(decay_time) ? 1.0 + 0im :
            (-2im*omega/decay_time)/((2pi*frequency)^2-omega^2-2im*omega/decay_time)
        gain*cis(omega*delay)*h
    end
end

"""
    rc_response(cutoff; kind=:lowpass, gain=1, delay=0)

Causal single-pole RC low-pass or high-pass voltage response. Cutoff is Hz;
delay is nonnegative seconds. Gain is real and may include polarity inversion.
Cascade callables to represent multiple poles. Parameters are explicit circuit
approximations, not manufacturer-specific measured transfer functions.
"""
function rc_response(cutoff::Real; kind::Symbol=:lowpass, gain::Real=1, delay::Real=0)
    isfinite(cutoff) && cutoff>0 || throw(ArgumentError("cutoff must be positive and finite"))
    kind in (:lowpass,:highpass) || throw(ArgumentError("kind must be :lowpass or :highpass"))
    isfinite(gain) && isfinite(delay) && delay>=0 || throw(ArgumentError("invalid gain or delay"))
    return f->begin
        isfinite(f) || throw(ArgumentError("frequency must be finite"))
        s=-im*f/cutoff
        gain*cis(2pi*f*delay)*(kind===:lowpass ? 1/(1+s) : s/(1+s))
    end
end

"""
    piezo_equivalent(capacitance, resistance, inductance, motional_capacitance)

Butterworth-Van Dyke one-mode equivalent: static capacitance [F] in parallel
with a series motional R [ohm], L [H], C [F]. Returns callable `impedance(f)`
[ohm] and `motion(f)` (motional current per terminal voltage, normalized to
unity at series resonance), plus `frequency` and `decay_time=2L/R`.
Static C can be zero; other parameters must be positive. At DC impedance is
infinite and motion is zero. Acoustic loading must already be represented in
the supplied effective motional parameters. Absolute velocity/voltage and
receive sensitivity require separate scale factors; this is not full piezo FEM.
"""
function piezo_equivalent(capacitance::Real,resistance::Real,inductance::Real,motional_capacitance::Real)
    isfinite(capacitance) && capacitance>=0 &&
        all(x->isfinite(x) && x>0,(resistance,inductance,motional_capacitance)) ||
        throw(ArgumentError("invalid equivalent-circuit elements"))
    branch=f->begin
        isfinite(f) || throw(ArgumentError("frequency must be finite"))
        iszero(f) && return ComplexF64(Inf)
        s=-2pi*im*f
        resistance+s*inductance+inv(s*motional_capacitance)
    end
    impedance=f->iszero(f) ? ComplexF64(Inf) : inv(-2pi*im*f*capacitance+inv(branch(f)))
    motion=f->iszero(f) ? 0.0 + 0im : resistance/branch(f)
    (;impedance,motion,frequency=inv(2pi*sqrt(inductance*motional_capacitance)),
        decay_time=2inductance/resistance)
end

"""
    transformer_response(turns_ratio; source_resistance=0, load_impedance)

Loaded ideal-transformer voltage ratio from an open-circuit amplifier output to
the secondary load. `turns_ratio=N_secondary/N_primary`, resistance/impedance in
ohm. Load is a scalar or frequency callable; positive infinite impedance means
open circuit. The ratio is `n*Zload/(Zload+n^2*Rsource)`. Transformer leakage,
magnetizing loss, cable delay and amplifier power limits are not implicit.
"""
function transformer_response(turns_ratio::Real; source_resistance::Real=0,load_impedance)
    isfinite(turns_ratio) && turns_ratio>0 && isfinite(source_resistance) && source_resistance>=0 ||
        throw(ArgumentError("invalid turns ratio or source resistance"))
    return f->begin
        isfinite(f) || throw(ArgumentError("frequency must be finite"))
        z=load_impedance isa Number ? load_impedance : load_impedance(f)
        z==Inf && return ComplexF64(turns_ratio)
        isfinite(z) && !iszero(z+turns_ratio^2*source_resistance) ||
            throw(ArgumentError("invalid or singular load impedance"))
        turns_ratio*z/(z+turns_ratio^2*source_resistance)
    end
end

"""
    signal_chain_response(frequencies; amplifier=1, matching=1, transmit=1,
        receive=1, electronics=1, feedthrough=0)

Sample a linear chain, using scalar, vector or callable responses. `amplifier`
and `matching` are V/V; `transmit` converts loaded terminal V to piston reference
pressure p0 [Pa]; `receive` converts aperture-averaged Pa to receiver-input V;
`electronics` is receiver V/V. Return `acoustic` [V/p0 times p0/V] and
`feedthrough` [V/V] responses per generator volt.

The electrical feedthrough branch is tapped at the amplifier output, bypasses
matching and both acoustic transducers, and enters before receiver electronics:
`acoustic=A*M*T*R*E`, `feedthrough=A*F*E`. Feedthrough is zero by default.
This explicit branch must not be multiplied by acoustic propagation or Tx/Rx
ring-down. All stages are linear; gains must use the stated pressure convention.
"""
function signal_chain_response(frequencies;amplifier=1,matching=1,transmit=1,
        receive=1,electronics=1,feedthrough=0)
    all(f->isfinite(f) && f>0,frequencies) || throw(ArgumentError("frequencies must be positive and finite"))
    a,m,t,r,e,c=map(x->_response_values(x,frequencies),
        (amplifier,matching,transmit,receive,electronics,feedthrough))
    acoustic=a.*m.*t.*r.*e
    prompt=a.*c.*e
    all(isfinite,acoustic) && all(isfinite,prompt) || throw(ArgumentError("chain gain overflow"))
    (;acoustic,feedthrough=prompt)
end

"""
    receiver_oscillogram(frequencies, paths::NamedTuple, voltage_pulse;
        chain=(;), voltage_limit=Inf, kwargs...)

Fully synthetic receiver voltage from acoustic path responses `p_received/p0`,
a generator-voltage spectrum [V s], and keywords for `signal_chain_response`.
Adds an explicit electrical `:prompt` component; that name is reserved. All
other keywords go to `oscillogram` (not additional instrument-response keywords).
Returns its fields plus `linear_signal` and `clipped`. Optional symmetric ADC
clipping is applied after coherent summation. `envelope` and `components` remain
those of the pre-clipping linear signal. Clipping cannot model amplifier slew,
power limiting, hysteresis or overload recovery. No measurements are read.
"""
function receiver_oscillogram(frequencies,paths::NamedTuple,voltage_pulse;
        chain::NamedTuple=(;),voltage_limit::Real=Inf,kwargs...)
    :prompt in keys(paths) && throw(ArgumentError("prompt is reserved for electrical feedthrough"))
    voltage_limit>0 && !isnan(voltage_limit) || throw(ArgumentError("voltage_limit must be positive"))
    any(k->k in (:transmit_response,:receive_response,:system_response),keys(kwargs)) &&
        throw(ArgumentError("put instrument responses in chain, not synthesis keywords"))
    h=signal_chain_response(frequencies;chain...)
    responses=map(path->_response_values(path,frequencies).*h.acoustic,paths)
    trace=oscillogram(frequencies,merge(responses,(prompt=h.feedthrough,)),voltage_pulse;kwargs...)
    linear_signal=trace.signal
    signal=clamp.(linear_signal,-voltage_limit,voltage_limit)
    merge(trace,(;signal,linear_signal,clipped=abs.(linear_signal).>voltage_limit))
end
