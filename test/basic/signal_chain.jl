using AcousticScattering, Test
using AcousticScattering: tone_burst, gaussian_pulse, time_synthesis
using AcousticScattering: linear_chirp, resonant_response, rc_response,
    piezo_equivalent, transformer_response, signal_chain_response, receiver_oscillogram
using QuadGK: quadgk

@testset "Synthetic excitation and causal responses" begin
    f0=120000.;duration=125e-6
    chirp=linear_chirp(f0,f0,duration;phase=.37,start_time=13e-6,amplitude=2)
    tone=tone_burst(f0,duration;phase=.37,start_time=13e-6)
    for f in (-130000.,100000.,119000.,120000.,140000.)
        @test chirp(f)≈2tone(f) atol=1e-12
    end
    pulse=linear_chirp(119000.,121000.,duration;window=:hann)
    for f in (110000.,120000.,130000.)
        value=quadgk(t->(1-cospi(2t/duration))/2*
            cos(2pi*(119000t+2000t^2/(2duration)))*cis(2pi*f*t),
            range(0,duration;length=129)...;atol=1e-14)[1]
        @test pulse(f)≈value atol=1e-12
        @test pulse(-f)≈conj(pulse(f)) atol=1e-12
    end
    tau=15e-6;beta=1/tau;wd=sqrt((2pi*f0)^2-beta^2)
    resonator=resonant_response(f0,tau)
    @test resonator(f0)≈1
    for f in (100000.,120000.,140000.)
        impulse=t->2beta*exp(-beta*t)*(cos(wd*t)-beta/wd*sin(wd*t))
        transformed=quadgk(t->impulse(t)*cis(2pi*f*t),
            range(0,30tau;length=201)...;atol=1e-10)[1]
        @test resonator(f)≈transformed atol=1e-10
        @test resonator(-f)≈conj(resonator(f))
    end
    @test resonant_response(f0,0;gain=2,delay=1e-6)(f0)≈2cis(2pi*f0*1e-6)
    low=rc_response(10000.);high=rc_response(10000.;kind=:highpass)
    @test abs(low(10000.))≈inv(sqrt(2))
    @test high(0)==0
    @test low(0)==1
    @test low(12345.)+high(12345.)≈1
    @test imag(low(10000.))>0 # positive phase represents positive time delay
    @test_throws ArgumentError linear_chirp(0.,f0,duration)
    @test_throws ArgumentError resonant_response(f0,-tau)
    @test_throws ArgumentError rc_response(0.)
end

@testset "Loaded matching and motional resonance" begin
    r=100.;l=.001;c=1/((2pi*120000)^2*l)
    piezo=piezo_equivalent(2e-9,r,l,c)
    @test piezo.frequency≈120000
    @test piezo.decay_time≈2l/r
    @test piezo.motion(120000.)≈1
    @test piezo.motion(0)==0
    @test piezo.impedance(0)==Inf
    for f in (50000.,120000.,180000.)
        @test real(piezo.impedance(f))>0
        @test piezo.impedance(-f)≈conj(piezo.impedance(f))
        @test piezo.motion(f)≈resonant_response(piezo.frequency,piezo.decay_time)(f)
    end
    @test transformer_response(2;source_resistance=50,load_impedance=200)(120000.)≈1
    @test transformer_response(2;source_resistance=50,load_impedance=Inf)(120000.)==2
    @test transformer_response(2;source_resistance=50,load_impedance=0)(120000.)==0
    @test_throws ArgumentError piezo_equivalent(-1.,r,l,c)
end

@testset "Receiver branch topology, delays and clipping" begin
    f=collect(100.:100.:10000.)
    chain=(amplifier=2,matching=3,transmit=5,receive=7,electronics=11,feedthrough=.1)
    h=signal_chain_response(f;chain...)
    @test all(==(2310),h.acoustic)
    @test all(==(2.2),h.feedthrough)
    pulse=gaussian_pulse(5000.,1000.)
    delay=.002
    path=cis.(2pi.*f.*delay)
    result=receiver_oscillogram(f,(echo=path,),pulse;chain,nfft=4096,start_time=-.001)
    reference=time_synthesis(f,path,pulse,result.times).*2310+
        time_synthesis(f,ones(length(f)),pulse,result.times).*2.2
    @test result.signal≈reference rtol=1e-11
    @test result.signal==result.linear_signal
    @test !any(result.clipped)
    @test result.times[argmax(abs.(result.components.echo))]≈delay atol=result.timestep
    @test result.times[argmax(abs.(result.components.prompt))]≈0 atol=result.timestep
    changed=receiver_oscillogram(f,(echo=path,),pulse;chain=merge(chain,(transmit=50,)),nfft=4096,start_time=-.001)
    @test changed.components.prompt==result.components.prompt
    @test changed.components.echo≈10result.components.echo
    clipped=receiver_oscillogram(f,(echo=path,),pulse;chain,nfft=4096,start_time=-.001,voltage_limit=1)
    @test maximum(abs,clipped.signal)<=1
    @test any(clipped.clipped)
    @test clipped.linear_signal==result.signal
    @test clipped.envelope==result.envelope
    @test_throws ArgumentError receiver_oscillogram(f,(prompt=path,),pulse)
    @test_throws ArgumentError receiver_oscillogram(f,(echo=path,),pulse;transmit_response=2)
end
