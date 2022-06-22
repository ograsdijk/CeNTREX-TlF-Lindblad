using Distributed
@everywhere using Waveforms

@everywhere begin
    """
        gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64

    Compute the 2D gaussian at point x,y for an amplitude a, mean value μx and μy,
    and a standard deviation σx and σy
    """
    function gaussian_2d(x::Float64, y::Float64, a::Float64, μx::Float64, μy::Float64, σx::Float64, σy::Float64)::Float64
        a.*exp(.- ((x.-μx).^2 ./ (2 .* σx.*σx) + (y.-μy).^2 ./ (2 .* σy.*σy)))
    end

    """
        phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64

    Compute phase modulation at frequency ω with a modudulation strength β at time t
    """
    function phase_modulation(t::Float64, β::Float64, ω::Float64)::ComplexF64
        return exp(1im.*β.*sin(ω.*t))
    end

    """
        square_wave(t::Float64, ω::Float64, phase::Float64)

    generate a square wave from 0 to 1 at frequency ω [2π Hz; rad/s] and phase offset ϕ [rad]
    """
    function square_wave(t::Float64, ω::Float64, phase::Float64)::Float64
        0.5.*(1 .+ squarewave(ω.*t .+ phase))
    end

    """
        sawtooth_wave(t::Float64, ω::Float64, phase::Float64)::Float64

    generate a sawtooth wave from 0 to 1 at frequency ω [2π Hz; rad/s] and phase offset phase [rad]
    """
    function sawtooth_wave(t::Float64, ω::Float64, phase::Float64)::Float64
        0.5.*(1 .+ sawtoothwave(ω.*t .+ phase))
    end
    """
        multipass_2d_intensity(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64

    generate a multipass with 2D gaussian intensity profiles for each pass
    """
    function multipass_2d_intensity(x::Float64, y::Float64, amplitudes::Vector{Float64}, xlocs::Vector{Float64}, ylocs::Vector{Float64}, σx::Float64, σy::Float64)::Float64
        intensity::Float64 = 0.0
        for i = 1:length(amplitudes::Vector{Float64})
            @inbounds intensity += gaussian_2d(x,y,amplitudes[i],xlocs[i],ylocs[i], σx,σy)
        end
        return intensity
    end
end