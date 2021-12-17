# ____________________________________________________________________
# SpecLineVoigt
#

include("../utils.jl")

mutable struct SpecLineVoigt <: AbstractComponent
    norm_g::Parameter
    norm_l::Parameter
    center::Parameter
    fwhm_g::Parameter
    fwhm_l::Parameter
    voff::Parameter
    spec_res_kms::Float64
    index::Vector{Int}  # Lorentzian optimization
    norm_integrated::Bool

    function SpecLineVoigt(center::Number)
        out = new(Parameter(1),
                  Parameter(1),
                  Parameter(center),
                  Parameter(3000),
                  Parameter(3000),
                  Parameter(0),
                  0., 
                  Vector{Int}(),
                  true)

        @assert center > 0
        out.norm_g.low = 0
        out.norm_l.low = 0
        out.center.low = 0
        out.fwhm_g.low = 0
        out.fwhm_l.low = 0
        out.voff.low = 0
        out.center.fixed = true
        return out
    end
end

function prepare!(comp::SpecLineVoigt, domain::Domain{1})
    return fill(NaN, length(domain))
end

function evaluate!(buffer, comp::SpecLineVoigt, x::Domain{1},
                   norm_g, norm_l, center, fwhm_g, fwhm_l, voff)

    x0 = center - (voff / 3.e5) * center
    lorentz = buffer
    gauss = buffer

    # Gaussian 
    sigma_line = fwhm_g / 2.355      / 3.e5 * center
    sigma_spec = comp.spec_res_kms / 3.e5 * center
    sigma = sqrt(sigma_line^2 + sigma_spec^2)
    map!(x -> begin
        (abs.(x - x0) .> 4sigma)  &&  (return 0.)
        ret = norm_g * exp(-((x - x0) / sigma)^2 / 2)
        comp.norm_integrated  &&  (ret /= (sqrt(2pi) * sigma))
        return ret
        end, gauss, x.coords[:,1])

    # Lorentzian
    lorentz[comp.index] .= 0.
    empty!(comp.index)
    hwhm = fwhm_l / 3.e5 * center / 2  # Note: this is in `center` units
    X = (x .- x0) ./ hwhm
    i = findall(abs.(X) .< 20) # optimization
    append!(comp.index, i)
    lorentz[i] .= norm_l ./ (1 .+ X[i].^2.)
    if comp.norm_integrated
        lorentz[i] ./= pi * hwhm
    end

    # Convolve Gauss and Lorentz we have constructed
    buffer = convol(gauss, lorentz)

end

#=
    x = Domain(500:1:1500.)
    comp = QSFit.SpecLineVoigt(1000.)
    comp.fwhm_g.val = 3e4
    comp.fwhm_l.val = 3e4
    ceval = GFit.CompEval(comp, x)
    GFit.evaluate_cached(ceval)
    @gp x ceval.buffer ./ maximum(ceval.buffer) "w l"
=#

