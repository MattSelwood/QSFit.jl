function voigt(x, σ, γ)
    faddeeva(z) = erfcx(-im * z)
    z = (x + im * γ) / σ / sqrt(2)
    return real(faddeeva(z)) / (σ * sqrt(2pi))
end

function voigt_fwhm(σ, γ)
    # J.J.Olivero and R.L. Longbothum in Empirical fits to the Voigt line width: A brief review, JQSRT 17, P233, 1977
    # http://snst-hu.lzu.edu.cn/zhangyi/ndata/Voigt_profile.html
    fg = σ * 2.355
    fl = 2 * γ
    0.5346 * fl + sqrt(0.2166 * fl^2 + fg^2)
    # fl * (0.5346 + sqrt(0.2166 + (fg/fl)^2))
end

#=
@gp "unset grid" xr=[-10,10]
x = -100:0.01:100
σ = 1.53;  γ = 0. ;  vp = voigt.(x, σ, γ);  @gp :- x vp "w l"; println(int_tabulated(x, vp)); print(voigt_fwhm(σ, γ) - QSFit.estimate_fwhm(x, vp))
σ = 1.30;  γ = 0.5;  vp = voigt.(x, σ, γ);  @gp :- x vp "w l"; println(int_tabulated(x, vp)); print(voigt_fwhm(σ, γ) - QSFit.estimate_fwhm(x, vp))
σ = 0.01;  γ = 1.8;  vp = voigt.(x, σ, γ);  @gp :- x vp "w l"; println(int_tabulated(x, vp)); print(voigt_fwhm(σ, γ) - QSFit.estimate_fwhm(x, vp))
σ = 1.  ;  γ = 1. ;  vp = voigt.(x, σ, γ);  @gp :- x vp "w l"; println(int_tabulated(x, vp)); print(voigt_fwhm(σ, γ) - QSFit.estimate_fwhm(x, vp))
σ = 1.  ;  γ = 5. ;  vp = voigt.(x, σ, γ);  @gp :- x vp "w l"; println(int_tabulated(x, vp)); print(voigt_fwhm(σ, γ) - QSFit.estimate_fwhm(x, vp))
=#

# ____________________________________________________________________
# SpecLineVoigt
#

mutable struct SpecLineVoigt <: AbstractComponent
    norm::Parameter
    center::Parameter
    fwhm::Parameter
    gamma::Parameter
    voff::Parameter
    norm_integrated::Bool

    function SpecLineVoigt(center::Number)
        out = new(Parameter(1),
                  Parameter(center),
                  Parameter(3000),
                  Parameter(3000),
                  Parameter(0),
                  true)

        @assert center > 0
        out.norm.low = 0
        out.center.low = 0
        out.fwhm.low = 0
        out.gamma.low = 0
        out.voff.low = 0
        out.center.fixed = true
        return out
    end
end

function prepare!(comp::SpecLineVoigt, domain::Domain{1})
    return fill(NaN, length(domain))
end

function evaluate!(buffer, comp::SpecLineVoigt, x::Domain{1},
                   norm, center, fwhm, gamma, voff)

    x0 = center - (voff / 3.e5) * center
    σ = fwhm  / 3.e5 * center / 2.355
    γ = gamma / 3.e5 * center / 2.
    buffer .= norm .* voigt.(x .- x0, σ, γ)
    comp.norm_integrated  ||  (buffer .*= (σ * sqrt(2pi)))
end

#=
    x = Domain(500:1:1500.)
    comp = QSFit.SpecLineVoigt(1000.)
    comp.fwhm.val  = 3e4
    comp.gamma.val = 3e4
    ceval = GFit.CompEval(comp, x)
    evaluate!(ceval)
    @gp x[:] ceval.buffer ./ maximum(ceval.buffer) "w l"
=#
