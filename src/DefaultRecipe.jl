abstract type DefaultRecipe <: AbstractRecipe end

function default_options(::Type{T}) where T <: DefaultRecipe
    out = OrderedDict{Symbol, Any}()
    out[:wavelength_range] = [1215, 7.3e3]
    out[:min_spectral_coverage] = Dict(:default => 0.6)
    out[:skip_lines] = [:OIII_5007_bw] #, Ha_base
    out[:host_template] = "Ell5"
    out[:use_host_template] = true
    out[:use_balmer] = true
    out[:use_ironuv] = true
    out[:use_ironopt] = true
    out[:n_unk] = 10
    out[:unk_avoid] = [4863 .+ [-1,1] .* 50, 6565 .+ [-1,1] .* 150]
    return out
end


function qso_cont_component(::Type{T}) where T <: DefaultRecipe
    comp = QSFit.powerlaw(3000)
    comp.alpha.val  = -1.5
    comp.alpha.low  = -3
    comp.alpha.high =  1
    return comp
end


line_breakdown(::Type{T}, name::Symbol, line::CombinedLine) where T <: DefaultRecipe =
    [(Symbol(:br_, name), ComboBroadLine( line.λ)),
     (Symbol(:na_, name), ComboNarrowLine(line.λ))]

line_group_name(::Type{T}, name::Symbol, line::BroadBaseLine)   where T <: DefaultRecipe = :BroadBaseLines
line_group_name(::Type{T}, name::Symbol, line::BroadLine)       where T <: DefaultRecipe = :BroadLines
line_group_name(::Type{T}, name::Symbol, line::NarrowLine)      where T <: DefaultRecipe = :NarrowLines
line_group_name(::Type{T}, name::Symbol, line::ComboBroadLine)  where T <: DefaultRecipe = line_group_name(T, name, BroadLine(line.λ))
line_group_name(::Type{T}, name::Symbol, line::ComboNarrowLine) where T <: DefaultRecipe = line_group_name(T, name, NarrowLine(line.λ))
line_group_name(::Type{T}, name::Symbol, line::UnkLine)         where T <: DefaultRecipe = :UnknownLines


function line_component(::Type{T}, line::BroadBaseLine) where T <: DefaultRecipe
    comp = SpecLineGauss(line.λ)
    comp.fwhm.val  = 2e4
    comp.fwhm.low  = 1e4
    comp.fwhm.high = 3e4
    comp.voff.fixed = true
    return comp
end

function line_component(::Type{T}, line::BroadLine) where T <: DefaultRecipe
    comp = SpecLineGauss(line.λ)
    comp.fwhm.val  = 5e3
    comp.fwhm.low  = 900
    comp.fwhm.high = 1.5e4
    comp.voff.low  = -3e3
    comp.voff.high =  3e3
    return comp
end

line_component(::Type{T}, line::ComboBroadLine) where T <: DefaultRecipe =
    line_component(T, BroadLine(line.λ))

function line_component(::Type{T}, line::NarrowLine) where T <: DefaultRecipe
    comp = SpecLineGauss(line.λ)
    comp.fwhm.val  = 5e2
    comp.fwhm.low  = 100
    comp.fwhm.high = 2e3
    comp.voff.low  = -1e3
    comp.voff.high =  1e3
    return comp
end

function line_component(::Type{T}, line::ComboNarrowLine) where T <: DefaultRecipe
    comp = line_component(T, NarrowLine(line.λ))
    comp.fwhm.high = 1e3
    return comp
end

function line_component(::Type{T}, line::UnkLine) where T <: DefaultRecipe
    comp = SpecLineGauss(line.λ)
    comp.norm.val = 0.
    comp.center.fixed = false
    comp.center.low = 0
    comp.center.high = Inf
    comp.fwhm.val  = 5e3
    comp.fwhm.low  = 600
    comp.fwhm.high = 1e4
    comp.voff.fixed = true
    return comp
end


function known_spectral_lines(::Type{T}) where T <: DefaultRecipe
    list = OrderedDict{Symbol, AbstractSpectralLine}()
    list[:Lyb         ] = CombinedLine( 1026.0  )
    list[:Lya         ] = CombinedLine( 1215.24 )
    list[:NV_1241     ] = NarrowLine(   1240.81 )
    list[:OI_1306     ] = BroadLine(    1305.53 )
    list[:CII_1335    ] = BroadLine(    1335.31 )
    list[:SiIV_1400   ] = BroadLine(    1399.8  )
    list[:CIV_1549    ] = CombinedLine( 1549.48 )
    list[:HeII        ] = BroadLine(    1640.4  )
    list[:OIII        ] = BroadLine(    1665.85 )
    list[:AlIII       ] = BroadLine(    1857.4  )
    list[:CIII_1909   ] = BroadLine(    1908.734)
    list[:CII         ] = BroadLine(    2326.0  )
    list[:F2420       ] = BroadLine(    2420.0  )
    list[:MgII_2798   ] = CombinedLine( 2799.117)
    list[:NeVN        ] = NarrowLine(   3346.79 )
    list[:NeVI_3426   ] = NarrowLine(   3426.85 )
    list[:OII_3727    ] = NarrowLine(   3729.875)
    list[:NeIII_3869  ] = NarrowLine(   3869.81 )
    list[:Hd          ] = BroadLine(    4102.89 )
    list[:Hg          ] = BroadLine(    4341.68 )
    list[:OIII_4363   ] = NarrowLine(   4363.00 )  # TODO: Check wavelength is correct
    list[:HeII        ] = BroadLine(    4686.   )
    list[:Hb          ] = CombinedLine( 4862.68 )
    list[:OIII_4959   ] = NarrowLine(   4960.295)
    list[:OIII_5007   ] = NarrowLine(   5008.240)
    list[:OIII_5007_bw] = NarrowLine(   5008.240)
    list[:HeI_5876    ] = BroadLine(    5877.30 )
    list[:OI_6300     ] = NarrowLine(   6300.00 )  # TODO: Check wavelength is correct
    list[:OI_6364     ] = NarrowLine(   6364.00 )  # TODO: Check wavelength is correct
    list[:NII_6549    ] = NarrowLine(   6549.86 )
    list[:Ha          ] = CombinedLine( 6564.61 )
    list[:Ha_base     ] = BroadBaseLine(6564.61 )
    list[:NII_6583    ] = NarrowLine(   6585.27 )
    list[:SII_6716    ] = NarrowLine(   6718.29 )
    list[:SII_6731    ] = NarrowLine(   6732.67 )
    return list
end


function fit(source::QSO{TRecipe}; id=1) where TRecipe <: DefaultRecipe
    elapsed = time()
    mzer = GFit.cmpfit()
    mzer.config.ftol = mzer.config.gtol = mzer.config.xtol = 1.e-6

    # Initialize components and guess initial values
    println(source.log, "\nFit continuum components...")
    λ = source.domain[id][1]
    model = Model(source.domain[id], :Continuum => Reducer(sum, [:qso_cont]),
                  :qso_cont => QSFit.qso_cont_component(TRecipe))
    c = model[:qso_cont]
    c.norm.val = interpol(source.data[id].val, λ, c.x0.val)

    # Host galaxy template
    if source.options[:use_host_template]
        add!(model, :Continuum => Reducer(sum, [:qso_cont, :galaxy]),
             :galaxy => QSFit.hostgalaxy(source.options[:host_template]))
        model[:galaxy].norm.val = interpol(source.data[id].val, λ, 5500)
    end

    # Balmer continuum and pseudo-continuum
    if source.options[:use_balmer]
        tmp = [:qso_cont, :balmer]
        source.options[:use_host_template]  &&  push!(tmp, :galaxy)
        add!(model, :Continuum => Reducer(sum, tmp),
             :balmer => QSFit.balmercont(0.1, 0.5))
        c = model[:balmer]
        c.norm.val  = 0.1
        c.norm.fixed = false
        c.norm.high = 0.5
        c.ratio.val = 0.5
        c.ratio.fixed = false
        c.ratio.low  = 0.3
        c.ratio.high = 1
        patch!(model) do m
            m[:balmer].norm *= m[:qso_cont].norm
        end
    end

    bestfit = fit!(model, source.data, minimizer=mzer);  show(source.log, bestfit)

    # QSO continuum renormalization
    freeze(model, :qso_cont)
    c = model[:qso_cont]
    println(source.log, "Cont. norm. (before): ", c.norm.val)
    while true
        residuals = (model() - source.data[id].val) ./ source.data[id].unc
        (count(residuals .< 0) / length(residuals) > 0.9)  &&  break
        c.norm.val *= 0.99
        evaluate(model)
    end
    println(source.log, "Cont. norm. (after) : ", c.norm.val)

    freeze(model, :qso_cont)
    source.options[:use_host_template]  &&  freeze(model, :galaxy)
    source.options[:use_balmer]         &&  freeze(model, :balmer)
    evaluate(model)

    # Fit iron templates
    println(source.log, "\nFit iron templates...")
    iron_components = Vector{Symbol}()
    if source.options[:use_ironuv]
        add!(model, :ironuv => QSFit.ironuv(3000))
        model[:ironuv].norm.val = 0.5
        push!(iron_components, :ironuv)
    end
    if source.options[:use_ironopt]
        add!(model,
             :ironoptbr => QSFit.ironopt_broad(3000),
             :ironoptna => QSFit.ironopt_narrow(500))
        model[:ironoptbr].norm.val = 0.5
        model[:ironoptna].norm.val = 0.0
        freeze(model, :ironoptna)  # will be freed during last run
        push!(iron_components, :ironoptbr, :ironoptna)
    end
    if length(iron_components) > 0
        add!(model, :Iron => Reducer(sum, iron_components))
        add!(model, :main => Reducer(sum, [:Continuum, :Iron]))
        evaluate(model)
        bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)
    else
        add!(model, :Iron => Reducer(() -> [0.], Symbol[]))
        add!(model, :main => Reducer(sum, [:Continuum, :Iron]))
    end
    source.options[:use_ironuv]   &&  freeze(model, :ironuv)
    source.options[:use_ironopt]  &&  freeze(model, :ironoptbr)
    source.options[:use_ironopt]  &&  freeze(model, :ironoptna)
    evaluate(model)

    # Add emission lines
    line_names = collect(keys(source.line_names[id]))
    line_groups = unique(collect(values(source.line_names[id])))
    println(source.log, "\nFit known emission lines...")
    add!(model, source.line_comps[id])
    for (group, lnames) in invert_dictionary(source.line_names[id])
        add!(model, group  => Reducer(sum, lnames))
    end
    add!(model, :main => Reducer(sum, [:Continuum, :Iron, line_groups...]))

    if haskey(model, :MgII_2798)
        model[:MgII_2798].voff.low  = -1000
        model[:MgII_2798].voff.high =  1000
    end
    if haskey(model, :OIII_5007_bw)
        model[:OIII_5007_bw].fwhm.val  = 500
        model[:OIII_5007_bw].fwhm.low  = 1e2
        model[:OIII_5007_bw].fwhm.high = 1e3
        model[:OIII_5007_bw].voff.low  = 0
        model[:OIII_5007_bw].voff.high = 2e3
    end

    # Guess values
    evaluate(model)
    y = source.data[id].val - model()
    for cname in line_names
        c = model[cname]
        yatline = interpol(y, λ, c.center.val)
        c.norm.val = 1.
        c.norm.val = abs(yatline) / QSFit.maxvalue(model[cname])
    end

    # Patch parameters
    if  haskey(model, :OIII_4959)  &&
        haskey(model, :OIII_5007)
        model[:OIII_4959].voff.fixed = true
        patch!(model) do m
            m[:OIII_4959].voff = m[:OIII_5007].voff
        end
    end
    if  haskey(model, :NII_6549)  &&
        haskey(model, :NII_6583)
        model[:NII_6549].voff.fixed = true
        patch!(model) do m
            m[:NII_6549].voff = m[:NII_6583].voff
        end
    end
    if  haskey(model, :OIII_5007_bw)  &&
        haskey(model, :OIII_5007)
        patch!(model) do m
            m[:OIII_5007_bw].voff += m[:OIII_5007].voff
            m[:OIII_5007_bw].fwhm += m[:OIII_5007].fwhm
        end
    end

    #=
    model[:br_Hb].voff.fixed = 1
    model[:br_Hb].fwhm.fixed = 1
    patch!(model) do m
        m[:br_Hb].voff = m[:br_Ha].voff
        m[:br_Hb].fwhm = m[:br_Ha].fwhm
    end
    =#

    bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)
    for lname in line_names
        freeze(model, lname)
    end

    # Add unknown lines
    println(source.log, "\nFit unknown emission lines...")
    if source.options[:n_unk] > 0
        tmp = OrderedDict{Symbol, AbstractComponent}()
        for j in 1:source.options[:n_unk]
            tmp[Symbol(:unk, j)] = line_component(TRecipe, UnkLine(5e3))
        end
        add!(model, :UnkLines => Reducer(sum, collect(keys(tmp))), tmp)
        add!(model, :main => Reducer(sum, [:Continuum, :Iron, line_groups..., :UnkLines]))
        evaluate(model)
        for j in 1:source.options[:n_unk]
            freeze(model, Symbol(:unk, j))
        end
        evaluate(model)

        # Set "unknown" line center wavelength where there is a maximum in
        # the fit residuals, and re-run a fit.
        λunk = Vector{Float64}()
        while true
            (length(λunk) >= source.options[:n_unk])  &&  break
            evaluate(model)
            Δ = (source.data[id].val - model()) ./ source.data[id].unc

            # Avoid considering again the same region (within 1A)
            for l in λunk
                Δ[findall(abs.(l .- λ) .< 1)] .= 0.
            end

            # Avoidance regions
            for rr in source.options[:unk_avoid]
                Δ[findall(rr[1] .< λ .< rr[2])] .= 0.
            end

            # Do not add lines close to from the edges since these may
            # affect qso_cont fitting
            Δ[findall((λ .< minimum(λ)*1.02)  .|
                      (λ .> maximum(λ)*0.98))] .= 0.
            iadd = argmax(Δ)
            (Δ[iadd] <= 0)  &&  break  # No residual is greater than 0, skip further residuals....
            push!(λunk, λ[iadd])

            cname = Symbol(:unk, length(λunk))
            model[cname].norm.val = 1.
            model[cname].center.val  = λ[iadd]
            model[cname].center.low  = λ[iadd] - λ[iadd]/10. # allow to shift 10%
            model[cname].center.high = λ[iadd] + λ[iadd]/10.

            thaw(model, cname)
            bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)
            freeze(model, cname)
        end
    end
    evaluate(model)

    # Last run with all parameters free
    println(source.log, "\nLast run with all parameters free...")
    thaw(model, :qso_cont)
    source.options[:use_host_template]  &&  thaw(model, :galaxy)
    source.options[:use_balmer]         &&  thaw(model, :balmer)
    source.options[:use_ironuv]         &&  thaw(model, :ironuv)
    source.options[:use_ironopt]        &&  thaw(model, :ironoptbr)
    source.options[:use_ironopt]        &&  thaw(model, :ironoptna)

    for lname in line_names
        thaw(model, lname)
    end
    for j in 1:source.options[:n_unk]
        cname = Symbol(:unk, j)
        if model[cname].norm.val > 0
            thaw(model, cname)
        else
            freeze(model, cname)
        end
    end
    bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)

    # Disable "unknown" lines whose normalization uncertainty is larger
    # than 3 times the normalization
    needs_fitting = false
    for ii in 1:source.options[:n_unk]
        cname = Symbol(:unk, ii)
        isfixed(model, cname)  &&  continue
        if bestfit[cname].norm.val == 0.
            freeze(model, cname)
            needs_fitting = true
            println(source.log, "Disabling $cname (norm. = 0)")
        elseif bestfit[cname].norm.unc / bestfit[cname].norm.val > 3
            model[cname].norm.val = 0.
            freeze(model, cname)
            needs_fitting = true
            println(source.log, "Disabling $cname (unc. / norm. > 3)")
        end
    end
    if needs_fitting
        println(source.log, "\nRe-run fit...")
        bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)
    end

    println(source.log, "\nFinal model and bestfit:")
    show(source.log, model)
    println(source.log)
    show(source.log, bestfit)

    elapsed = time() - elapsed
    println(source.log, "\nElapsed time: $elapsed s")
    close_log(source)

    populate_metadata!(source, model)
    return (model, bestfit)
end
