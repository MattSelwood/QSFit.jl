calibsum(calib, args...) = calib .* (.+(args...))

function multiepoch_fit(source::QSO{TRecipe}; ref_id=1) where TRecipe <: DefaultRecipe
    Nspec = length(source.domain)

    elapsed = time()
    mzer = GFit.cmpfit()
    mzer.config.ftol = mzer.config.gtol = mzer.config.xtol = 1.e-6

    # Arrays containing best fit values to be constrained across epochs
    galaxy_best = Vector{Float64}()
    galaxy_unc  = Vector{Float64}()
    OIII_best = Vector{Float64}()
    OIII_unc  = Vector{Float64}()

    # Initialize components and guess initial values
    println(source.log, "\nFit continuum components...")
    preds = Vector{Prediction}()
    for id in 1:Nspec
        λ = source.domain[id][:]
        pred = Prediction(source.domain[id], :Continuum => Reducer(sum, [:qso_cont]),
                          :qso_cont => QSFit.qso_cont_component(TRecipe))
        push!(preds, pred)
        c = pred[:qso_cont]
        c.norm.val = interpol(source.data[id].val, λ, c.x0.val)
    end
    model = Model(preds)

    for id in 1:Nspec
        λ = source.domain[id][:]

        # Host galaxy template
        if source.options[:use_host_template]
            add!(model[id], :Continuum => Reducer(sum, [:qso_cont, :galaxy]),
                 :galaxy => QSFit.hostgalaxy(source.options[:host_template]))
            model[id][:galaxy].norm.val = interpol(source.data[id].val, λ, 5500)
        end

        # Balmer continuum and pseudo-continuum
        if source.options[:use_balmer]
            tmp = [:qso_cont, :balmer]
            source.options[:use_host_template]  &&  push!(tmp, :galaxy)
            add!(model[id], :Continuum => Reducer(sum, tmp),
                 :balmer => QSFit.balmercont(0.1, 0.5))
            c = model[id][:balmer]
            c.norm.val  = 0.1
            c.norm.fixed = false
            c.norm.high = 0.5
            c.ratio.val = 0.5
            c.ratio.fixed = false
            c.ratio.low  = 0.3
            c.ratio.high = 1
            patch!(model) do m
                m[id][:balmer].norm *= m[id][:qso_cont].norm
            end
        end
    end

    for id in 1:Nspec
        bestfit = fit!(model, only_id=id, source.data, minimizer=mzer);  show(source.log, bestfit)
        push!(galaxy_best, bestfit[id][:galaxy].norm.val)
        push!(galaxy_unc , bestfit[id][:galaxy].norm.unc)
    end

    # QSO continuum renormalization
    for id in 1:Nspec
        freeze(model[id], :qso_cont)
        c = model[id][:qso_cont]
        println(source.log, "Cont. norm. (before): ", c.norm.val)
        while true
            residuals = (model[id]() - source.data[id].val) ./ source.data[id].unc
            (count(residuals .< 0) / length(residuals) > 0.9)  &&  break
            c.norm.val *= 0.99
            evaluate!(model)
        end
        println(source.log, "Cont. norm. (after) : ", c.norm.val)

        freeze(model[id], :qso_cont)
        source.options[:use_host_template]  &&  freeze(model[id], :galaxy)
        source.options[:use_balmer]         &&  freeze(model[id], :balmer)
    end
    evaluate!(model)

    # Fit iron templates
    println(source.log, "\nFit iron templates...")
    for id in 1:Nspec
        iron_components = Vector{Symbol}()
        if source.options[:use_ironuv]
            add!(model[id], :ironuv => QSFit.ironuv(3000))
            model[id][:ironuv].norm.val = 0.5
            push!(iron_components, :ironuv)
        end
        if source.options[:use_ironopt]
            add!(model[id],
                 :ironoptbr => QSFit.ironopt_broad(3000),
                 :ironoptna => QSFit.ironopt_narrow(500))
            model[id][:ironoptbr].norm.val = 0.5
            model[id][:ironoptna].norm.val = 0.0
            freeze(model[id], :ironoptna)  # will be freed during last run
            push!(iron_components, :ironoptbr, :ironoptna)
        end
        if length(iron_components) > 0
            add!(model[id], :Iron => Reducer(sum, iron_components))
            add!(model[id], :main => Reducer(sum, [:Continuum, :Iron]))
            evaluate!(model)
            bestfit = fit!(model, only_id=id, source.data, minimizer=mzer); show(source.log, bestfit)
        else
            add!(model[id], :Iron => Reducer(() -> [0.], Symbol[]))
            add!(model[id], :main => Reducer(sum, [:Continuum, :Iron]))
        end
        source.options[:use_ironuv]   &&  freeze(model[id], :ironuv)
        source.options[:use_ironopt]  &&  freeze(model[id], :ironoptbr)
        source.options[:use_ironopt]  &&  freeze(model[id], :ironoptna)
    end
    evaluate!(model)

    # Add emission lines
    line_names = [collect(keys(source.line_names[id])) for id in 1:Nspec]
    line_groups = [unique(collect(values(source.line_names[id]))) for id in 1:Nspec]
    println(source.log, "\nFit known emission lines...")
    for id in 1:Nspec
        λ = source.domain[id][:]

        add!(model[id], source.line_comps[id])
        for (group, lnames) in invert_dictionary(source.line_names[id])
            add!(model[id], group => Reducer(sum, lnames))
        end
        add!(model[id], :main => Reducer(sum, [:Continuum, :Iron, line_groups[id]...]))

        if haskey(model[id], :MgII_2798)
            model[id][:MgII_2798].voff.low  = -1000
            model[id][:MgII_2798].voff.high =  1000
        end
        if haskey(model[id], :OIII_5007_bw)
            model[id][:OIII_5007_bw].fwhm.val  = 500
            model[id][:OIII_5007_bw].fwhm.low  = 1e2
            model[id][:OIII_5007_bw].fwhm.high = 1e3
            model[id][:OIII_5007_bw].voff.low  = 0
            model[id][:OIII_5007_bw].voff.high = 2e3
        end

        # Guess values
        evaluate!(model)
        y = source.data[id].val - model[id]()
        for cname in line_names[id]
            c = model[id][cname]
            yatline = interpol(y, λ, c.center.val)
            c.norm.val = 1.
            c.norm.val = abs(yatline) / QSFit.maxvalue(model[id][cname])
        end

        # Patch parameters
        if  haskey(model[id], :OIII_4959)  &&
            haskey(model[id], :OIII_5007)
            model[id][:OIII_4959].voff.fixed = true
            patch!(model) do m
                m[id][:OIII_4959].voff = m[id][:OIII_5007].voff
            end
        end
        if  haskey(model[id], :NII_6549)  &&
            haskey(model[id], :NII_6583)
            model[id][:NII_6549].voff.fixed = true
            patch!(model) do m
                m[id][:NII_6549].voff = m[id][:NII_6583].voff
            end
        end
        if  haskey(model[id], :OIII_5007_bw)  &&
            haskey(model[id], :OIII_5007)
            patch!(model) do m
                m[id][:OIII_5007_bw].voff += m[id][:OIII_5007].voff
                m[id][:OIII_5007_bw].fwhm += m[id][:OIII_5007].fwhm
            end
        end

        #=
        model[id][:br_Hb].voff.fixed = 1
        model[id][:br_Hb].fwhm.fixed = 1
        patch!(model) do m
            m[id][:br_Hb].voff = m[id][:br_Ha].voff
            m[id][:br_Hb].fwhm = m[id][:br_Ha].fwhm
        end
        =#

        bestfit = fit!(model, only_id=id, source.data, minimizer=mzer); show(source.log, bestfit)
        push!(OIII_best, bestfit[id][:OIII_5007].norm.val)
        push!(OIII_unc , bestfit[id][:OIII_5007].norm.unc)

        for lname in line_names[id]
            freeze(model[id], lname)
        end
    end

    # Add unknown lines
    println(source.log, "\nFit unknown emission lines...")
    if source.options[:n_unk] > 0
        for id in 1:Nspec
            tmp = OrderedDict{Symbol, AbstractComponent}()
            for j in 1:source.options[:n_unk]
                tmp[Symbol(:unk, j)] = line_component(TRecipe, UnkLine(5e3))
            end
            add!(model[id], :UnkLines => Reducer(sum, collect(keys(tmp))), tmp)
            add!(model[id], :main => Reducer(sum, [:Continuum, :Iron, line_groups[id]..., :UnkLines]))
            evaluate!(model)
            for j in 1:source.options[:n_unk]
                freeze(model[id], Symbol(:unk, j))
            end
        end
    else
        # Here we need a :UnkLines reducer, even when n_unk is 0
        for id in 1:Nspec
            add!(model[id], :UnkLines => Reducer(() -> [0.], Symbol[]))
            add!(model[id], :main => Reducer(sum, [:Continuum, :Iron, line_groups[id]..., :UnkLines]))
        end
    end
    evaluate!(model)

    # Set "unknown" line center wavelength where there is a maximum in
    # the fit residuals, and re-run a fit.
    for id in 1:Nspec
        λ = source.domain[id][:]
        λunk = Vector{Float64}()
        while true
            (length(λunk) >= source.options[:n_unk])  &&  break
            evaluate!(model)
            Δ = (source.data[id].val - model[id]()) ./ source.data[id].unc

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
            model[id][cname].norm.val = 1.
            model[id][cname].center.val  = λ[iadd]
            model[id][cname].center.low  = λ[iadd] - λ[iadd]/10. # allow to shift 10%
            model[id][cname].center.high = λ[iadd] + λ[iadd]/10.

            thaw(model, cname)
            bestfit = fit!(model, only_id=id, source.data, minimizer=mzer); show(source.log, bestfit)
            freeze(model, cname)
        end
    end
    evaluate!(model)

    # ----------------------------------------------------------------
    # Constrain component normalization across epochs.  Note:
    # reference spectrum must have reliable estimation of all common
    # components
    rval = [galaxy_best[ref_id], OIII_best[ref_id]]
    runc = [galaxy_unc[ ref_id], OIII_unc[ ref_id]]
    @assert count((rval .!= 0)  .&  (runc .!= 0)) == 2

    # Estimate calibration in all epochs w.r.t. reference epoch
    for id in 1:Nspec
        if id != ref_id
            val = [galaxy_best[id], OIII_best[id]]
            unc = [galaxy_unc[ id], OIII_unc[ id]]
            j = findall((val .!= 0)  .&  (unc .!= 0))

            R = val[j] ./ rval[j]
            R_unc = (unc[j] ./ val[j] .+ runc[j] ./ rval[j]) .* R
            ratio = sum(R ./ R_unc) ./ sum(1 ./ R_unc)
            for cname in keys(preds[id].cevals)
                if :norm in propertynames(model[cname])
                    model[cname].norm.val /= ratio
                end
            end
            model[id][:galaxy].norm.fixed = true
            model[id][:OIII_5007].norm.fixed = true

            add!(model[id],
                 :main => Reducer(calibsum, [:calib, :Continuum, :Iron, line_groups[id]..., :UnkLines]),
                 :calib => ratio)
        end
    end
    evaluate!(model)

    for id in 1:Nspec
        if id != ref_id
            patch!(model) do m
                m[id][   :galaxy].norm = m[ref_id][   :galaxy].norm
                m[id][:OIII_5007].norm = m[ref_id][:OIII_5007].norm
            end
        end
    end
    evaluate!(model)

    # Last run with all parameters free
    println(source.log, "\nLast run with all parameters free...")
    for id in 1:Nspec
        thaw(model[id], :qso_cont)
        source.options[:use_host_template]  &&  thaw(model[id], :galaxy)
        source.options[:use_balmer]         &&  thaw(model[id], :balmer)
        source.options[:use_ironuv]         &&  thaw(model[id], :ironuv)
        source.options[:use_ironopt]        &&  thaw(model[id], :ironoptbr)
        source.options[:use_ironopt]        &&  thaw(model[id], :ironoptna)

        for lname in line_names[id]
            thaw(model[id], lname)
        end
        for j in 1:source.options[:n_unk]
            cname = Symbol(:unk, j)
            if model[id][cname].norm.val > 0
                thaw(model[id], cname)
            else
                freeze(model[id], cname)
            end
        end
        if id != ref_id
            thaw(model[id], :calib)  # parameter is fixed in preds[ref_id]
        end
    end
    bestfit = fit!(model, source.data, minimizer=mzer); show(source.log, bestfit)

    # Disable "unknown" lines whose normalization uncertainty is larger
    # than 3 times the normalization
    needs_fitting = false
    for id in 1:Nspec
        for ii in 1:source.options[:n_unk]
            cname = Symbol(:unk, ii)
            isfixed(model[id], cname)  &&  continue
            if bestfit[id][cname].norm.val == 0.
                freeze(model[id], cname)
                needs_fitting = true
                println(source.log, "Disabling $cname (norm. = 0)")
            elseif bestfit[id][cname].norm.unc / bestfit[id][cname].norm.val > 3
                model[id][cname].norm.val = 0.
                freeze(model[id], cname)
                needs_fitting = true
                println(source.log, "Disabling $cname (unc. / norm. > 3)")
            end
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
