# QSFit

Quasar Spectral FITting package - http://qsfit.inaf.it/

** Warning **: the software is still under development...


## Install
```julia
using Pkg
Pkg.add(url="https://github.com/gcalderone/QSFit.jl", rev="master")
Pkg.add(url="https://github.com/lnicastro/GFitViewer.jl", rev="master")
```

## Example
```julia
using QSFit, GFitViewer

source = QSO{DefaultRecipe}("My SDSS source", 0.3806, ebv=0.)
add_spec!(source, Spectrum(Val(:SDSS_DR10), "spec-0752-52251-0323.fits"))
res = qsfit(source);
viewer(res, filename="test_qsfit.html")
```
