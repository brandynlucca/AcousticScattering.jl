# Preserve the documented opt-out check using the isolated installation harness.
include(joinpath(@__DIR__, "..", "dev", "check_install.jl"))
check_install(["--mode=opt-out"; ARGS])
