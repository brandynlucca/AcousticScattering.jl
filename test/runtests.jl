# TEST_SUITE selects the lean regression suite, the comprehensive suite, or both.
# GitHub Actions can set TEST_SUITE=Basic or TEST_SUITE=Extended per job.
const TEST_SUITE = get(ENV, "TEST_SUITE", isempty(ARGS) ? "Basic" : only(ARGS))
TEST_SUITE in ("Basic", "Extended", "All") ||
    throw(ArgumentError("Unknown TEST_SUITE=$TEST_SUITE. Choose Basic, Extended, or All."))

@info "Running $TEST_SUITE test suite"

if TEST_SUITE in ("Basic", "All")
    include("basic/runtests.jl")
end

if TEST_SUITE in ("Extended", "All")
    include("extended/runtests.jl")
end
