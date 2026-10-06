module VolumeMeshChecks
using AcousticScattering, Test
const AS = AcousticScattering
const F = AS.Ferrite

@testset "Volume mesh validation" begin
    points = F.reference_coordinates(AS._VOLUME_GIP)
    nodes = [F.Node(Tuple(p)) for p in points]
    cell = F.QuadraticTetrahedron(Tuple(1:10))
    grid = F.Grid([cell], nodes)
    @test AS._volume_valid_grid(grid, [AS._REGION_FLUID])
    reflected = [F.Node((-p[1], p[2], p[3])) for p in points]
    @test !AS._volume_valid_grid(F.Grid([cell], reflected), [AS._REGION_FLUID])
    @test AS._volume_valid_grid(F.Grid([cell], reflected), [0])
    # Invalid connectivity is a programming/data error, not a remeshing request.
    broken = F.QuadraticTetrahedron((11, 2, 3, 4, 5, 6, 7, 8, 9, 10))
    @test_throws "BoundsError" AS._volume_valid_grid(
        F.Grid([broken], nodes), [AS._REGION_FLUID])
end
end
