@testset "Curved surface bounds" begin
    for (x, y) in ((0.1, 0.2), (nextfloat(1.0), prevfloat(1.0)),
            (floatmin(Float64), floatmin(Float64)), (1e100, -1e100)),
        operation in (+, -, *)

        bound = operation(AS._SurfaceBound(x), AS._SurfaceBound(y))
        exact = operation(Rational{BigInt}(x), Rational{BigInt}(y))
        @test Rational{BigInt}(bound.lo) <= exact <= Rational{BigInt}(bound.hi)
    end
    nodes = [0.0 1 0 0 1/3 2/3 0 0 0 0 2/3 1/3 2/3 1/3 0 0 1/3 1/3 0 1/3;
             0 0 1 0 0 0 1/3 2/3 0 0 1/3 2/3 0 0 2/3 1/3 1/3 0 1/3 1/3;
             0 0 0 1 0 0 0 0 1/3 2/3 0 0 1/3 2/3 1/3 2/3 0 1/3 1/3 1/3]
    triangles = [1 1 1 2; 3 2 4 3; 2 4 3 4; 7 5 9 11; 8 6 10 12;
                 12 13 16 15; 11 14 15 16; 6 10 8 14; 5 9 7 13; 17 18 19 20]
    plain = mesh(nodes, triangles)
    @test plain.body.validation.arithmetic == :outward_rounded
    @test plain.body.validation.intersection_check == :adaptive_bernstein

    # These cubic perturbations hide a fold or crossing between the old sixth-edge samples.
    folded = copy(nodes)
    folded[2, 17] = 0.4851851851851852
    @test_throws r"nonpositive projected Jacobian" mesh(folded, triangles)
    crossing = copy(nodes)
    crossing[3, 17] = 0.15
    @test_throws r"validation unresolved.*separation" mesh(crossing, triangles)

    regular = copy(nodes)
    regular[2, 17] = 0.4777777777777778
    near_fold = mesh(regular, triangles)
    @test near_fold.body.validation.depth > 0
    close = copy(nodes)
    close[3, 17] = 0.14
    near_contact = mesh(close, triangles)
    @test near_contact.body.validation.depth > 0
    @test_throws r"refinement limit" mesh(close, triangles; validation = (maxdepth = 0,))
    @test_throws r"work limit" mesh(nodes, triangles; validation = (maxwork = 1,))
    @test_throws ArgumentError mesh(nodes, triangles; validation = (maxdepth = -1,))
    @test_throws ArgumentError mesh(nodes, triangles; validation = (maxwork = 0,))
    @test AS.gmsh.isInitialized() == 0
    scaled = mesh(1000close, triangles; units = :mm)
    @test AS.coordinates(scaled) ≈ AS.coordinates(near_contact)
    @test AS.normals(scaled) ≈ AS.normals(near_contact)
    shifted = mesh(close .+ [100.0, -20.0, 10.0], triangles)
    @test AS.normals(shifted) ≈ AS.normals(near_contact)
    @test_throws r"validation unresolved.*separation" mesh(crossing,
        triangles[:, [4, 2, 3, 1]])
end
