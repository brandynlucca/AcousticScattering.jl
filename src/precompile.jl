@compile_workload begin
    let radius = 0.01, len = 0.07, k = 2π * 38000.0 / 1477.3
        kirchhoff(Cylinder(radius, len; radius_curvature = 1e8len), Rigid(), k; incidence_angle = π /
                                                                                                  2)
    end
    let body = Spheroid(0.05, 0.02), k = 2π * 38000.0 / 1477.4
        kirchhoff(body, Rigid(), k; incidence_angle = π / 2)
    end
end
