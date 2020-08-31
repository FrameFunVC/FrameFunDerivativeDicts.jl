using FrameFunDerivativeDicts, BasisFunctions, Test
using SymbolicDifferentialOperators: Δ, I, δx, δy

using Calculus: derivative
import BasisFunctions: diff
diff(args...) = derivative(args...)
@testset begin
    B = Fourier(10)
    g = interpolation_grid(B)
    @test evaluation(δx(B),g)≈evaluation(diff(B,1),g)
    @test_throws  DimensionMismatch (5*δx^0*δy)(Fourier(10))
    @test evaluation(5Δ(B), g)≈evaluation(5diff(B,2),g)
    @test evaluation(I(B), g)≈evaluation(B,g)
    @test evaluation(5I(B), g)≈evaluation(5B,g)
    @test_throws  DimensionMismatch (δx+δy)(B)


    B = cos*Fourier(10)
    g = interpolation_grid(B)
    @test evaluation(δx(B),g)≈evaluation(diff(B,1),g)
    @test_throws  DimensionMismatch (5*δx^0*δy)(Fourier(10))
    @test evaluation(5Δ(B), g)≈evaluation(5diff(B,2),g)
    @test evaluation(I(B), g)≈evaluation(B,g)
    @test evaluation(5I(B), g)≈evaluation(5B,g)
    @test_throws  DimensionMismatch (δx+δy)(B)


    B = Fourier(10)⊗Fourier(11)
    g = interpolation_grid(B)
    @test evaluation(δx(B),g)≈evaluation(diff(B,(1,0)),g)
    @test evaluation((5*δx^0*δy)(B), g)≈evaluation(  5diff(B,(0,1)), g)
    @test evaluation(Δ(B), g)≈evaluation(diff(B,(2,0))+diff(B,(0,2)),g)
    @test evaluation(I(B), g)≈evaluation(B,g)
    @test evaluation((δx+δy)(B), g)≈evaluation(diff(B,(1,0))+diff(B,(0,1)),g)
    @test evaluation((δx+5δy)(B), g)≈evaluation(diff(B,(1,0))+5diff(B,(0,1)),g)

    B = Fourier(10)⊗Fourier(11)⊗Fourier(9)
    g = interpolation_grid(B)
    @test evaluation((5*δx^2*δy)(B), g)≈evaluation(5diff(B,(2,1,0)), g)
    @test evaluation(Δ(B), g)≈evaluation(diff(B,(2,0,0))+diff(B,(0,2,0))+diff(B,(0,0,2)),g)
    @test evaluation(I(B), g)≈evaluation(B,g)
end
