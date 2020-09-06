module FrameFunDerivativeDicts


include("BasisFunctions/summation_dict.jl")

using Reexport
module SymbolicDiffs

using Reexport, BasisFunctions
@reexport using SymbolicDifferentialOperators

using SymbolicDifferentialOperators: AbstractDiffOperator, PartialDifferentialOperator,
    ScaledDifferentialOperator, dimension_names, IdentityCoefficient, ConstantCoefficient,
    ProductDifferentialOperator, LaplaceOperator, IdentityDifferentialOperator,
    SumDifferentialOperator, full_elements

(L::AbstractDiffOperator)(obj) = _eval(L,obj)
_eval(L::AbstractDiffOperator, obj) =
    _eval1(L, obj)
_eval1(L::AbstractDiffOperator, obj) =
    _eval2(L, obj)
_eval2(L::AbstractDiffOperator, obj) = error("Influence of $(typeof(L)) on $(typeof(obj)) is unknown")

_eval(L::AbstractDiffOperator, exp::Expansion) = expansion(L(dictionary(exp)), coefficients(exp))

function _eval1(L::PartialDifferentialOperator, dict)
    if dimension(dict)==1
        diff(dict, 1)
    else
        @warn """
            Partial derivative in 1 dimension on multidimensional dictionary is ambigious.
            Differentiating the first dimension.
            """
        diff(dict; order=1,dim=1)
    end
end

_eval1(L::ScaledDifferentialOperator,dict) =
    _eval(L.coeff, L.diff, dict)
_eval(coef::IdentityCoefficient, diff, dict) =
    diff(dict)
_eval(coef::ConstantCoefficient, diff, dict) =
    coef.scalar*diff(dict)

function _eval1(L::ProductDifferentialOperator, dict)
    diff_L = length(dimension_names(L))
    dict_L = dimension(dict)
    if diff_L == dict_L
        diff(dict, L.orders)
    elseif diff_L < dict_L
        @warn """
            Derivative in $(diff_L) dimensions on $(dict_L)-D dictionary is ambigious.
            Differentiating in the first dimensions.
            """

        diff(dict, (L.orders..., ntuple(k->0,Val(dict_L-diff_L))...))
    else
        throw(DimensionMismatch("To many dimensions in $(typeof(L)) to act on $(typeof(dict))"))
    end
end


using ..SummationDicts: laplace
_eval1(::LaplaceOperator, dict) = laplace(dict)
_eval1(::IdentityDifferentialOperator, dict) = dict
function _eval1(L::SumDifferentialOperator, dict)
    if dimension(dict) != length(dimension_names(L))
        throw(DimensionMismatch("To dimensions in $(typeof(L)) ($(length(dimension_names(L)))) do not match the dimension of $(name(dict)) ($(dimension(dict)))"))
    end
    +([op(dict) for op in full_elements(L)]...)
end
end

module PDEs
    using BasisFunctions
    using SymbolicDifferentialOperators: AbstractDiffOperator, NormalDerivativeOperator
    using GridArrays: AbstractGrid
    using FrameFun: normal

    export PDERule
    struct PDERule
        operator::AbstractDiffOperator
        dict::Dictionary
        grid::AbstractGrid
        rhs
    end

    export PDENormalRule
    PDENormalRule(domain, dict, grid, f) =
        PDERule(NormalDerivativeOperator(domain),dict,grid,f)

    export PDEBlock
    struct PDEBlock
        dict::Dictionary
        lhs::DictionaryOperator
        rhs::AbstractArray
    end
    export PDE
    struct PDE
        blocks
        lhs::DictionaryOperator
        rhs::AbstractVector
    end

    _pde_block(L::AbstractDiffOperator, dict::Dictionary, g::AbstractGrid, rhs_function) =
        evaluation(L(dict), g), sample(g, rhs_function, codomaintype(dict))
    function _normals(::Type{T}, grid, domain) where T
        C = Array{T}(undef, length(grid),dimension(grid))
        _normals!(C, grid, domain)
        C
    end
    function _normals!(C, grid, domain)
        for (i,x) in enumerate(grid)
            @views C[i,:] .= normal(x,domain)
        end
    end

    function _pde_block(L::NormalDerivativeOperator, dict::Dictionary, grid::AbstractGrid, rhs_function)
        # E = evaluation(dict, grid)
        D = dimension(grid)
        DEs = [evaluation(diff(dict,ntuple(k->k==i ? 1 : 0,Val(D))),grid) for i in 1:D]
        T = eltype(DEs[1])
        dicts = dest.(DEs)
        C = _normals(T, grid, L.domain)
        Ds = [DiagonalOperator(dicts[i], @views(C[:,i])) for i in 1:D]
        op = .+(map((x,y)->x*y,Ds,DEs)...)
        op, sample(grid, rhs_function, codomaintype(dict))
    end

    PDEBlock(rule::PDERule) =
        PDEBlock(rule.dict, _pde_block(rule.operator, rule.dict, rule.grid, rule.rhs)...)
    PDE(block::PDEBlock) =
        PDE((block,), block.lhs, reshape(block.rhs,length(block.rhs)))
    PDE(blocks::PDEBlock...) =
        PDE(blocks, vcat([block.lhs for block in blocks]...), vcat([reshape(block.rhs,length(block.rhs)) for block in blocks]...))
    PDE(rules::PDERule...) =
        PDE(PDEBlock.(rules)...)
    # Atom slows down
    Base.show(io::IO,::PDE) = Base.show(io,PDE)
end
@reexport using .PDEs

end # module
