module SummationDicts

import Base: +, -, *, size, length, eltype, axes, getindex, diff
import Base.Broadcast: broadcastable, BroadcastStyle, materialize, result_style
import BasisFunctions: evaluation, support,
    src, dest, isinplace, apply_not_inplace!, apply_inplace!,
    domaintype, codomaintype, plotgrid
import SparseArrays: sparse

using BasisFunctions, GridArrays
using BasisFunctions: TypedFunction, CompositeDict
using Base.Broadcast: DefaultArrayStyle, Broadcasted, broadcasted
using Base: tail
using MacroTools: @forward


struct  BCDictStyle{S,T,N} <: Broadcast.BroadcastStyle end
struct  BCOperatorStyle{T} <: Broadcast.BroadcastStyle end

broadcastable(dict::Dictionary) = dict
BroadcastStyle(::Type{<:Dictionary{S,T}}) where {S,T} = BCDictStyle{S,T,1}()
BroadcastStyle(::Type{<:TensorProductDict{N,DICTS,S,T}}) where {N,DICTS,S,T} = BCDictStyle{S,T,N}()
result_style(s1::BCDictStyle{S,T}...) where {S,T} = BCDictStyle{S,T,1}()
materialize(bc::Broadcasted{<:BCDictStyle}) = LazyDictionary(bc)
domaintype(::Broadcasted{<:BCDictStyle{S}}) where {S} = S
codomaintype(::Broadcasted{<:BCDictStyle{S,T}}) where {S,T} = T
support(bc::Broadcasted{<:BCDictStyle}) = support(bc.args[1])
plotgrid(bc::Broadcasted{<:BCDictStyle},n) = plotgrid(bc.args[1],n)

broadcastable(op::DictionaryOperator) = op
BroadcastStyle(::Type{<:DictionaryOperator{T}}) where T = BCOperatorStyle{T}()
materialize(bc::Broadcasted{<:BCOperatorStyle}) = LazyDictionaryOperator(bc)
eltype(::Broadcasted{BCOperatorStyle{T}}) where T = T
sparse(A::Broadcasted{<:BCOperatorStyle}) =
    broadcast(A.f, (sparse(arg) for arg in A.args)...)

+(dict::Dictionary) =
    dict
*(dict::Dictionary) =
    dict

struct LazyDictionary{S,T,BC<:Broadcasted} <: Dictionary{S,T}
    bc::BC
    function LazyDictionary(bc::Broadcasted)
        S = domaintype(bc.args[1])
        T = codomaintype(bc.args[1])
        sup = support(bc.args[1])
        @assert all(S==domaintype(arg) for arg in tail(bc.args))
        @assert all(T==codomaintype(arg) for arg in tail(bc.args))
        @assert all(sup≈support(arg) for arg in tail(bc.args))

        new{S,T,typeof(bc)}(bc)
    end
end

struct LazyDictionaryOperator{T,BC<:Broadcasted} <: DictionaryOperator{T}
    bc::BC
    scratch::Array{T}
    function LazyDictionaryOperator(bc::Broadcasted)
        T = eltype(bc.args[1])
        @assert all(T==eltype(arg) for arg in tail(bc.args))
        new{T,typeof(bc)}(bc,zeros(dest(bc.args[1])))
    end
end

@forward LazyDictionary.bc size, length, eltype, axes, getindex, domaintype, codomaintype, support
broadcastable(dict::LazyDictionary) = dict.bc
plotgrid(dict::LazyDictionary,n) = plotgrid(dict.bc,n)

@forward LazyDictionaryOperator.bc size, length, eltype, axes, getindex, sparse
broadcastable(A::LazyDictionaryOperator) = A.bc
src(A::LazyDictionaryOperator) = src(A.bc.args[1])
dest(A::LazyDictionaryOperator) = dest(A.bc.args[1])
isinplace(A::LazyDictionaryOperator) = false
size(A::LazyDictionaryOperator, i::Int) = size(A)[i]

apply_not_inplace!(op::LazyDictionaryOperator, coef_dest, coef_src) =
    apply_not_inplace!(op.bc.f, op, coef_dest, coef_src)
function apply_not_inplace!(f, op::LazyDictionaryOperator, coef_dest, coef_src)
    fill!(coef_dest, 0)
    for opi in op.bc.args
        if isinplace(opi)
            copy!(op.scratch, coef_src)
            apply!(opi, op.scratch)
            for i in eachindex(coef_dest)
                coef_dest[i] = f(coef_dest[i], op.scratch[i])
            end
        else
            apply!(opi, op.scratch, coef_src)
            for i in eachindex(coef_dest)
                coef_dest[i] = f(coef_dest[i], op.scratch[i])
            end
        end
    end
    coef_dest
end


for op in (:+,:-,:*)
    @eval $op(f1::TypedFunction,f2::TypedFunction) =
        LazyTypedFunction($op,f1,f2)
    @eval $op(f::TypedFunction...) =
        LazyTypedFunction($op,f...)
    @eval $op(dict1::Dictionary, dicts::Dictionary...) =
        broadcast($op, dict1, dicts...)
end

export laplace
laplace(dict::Dictionary) =
    +((_partialderivative(dict,dim,2) for dim in 1:dimension(dict))...)
laplace(dict::MultiDict) =
    multidict([laplace(dicti) for dicti in elements(dict)]...)

_partialderivative(dict::Dictionary,dim::Int,order::Int) =
    diff(dict,_be_forgiving(ntuple(k->k==dim ? order : 0, Val(dimension(dict)))))
_be_forgiving(i::Tuple{Int}) = i[1]
_be_forgiving(i) = i

struct LazyTypedFunction{S,T,F,Args<:Tuple}  <: TypedFunction{S,T}
    f::F
    args::Args
    LazyTypedFunction(op, f::TypedFunction...) = error()
    LazyTypedFunction(op, f::TypedFunction{S,T}...) where {S,T} = new{S,T,typeof(op),typeof(f)}(op,f)
end

(f::LazyTypedFunction{S})(x::S) where {S}= reduce(f.f, fi(x) for fi in f.args)

function evaluation(::Type{T}, dict::LazyDictionary, gb::GridBasis, grid::AbstractGrid; options...) where {T}
    As = (evaluation(T, dicti, gb, grid; options...) for dicti in dict.bc.args)
    fops, ops, bops = _extract(As...)
    if fops==nothing==bops
        LazyDictionaryOperator(broadcasted(dict.bc.f, ops...))
    elseif bops==nothing
        fops*LazyDictionaryOperator(broadcasted(dict.bc.f, ops...))
    elseif fops==nothing
        LazyDictionaryOperator(broadcasted(dict.bc.f, ops...))*bops
    else
        fops*LazyDictionaryOperator(broadcasted(dict.bc.f, ops...))*bops
    end
end

function _extract(ops::DictionaryOperator...)
    fops = nothing
    bops = nothing

    # r =  _restricted(ops...)
    # if !isnothing(r)
    #     fops, ops = r
    # end
    fops, ops, bops
end

function _restricted(ops::TensorProductOperator...)
    R = [_restricted((element(op,i) for op in ops)...) for i in 1:dimension(ops)]
    if all(isnothing, R)
        return nothing
    end
    fops = tensorproduct([isnothing(Ri) ? IdentityOperator(dest(op),dest(op)) : Ri[1] for (Ri,op) in zip(R,elements(ops[1]))]...)
    ops = [tensorproduct([isnothing(Ri) ? e : Ri[2][i] for (Ri,e) in zip(R,elements(op))]...)    for (i,op) in enumerate(ops)]
    fops, ops
end
import Base:*
*(op::DictionaryOperator) = op
function _restricted(ops::CompositeOperator...)
    fops = tuple([element(op,length(elements(op))) for op in ops]...)
    if all(x->isa(x,IndexRestriction), fops)
        ix = subindices(fops[1])
        if all(x->subindices(x)==ix, Base.tail(fops))
            return fops[1], [*(reverse(elements(op)[1:end-1])...)  for op in ops]
        end
        return nothing
    end
    return nothing
end
end
