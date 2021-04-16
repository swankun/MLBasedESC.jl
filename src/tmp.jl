# partlen = cumsum(2:5) .+ 1
# start = [1; partlen[1:end-1]]
# idx = [range(a, step=1, length=b) for (a,b) in zip(cumsum(start), partlen)]
# L = zeros(Int, 14, 14)
# θ = 1:34
# rows = Int[]
# cols = Int[]
# cartids = Tuple{Int,Int}[]

# for (ci, fi) in zip(start,idx)
#     l = vec2tril(θ[fi])
#     n = size(l,1) - 1 
#     L[ci:ci+n, ci:ci+n] = l 
#     ip = Iterators.product(ci:ci+n, ci:ci+n)
#     append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
# end

# For N = 2, D = 1:4 monovec
cartids = Tuple{Int,Int}[]
ip = Iterators.product(1:5,   1:5);   append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
ip = Iterators.product(6:9,   6:9);   append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
ip = Iterators.product(10:12, 10:12); append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
ip = Iterators.product(13:14, 13:14); append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
ip = Iterators.product(1:5, 10:12);   append!( cartids, collect(ip) )
ip = Iterators.product(6:9, 13:14);   append!( cartids, collect(ip) )
rows = last.(cartids)
cols = first.(cartids)
θ = 1:57 # randn(57)

@time L = sparse(rows, cols, θ);
@time begin
    D = zeros(eltype(θ), 14, 14)
    for (x,i,j) in zip(θ, rows, cols)
        setindex!(D, x, i, j)
    end
end


# For N = 2, D = 1:2 monovec
cartids = Tuple{Int,Int}[]
ip = Iterators.product(1:3,   1:3);   append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
ip = Iterators.product(4:5,   4:5);   append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
rows = last.(cartids)
cols = first.(cartids)
θ = 1:9

@time L = sparse(rows, cols, θ);
@time begin
    D = zeros(eltype(θ), 5, 5)
    for (x,i,j) in zip(θ, rows, cols)
        setindex!(D, x, i, j)
    end
end