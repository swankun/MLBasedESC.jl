partlen = cumsum(2:5) .+ 1
start = [1; partlen[1:end-1]]
idx = [range(a, step=1, length=b) for (a,b) in zip(cumsum(start), partlen)]
L = zeros(Int, 14, 14)
θ = 1:34
rows = Int[]
cols = Int[]
cartids = Tuple{Int,Int}[]

for (ci, fi) in zip(start,idx)
    l = vec2tril(θ[fi])
    n = size(l,1) - 1 
    L[ci:ci+n, ci:ci+n] = l 
    ip = Iterators.product(ci:ci+n, ci:ci+n)
    append!( cartids, filter(x->first(x)<=last(x), collect(ip)) )
end
ip = Iterators.product(1:2, 6:9);   append!( cartids, collect(ip) )
ip = Iterators.product(3:5, 10:14); append!( cartids, collect(ip) )
rows = last.(cartids)
cols = first.(cartids)
