export optimize!

function optimize!(loss::IDAPBCLoss{P}, paramvec::AbstractVector, data; 
    η=0.001, batchsize=64, maxiters=1e4, tol=1e-4) where
    {J2,M,MD,V,VD<:Function,P<:IDAPBCProblem{J2,M,MD,V,VD}}

    batchgs = Vector{typeof(paramvec)}(undef, batchsize)
    dataloader = Flux.Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    max_batch = round(Int, length(data) / dataloader.batchsize, RoundUp)
    optimizer = Flux.ADAM(η)
    nepoch = 0;
    train_loss = 1/length(data) * pmap(x->loss(x,paramvec), data)
    losstypestr = typeof(loss).name.name |> string
    while nepoch < maxiters && train_loss > tol
        estatus(losstypestr, nepoch, train_loss, maxiters)
        nbatch = 1
        for batch in dataloader
            npoints = length(batch)
            batchloss = 1/npoints * pmap(x->loss(x,paramvec), batch)
            bstatus(nbatch, max_batch, batchloss)
            pmap!(batchgs, x->gradient(loss,x,paramvec), batch)
            grads = 1/npoints * sum(batchgs[1:npoints])
            if !any(isnan.(grads))
                Flux.Optimise.update!(optimizer, paramvec, grads)
                nbatch += 1
            else
                @warn "∂ℓ/∂θ computations yielded NaNs."
            end
        end
        nepoch += 1
        train_loss = 1/length(data) * pmap(x->loss(x,paramvec), data)
    end
    estatus(losstypestr, nepoch, 1/length(data) * pmap(x->loss(x,paramvec), data), maxiters)
end

function optimize!(loss::IDAPBCLoss{P}, paramvec::Flux.Params, data; 
    η=0.001, batchsize=64, maxiters=1e4, tol=1e-4) where
    {J2,M,MD,V,VD<:Chain,P<:IDAPBCProblem{J2,M,MD,V,VD}}
    
    batchgs = Vector{Zygote.Grads}(undef, batchsize)
    dataloader = Flux.Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    max_batch = round(Int, length(data) / dataloader.batchsize, RoundUp)
    optimizer = Flux.ADAM(η)
    nepoch = 0;
    train_loss = 1/length(data) * pmap(x->loss(x), data)
    losstypestr = typeof(loss).name.name |> string
    while nepoch < maxiters && train_loss > tol
        estatus(losstypestr, nepoch, train_loss, maxiters)
        nbatch = 1
        for batch in dataloader
            npoints = length(batch)
            batchloss = 1/npoints * pmap(loss, batch)
            bstatus(nbatch, max_batch, batchloss)
            pmap!(batchgs, x->gradient(loss,x,paramvec), batch)
            grads = @. +(batchgs[1:npoints]...)
            foreach(grads) do x
                x = x/npoints
            end
            if !any(mapreduce(x->vec(isnan.(x)), vcat, grads))
                Flux.Optimise.update!(optimizer, paramvec, grads)
                nbatch += 1
            else
                @warn "∂ℓ/∂θ computations yielded NaNs."
            end
        end
        nepoch += 1
        train_loss = 1/length(data) * pmap(loss, data)
    end
    estatus(losstypestr, nepoch, 1/length(data) * pmap(loss, data), maxiters)
end


bstatus(b,bs,l) = @printf("Batch %05d/%05d | %10.4e\r", b, bs, l)
estatus(t,e,l,maxiters) = @printf("EPOCH %05d/%05d | TRAIN LOSS (%s) = %10.4e\n", e, maxiters, t, l)

function pmap(f, batch::Vector{Vector{T}}) where {T<:Real}
    l = Threads.Atomic{eltype(first(batch))}(0)
    n = length(batch)
    Threads.@threads for id in 1:n
        Threads.atomic_add!(l, f(batch[id]))
    end
    return l.value
end

function pmap!(out, f, batch::Vector{Vector{T}}) where {T<:Real}
    n = length(batch)
    Threads.@threads for id in 1:n
        out[id] = f(batch[id])
    end
end
