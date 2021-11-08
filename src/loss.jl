export PDELossMd, PDELossVd, PDELossCombined, ConvexVdLoss, optimize!,
    loss_massd, ∂loss_massd, loss_ped, ∂loss_ped

###############################################################################

function loss_massd(prob::IDAPBCProblem, q, θ=prob.init_params)
    massd_inv_ps = getindex(θ, prob.ps_index[:mass_inv])
    massd_inv = prob.hamd.mass_inv(q, massd_inv_ps)
    massd_inv_gs = prob.hamd.jac_mass_inv(q, massd_inv_ps)
    massd = inv(massd_inv)
    mass_inv = prob.ham.mass_inv(q)
    mass_inv_gs = prob.ham.jac_mass_inv(q)
    
    nq = size(massd,1)
    map(1:nq) do j
        uk_ps = getindex(θ, prob.ps_index[Symbol("j2_u", j)])
        uk = prob.interconnection[j](q, uk_ps)
        sum(abs, prob.input_annihilator * 
            (transpose(mass_inv_gs[j]) - (massd*mass_inv)*transpose(massd_inv_gs[j]) + uk*massd_inv))
    end |> sum
end
function ∂loss_massd(prob::IDAPBCProblem)
    if !isa(prob.hamd.mass_inv, FunctionApproxmiator)
        return (q,ps) -> zeros(eltype(ps), size(ps))
    else
        return (q,ps) -> first(Zygote.gradient(w->loss_massd(prob,q,w), ps))
    end
end

function loss_ped(prob::IDAPBCProblem, q, θ=prob.init_params)
    ped_ps = getindex(θ, prob.ps_index[:potential])
    ped_gs = vec(prob.hamd.jac_pe(q, ped_ps))
    pe_gs = prob.ham.jac_pe(q)[:]
    massd_inv_ps = getindex(θ, prob.ps_index[:mass_inv])
    massd = inv(prob.hamd.mass_inv(q,massd_inv_ps))
    mass_inv = prob.ham.mass_inv(q)
    abs(dot( prob.input_annihilator, pe_gs - (massd*mass_inv)*ped_gs ))
end
∂loss_ped(prob::IDAPBCProblem) = (q,ps) -> ReverseDiff.gradient(w -> loss_ped(prob,q,w), ps)

###############################################################################

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

function zeroexcept!(vec::Vector, target, keypairs)
    for (key,val) in keypairs
        if key != target
            vec[val] .= zero(eltype(vec))
        end
    end
end

###############################################################################

abstract type IDAPBCLoss end
abstract type PDELoss <: IDAPBCLoss end
(l::PDELoss)(x, paramvec) = l.J(x, paramvec)
gradient(l::PDELoss, x, paramvec) = l.∂J(x, paramvec)

###############################################################################

struct PDELossMd{TP,F1,F2} <: PDELoss
    prob::TP
    J::F1
    ∂J::F2
end
function PDELossMd(prob::IDAPBCProblem)
    ℓ1  = loss_massd
    ∂ℓ1 = ∂loss_massd(prob)
    J(x, paramvec) = ℓ1(prob,x,paramvec) 
    ∂J(x, paramvec) = ∂ℓ1(x,paramvec)
    return PDELossMd{typeof(prob),typeof(J),typeof(∂J)}(prob, J, ∂J)
end

struct PDELossVd{TP,F1,F2} <: PDELoss
    prob::TP
    J::F1
    ∂J::F2
end
function PDELossVd(prob::IDAPBCProblem)
    ℓ2(q,θ) = loss_ped(prob,q,θ)
    ∂ℓ2 = (q,θ) -> ReverseDiff.gradient(w->ℓ2(q,w), θ)
    J(x, paramvec) = ℓ2(x,paramvec)
    ∂J(x, paramvec) = ∂ℓ2(x,paramvec)
    return PDELossVd{typeof(prob),typeof(J),typeof(∂J)}(prob, J, ∂J)
end

struct PDELossCombined{TP,F1,F2} <: PDELoss
    J::F1
    ∂J::F2
end
function PDELossCombined(prob::IDAPBCProblem)
    ℓ1 = PDELossMd(prob)
    ℓ2 = PDELossVd(prob)
    J(x, paramvec) = ℓ1(x,paramvec) + ℓ2(x,paramvec)
    ∂J(x, paramvec) = gradient(ℓ1,x,paramvec) + gradient(ℓ2,x,paramvec)
    return PDELossCombined{typeof(prob),typeof(J),typeof(∂J)}(J, ∂J)
end

function optimize!(loss::PDELoss, paramvec, data; η=0.001, batchsize=64, maxiters=1e4, tol=1e-4)
    batchgs = Vector{typeof(paramvec)}(undef, batchsize)
    dataloader = Data.DataLoader(data; batchsize=batchsize, shuffle=true)
    max_batch = round(Int, dataloader.imax / dataloader.batchsize, RoundUp)
    optimizer = ADAM(η)
    nepoch = 0; 
    train_loss = 1/length(data) * pmap(x->loss(x,paramvec), data)
    while nepoch < maxiters && train_loss > tol
        estatus("PDE", nepoch, train_loss, maxiters)
        nbatch = 1
        for batch in dataloader
            npoints = length(batch)
            batchloss = 1/npoints * pmap(x->loss(x,paramvec), batch)
            bstatus(nbatch, max_batch, batchloss)
            pmap!(batchgs, x->gradient(loss,x,paramvec), batch)
            grads = 1/npoints * sum(batchgs[1:npoints])
            if isa(loss, PDELossVd)
                zeroexcept!(grads, :potential, prob.ps_index)
            end
            if !any(isnan.(grads))
                Optimise.update!(optimizer, paramvec, grads)
                nbatch += 1
            end
        end
        nepoch += 1
        train_loss = 1/length(data) * pmap(x->loss(x,paramvec), data)
    end
    estatus("PDE", nepoch, 1/length(data) * pmap(x->loss(x,paramvec), data), maxiters)
end

###############################################################################

struct ConvexVdLoss{TP,F} <: IDAPBCLoss
    prob::TP
    J::F
end
function ConvexVdLoss(prob::IDAPBCProblem, inputJ::Function)
    Vd = prob.hamd.potential
    Nθ = length(Vd.θ)
    Lidx = coeff_matrix(Vd,1:Nθ)
    Ridx = Lidx + Lidx' - Diagonal(diag(Lidx))
    N = length(Ridx)
    
    # Compute J(x) such that J*P =  ∇x_Vd, where P are the elements of the Q matrix of Vd.
    J(x) = begin
        # Compute the coefficients corresponding to each element of the parameter matrix Q of
        # the gradient of a SOS polynomial Vd = m(x)ᵀQm(x). The desired gradient is
        # ∇x_Vd = 2 * (∂m/∂x)ᵀ * Q * m(x). It is linear in elements of Q. These coefficients can be 
        # obtained by C = ∂m/∂x * m(x)ᵀ, which we compute below. 
        coeffs = transpose(inputJ(x)) * reduce(vcat, 
            [reshape(2*v*monsub(Vd,x)',(1,N)) for v in eachcol(∂mon∂q(Vd, x))]
        )
        # Since Q is symmetric, we can combine some elements of the matrix C into fewer terms
        # This is collected into the matrix R
        R = zeros(eltype(coeffs), (size(coeffs,1),Nθ))
        for (row,C) in enumerate(eachrow(coeffs))
            for (c, θidx) in zip(C, Ridx)
                R[row,θidx] += c
            end
        end
        return R
    end 
    return ConvexVdLoss{typeof(prob),typeof(J)}(prob, J)
end
(l::ConvexVdLoss)(x) = l.J(x)

function optimize!(loss::ConvexVdLoss, paramvec, data)
    Vd = loss.prob.hamd.potential
    Mdθ = getindex(paramvec, loss.prob.ps_index[:mass_inv])
    Md(x) = inv(loss.prob.hamd.mass_inv(x, Mdθ))
    M_inv = loss.prob.ham.mass_inv
    Gperp = loss.prob.input_annihilator
    A = mapreduce(x->Gperp*(-Md(x)*M_inv(x)*loss.J(x)), vcat, data)
    B = mapreduce(x->Gperp*(-loss.prob.ham.jac_pe(x)[:]), vcat, data)
    # return A, B
    # res = A \ B
    # Q = vec2tril(res) + vec2tril(res)' - Diagonal(diag(vec2tril(res)))
    # try
    #     L = cholesky(Q + eltype(paramvec)(1-4)*I).L  # may fail due to non PSD-ness.
    #     paramvec[loss.prob.ps_index[:potential]] .= tril2vec(L)
    #     return A*res - B
    # catch e
    #     if isa(e, PosDefException)
    #         @warn "Could not perform Cholesky decomposition."
    #         return Q, A*res - B
    #     else
    #         throw(e)
    #     end
    # end
    m = length(Vd.mono)
    R = Variable((m,m))
    idx = tril2vec_perm(m)
    p = minimize(sumsquares(A*R[idx] - B), isposdef(R))
    solver = () -> Mosek.Optimizer()
    solve!(p, solver)
    if p.status == MosekTools.MathOptInterface.OPTIMAL
        Q = evaluate(R)
        L = cholesky(Q).L
        paramvec[loss.prob.ps_index[:potential]] .= tril2vec(L)
        nothing
    else
        @warn "Optimizer did not return OPTIMAL status"
    end
end

function test(loss::ConvexVdLoss)
    Vd = loss.prob.hamd.potential
    JVd = loss.prob.hamd.jac_pe
    θ = rand(length(loss.prob.ps_index[:potential]))
    L = coeff_matrix(Vd, θ)
    tril_idx = filter(collect(Iterators.product(1:size(L,1), 1:size(L,2)))) do x
        first(x) <= last(x)
    end
    Q = L*L'
    Qvec = [Q[CartesianIndex(i,j)] for (i,j) in tril_idx]
    x = rand(length(Vd.mono))
    @assert isapprox(loss.J(x)*Qvec, JVd(x, θ)[:], atol=1e-6)
end
