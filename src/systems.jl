export ParametricControlSystem, trajectory

struct ParametricControlSystem{
    IIP,
    DIM,
    S,
    EOM,
    POL
}
    f::EOM  # f(x,u) or f!(dx,x,u)
    u::POL  # u(x,θ) or u(x)
    x0::S
end

const PCS = ParametricControlSystem

function ParametricControlSystem{iip}(f::F, u::U, N; x0=rand(N)) where {iip,F<:Function,U<:Function}
    ParametricControlSystem{iip,N,typeof(x0),typeof(f),typeof(u)}(f,u,x0)
end

create_closedloop(sys::PCS{true}) = (dx,x,p,t) -> sys.f(dx, x, sys.u(x,p))
create_closedloop(sys::PCS{false}) = (x,p,t) -> sys.f(x, sys.u(x,p))

function SciMLBase.ODEProblem(sys::PCS{iip}, tspan; x0=sys.x0) where {iip}
    f = create_closedloop(sys)
    ode = ODEProblem{iip}(f, x0, tspan)
end
function SciMLBase.ODEProblem(sys::PCS{iip}, θ, tspan; x0=sys.x0) where {iip}
    f = create_closedloop(sys)
    ode = ODEProblem{iip}(f, x0, tspan, θ)
end

function trajectory(integ::ODEProblem, x0; odekwargs...)
    Array(solve(remake(integ, u0=x0), Tsit5(); odekwargs...))
end
function trajectory(integ::ODEProblem, x0, θ; odekwargs...)
    Array(solve(remake(integ, u0=x0), Tsit5(), p=θ; odekwargs...))
end

