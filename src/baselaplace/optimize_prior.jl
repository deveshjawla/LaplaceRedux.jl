"""
	optimize_prior!(
		la::AbstractLaplace; 
		n_steps::Int=100, lr::Real=1e-1,
		λinit::Union{Nothing,Real}=nothing,
		σinit::Union{Nothing,Real}=nothing
	)
	
Optimize the prior precision post-hoc through Empirical Bayes (marginal log-likelihood maximization).
"""
function optimize_prior!(
	la::AbstractLaplace;
	n_steps::Int = 100,
	lr::Real = 1e-1,
	λinit::Union{Nothing, Real} = nothing,
	σinit::Union{Nothing, Real} = nothing,
	verbosity::Int = 0,
	tune_σ::Bool = la.likelihood == :regression,
)

	# Setup:
	logP₀ = isnothing(λinit) ? log.(unique(diag(la.prior.prior_precision_matrix))) : log.([λinit])   # prior precision (scalar)
	logσ = isnothing(σinit) ? log.([la.prior.observational_noise]) : log.([σinit])                 # noise (scalar)
	opt = Adam(lr)
	loss(P₀, σ) = -log_marginal_likelihood(la; P₀ = P₀[1], σ = σ[1])

	# opt = OptimiserChain(WeightDecay(), AdaBelief())
	show_every = round(n_steps / 10)
	i = 0
	if tune_σ
		@assert la.likelihood == :regression "Observational noise σ tuning only applicable to regression."
		ps = (logP₀, logσ)
	else
		if la.likelihood == :regression
			@warn "You have specified not to tune observational noise σ, even though this is a regression model. Are you sure you do not want to tune σ?"
		end
		ps = (logP₀)
	end
	opt_state = Flux.setup(opt, ps)

	# Optimization:
	while i < n_steps
		if tune_σ
			l, gs = Flux.withgradient(p -> loss(exp.(p[1]), exp.(p[2])), ps)
		else
			l, gs = Flux.withgradient(p -> loss(exp.(p[1]), exp.(logσ)), ps)
		end
		Flux.update!(opt_state, ps, gs[1])
		i += 1
		if verbosity > 0
			if i % show_every == 0
				@info "Iteration $(i): P₀=$(exp(logP₀[1])), σ=$(exp(logσ[1]))"
				@show l
				println("Log likelihood: $(log_likelihood(la))")
				println("Log det ratio: $(log_det_ratio(la))")
				println("Scatter: $(_weight_penalty(la))")
			end
		end
	end
end
