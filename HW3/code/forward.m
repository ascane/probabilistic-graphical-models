function [log_alpha] = forward(observations, init_dist, A, mu, sigma)
% forward recursion / alpha-message function / filtering
%
% Inputs:
%   observations: T x d observations array with d dimenstion of data and T
%   time steps (u_1, u_2, ..., u_T).
%   init_dist: 1 x K initial distribution for states (q_t)
%   A: K x K transition matrix between different states
%   mu/sigma: parameters for conditional probabilities 
%             u_t|t_t = i ~ N(mu_i, sigma_i) for i = 1, ..., K
%             mu: K x d, sigma: d x d x K
% Outputs:
%   log_alpha: log alpha-message T x K matrix

[T, d] = size(observations);
K = length(A);

log_alpha = zeros(T, K);
% initial values
log_alpha(1, :) = log(init_dist) + log(mvnpdf(repmat(observations(1, :), K, 1), mu, sigma))';

for t = 2:T
    log_alpha(t, :) = log(mvnpdf(repmat(observations(t, :), K, 1), mu, sigma))';
    log_alpha(t, :) = log_alpha(t, :) + logsumexp(repmat(log_alpha(t - 1, :), K, 1) + log(A))';
end


end