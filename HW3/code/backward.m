function [log_beta] = backward(observations, A, mu, sigma)
% backward recursion / beta-message function / filtering
%
% Inputs:
%   observations: T x d observations array with d dimenstion of data and T
%   time steps (u_1, u_2, ..., u_T).
%   A: K x K transition matrix between different states
%   mu/sigma: parameters for conditional probabilities 
%             u_t|t_t = i ~ N(mu_i, sigma_i) for i = 1, ..., K
%             mu: K x d, sigma: d x d x K
% Outputs:
%   log_beta: log beta-message T x K matrix

[T, d] = size(observations);
K = length(A);


log_beta = zeros(T, K);

for t = (T-1):-1:1
    log_gaussian = log(mvnpdf(repmat(observations(t+1, :), K, 1), mu, sigma))';
    log_beta(t, :) = logsumexp(repmat(log_gaussian + log_beta(t+1, :), K, 1) + log(A'))';
end

end