function [pi, A, mu, sigma, loglik, loglik_test] = EM(observations, init_dist, A, mu, sigma, n_iters, test_data)
% EM algorithm to learn HMM parameters
%
% Inputs:
%   observations: T x d observations array with d dimenstion of data and T
%   time steps (u_1, u_2, ..., u_T).
%   init_dist: 1 x K initial distribution for states (q_0)
%   A: K x K initial transition matrix between different states
%   mu/sigma: initial parameters for conditional probabilities 
%             u_t|t_t = i ~ N(mu_i, sigma_i) for i = 1, ..., K
%             mu: K x d, sigma: d x d x K
% Outputs:
%   

[T, d] = size(observations);

K = length(A);

pi = init_dist;

smooth_lag_one = zeros(K, K, T-1);

loglik = zeros(1,n_iters);
loglik_test = zeros(1,n_iters);

for i = 1:n_iters
    % E step: expectation
    % we use alpha/beta-recursion i = 1:n_iters
    log_alpha = forward(observations, pi, A, mu, sigma);
    log_beta = backward(observations, A, mu, sigma);
    smooth_states = log_alpha + log_beta;
    log_likelihood_obs = logsumexp(smooth_states(1,:));
    smooth_states = exp(smooth_states - log_likelihood_obs);
    
    
    for t = 1:T-1
        log_gaussian = log(mvnpdf(repmat(observations(t+1, :), K, 1), mu, sigma));
%         repmat(log_alpha(t, :), K, 1)
%         repmat((log_beta(t+1, :)+loggaussian)', 1, K)
        smooth_lag_one(:,:,t) = repmat(log_alpha(t, :), K, 1) ...
            + repmat(log_beta(t+1, :)'+log_gaussian, 1, K) + log(A) ...
            - log_likelihood_obs;
    end
    smooth_lag_one = exp(smooth_lag_one);
    
    loglik(i) = log_likelihood_obs;
    
    if nargin == 7
        log_alpha_test = forward(test_data, pi, A, mu, sigma);
        log_beta_test = backward(test_data, A, mu, sigma);
        smooth_states_test = log_alpha_test + log_beta_test;
        log_likelihood_test = logsumexp(smooth_states_test(1,:));
        loglik_test(i) = log_likelihood_test;
    end
    
    
    % M step: maximize complete log-likelihood
    pi = smooth_states(1, :)./sum(smooth_states(1, :));
    A = sum(smooth_lag_one, 3);
    A = A./ repmat(sum(A,1), K, 1);
    
    for k = 1:K
        mu(k, :) = sum(observations.*repmat(smooth_states(:, k), 1, d), 1) ...
            / sum(smooth_states(:, k));
        sigma_tilde = zeros(d,d);
        for t = 1:T
            centered_obs = observations(t, :)-mu(k, :);
            aux = centered_obs'*centered_obs;
            sigma_tilde = sigma_tilde+aux * smooth_states(t, k);
        end
        sigma(:,:,k) = sigma_tilde/sum(smooth_states(:, k));
    end
    
    
end



end