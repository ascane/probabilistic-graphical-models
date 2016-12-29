function [states] = viterbi(observations, pi, A, mu, sigma)
% Viterbi decoding algorithm

[T, d] = size(observations);
K = length(A);
states = zeros(1, T);

% forward recursion
log_alpha = zeros(T, K);
log_alpha(1, :) = log(pi) + log(mvnpdf(repmat(observations(1, :), K, 1), mu, sigma))';
indices = zeros(T, K);

for t = 2:T
    [log_alpha(t, :), indices(t, :)] = max(repmat(log_alpha(t - 1, :), K, 1) + log(A), [], 2);
    log_alpha(t, :) = log_alpha(t, :) + log(mvnpdf(repmat(observations(t, :), K, 1), mu, sigma))';
end

[~, k] = max(log_alpha(T,:));
states(T) = k;

% backward recursion

for t = (T-1):-1:1
    states(t) = indices(t+1, states(t+1));
end


end