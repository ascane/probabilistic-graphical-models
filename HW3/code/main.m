%% load data
data = importdata('classification_data_HWK3/EMGaussian.data');
test_data = importdata('classification_data_HWK3/EMGaussian.test');

[T, d] = size(test_data);
K = 4; 

pi = ones(1, K) / K;

mu = [-2.0344    4.1726
    3.9779    3.7735
    3.8007   -3.7972
   -3.0620   -3.5345];

sigma(:,:,1) = [2.9044    0.2066
                0.2066    2.7562];
sigma(:,:,2) = [0.2104    0.2904
                0.2904   12.2392];
sigma(:,:,3) = [0.9213    0.0574
                0.0574    1.8660];
sigma(:,:,4) = [6.2414    6.0502
                6.0502    6.1825];

A = ones(K, K)/6;
A(1:(K+1):end) = 1/2;

%% Q2 Data smoothing
log_alpha = forward(test_data, pi, A, mu, sigma);
log_beta = backward(test_data, A, mu, sigma);

smooth_states = log_alpha + log_beta;
log_likelihood = logsumexp(smooth_states(1,:));
 
smooth_states = exp(smooth_states - log_likelihood);

% plot 
figure

for k = 1:K
    subplot(4,1,k);
    plot(smooth_state(1:100, k));
end

% for t = 1:T-1
%     log_gaussian = log(mvnpdf(repmat(test_data(t+1, :), K, 1), mu, sigma));
% %         repmat(log_alpha(t, :), K, 1)
% %         repmat((log_beta(t+1, :)+loggaussian)', 1, K)
%     smooth_lag_one(:,:,t) = repmat(log_alpha(t, :), K, 1) ...
%         + repmat(log_beta(t+1, :)'+log_gaussian, 1, K) + log(A) ...
%         - log_likelihood;
% end
% smooth_lag_one = exp(smooth_lag_one);
% A = sum(smooth_lag_one, 3);
% A = A./ repmat(sum(A,1), K, 1);
% 
% for k = 1:K
%     mu(k, :) = sum(test_data.*repmat(smooth_states(:, k), 1, d), 1) ...
%         / sum(smooth_states(:, k));
%     sigma_tilde = zeros(d,d);
%     for t = 1:T
%         centered_obs = test_data(t, :)-mu(k, :);
%         aux = centered_obs'*centered_obs;
%         sigma_tilde = sigma_tilde+ aux * smooth_states(t, k);
%     end
%     sigma(:,:,k) = sigma_tilde/sum(smooth_states(:, k));
% end


%% Q3 Parameters estimation using EM algorithm
n_iters = 10;
[new_pi, new_A, new_mu, new_sigma, loglik] = EM(data, pi, A, mu, sigma, n_iters);

% plot log-likelihood
figure
plot(loglik)

%% Viterbi decoding algorithm



