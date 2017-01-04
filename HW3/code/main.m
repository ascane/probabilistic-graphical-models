%% load data
train_data = importdata('classification_data_HWK3/EMGaussian.data');
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

lw=3; %Linewidth
fs=15; %Fontsize
fw='Bold'; %FontWeight
fsa=11; %Fontsize

%% Q2 Data smoothing using initial parameters
log_alpha_init = forward(test_data, pi, A, mu, sigma);
log_beta_init = backward(test_data, A, mu, sigma);

marginal_init = log_alpha_init + log_beta_init;
loglik_init = logsumexp(marginal_init(1,:));
 
marginal_init = exp(marginal_init - loglik_init);

% plot 
figure

for k = 1:K
    subplot(4,1,k);
    plot(marginal_init(1:100, k), 'LineWidth', lw);
    title(sprintf('state %d', k), 'FontWeight', fw, 'FontSize', fs)
    ylabel('probability', 'FontWeight', fw, 'FontSize', fs)
end
xlabel('time', 'FontWeight', fw, 'FontSize', fs)


%% Q4,Q5: Parameters estimation using EM algorithm
n_iters = 10;
[new_pi, new_A, new_mu, new_sigma, loglik_list, loglik_test_list] = EM(train_data, pi, A, mu, sigma, n_iters, test_data);

% plot log-likelihood
figure
hold on
plot(loglik_list, 'LineWidth', lw)
plot(loglik_test_list, 'LineWidth', lw)
h = legend('train', 'test', 'Location', 'best');
set(h, 'FontSize', fsa)
xlabel('iterations', 'FontWeight', fw, 'FontSize', fs)
ylabel('log-likelihood', 'FontWeight', fw, 'FontSize', fs)
title('Convergence of EM algorithm', 'FontWeight', fw, 'FontSize', fs)

%% Q6 Compute log-likelihoods
loglik_train = loglik_list(end)/T
loglik_test = loglik_test_list(end)/T
loglik_train = loglik_list(end)
loglik_test = loglik_test_list(end)

%% Q8 Viterbi decoding algorithm
states_train = viterbi(train_data, new_pi, new_A, new_mu, new_sigma);
states_test = viterbi(test_data, new_pi, new_A, new_mu, new_sigma);
figure
subplot(1,2,1)
hold on
gscatter(train_data(:, 1), train_data(:, 2), states_train, [], [], 15);
plot(new_mu(:, 1), new_mu(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', lw)
legend('off')
title('Train', 'FontWeight', fw, 'FontSize', fs )
subplot(1,2,2)
hold on
gscatter(test_data(:, 1), test_data(:, 2), states_test, [], [], 15);
plot(new_mu(:, 1), new_mu(:, 2), 'kx', 'MarkerSize', 15, 'LineWidth', lw)
legend('off')
title('Test', 'FontWeight', fw, 'FontSize', fs )
suptitle('Viterbi decoding algorithm')


%% Q9: Compute marginal probability for the test data with parameters learned on the training data
log_alpha_test = forward(test_data, new_pi, new_A, new_mu, new_sigma);
log_beta_test = backward(test_data, new_A, new_mu, new_sigma);
marginal_test = log_alpha_test + log_beta_test;
loglik_test = logsumexp(marginal_test(1,:));

marginal_test = exp(marginal_test - loglik_test);

figure

for k = 1:K
    subplot(4,1,k);
    plot(marginal_test(1:100, k), 'LineWidth', lw);
    title(sprintf('state %d', k), 'FontWeight', fw, 'FontSize', fs)
    ylabel('probability',  'FontWeight', fw, 'FontSize', fs )
end
xlabel('time (datapoint index)')


%% Q10: Compute most likely state according to the marginal probability
% Q11: Compute most likely sequence of states using viterbi algorithm

% most likely state according to the marginal proba
[~, marginal_states] = max(marginal_test, [], 2);
% most likely state according to Viterbi
viterbi_states = viterbi(test_data, new_pi, new_A, new_mu, new_sigma);

figure
subplot(2,1,1);
plot(marginal_states(1:100),'LineWidth', lw);
title('most likely state according to the marginal probability', 'FontWeight', fw, 'FontSize', fs)
subplot(2,1,2);
plot(viterbi_states(1:100),'LineWidth', lw);
title('most likely state according to Viterbi algorithm', 'FontWeight', fw, 'FontSize', fs)




