function [res] = logsumexp(x)
% compute log sum exp x_i for small value x
[~, d] = size(x);
a = max(x,[],2);
res = a + log( sum( exp( x - repmat(a,1,d) ) , 2 ) ) ;
end