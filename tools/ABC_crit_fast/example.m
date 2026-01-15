clear all
clc
close all

N = 100;
T = 200;
r = 5;              % number of factors
psi = 0.7;          % parameter for factors persistence
alpha = 0.4;        % parameter for idio cross-corr
phi = 0.3;          % parameter for idio serial corr
theta = 0.5;        % var(idio)/var(common)

%% simulate common component
L = rand(N,r);                                                              % loadings

u = randn(T,r);                                                             % the factors follow a stationary VAR(1) with max eigenvalue psi
f(1,:) = zeros(1,r);
A = rand(r,r);
A = A./max(eig(A))*psi; 

for t =1:T-1
    f(t+1,:)=f(t,:)*A+u(t+1,:);
end

chi = f*L';

%% simulate idio component
for k=1:N
    c(1,k) = alpha^(k-1);
end
S_xi_N = toeplitz(c);

for k=1:T
    b(1,k) = phi^(k-1);
end
S_xi_T = toeplitz(b);

xi = chol(S_xi_T)'*randn(T,N)*chol(S_xi_N)';

%% simulate data
chi = chi./(ones(T,1)*std(chi));
xi = sqrt(theta)*(xi./(ones(T,1)*std(xi)));

x = chi + xi;


%% estimate number of factors with ABC criterion
kmax = 10;
nbck = floor(N/10);
cmax = 3;
graph = 1;
[rhat1 rhat2] = ABC_crit(x, kmax, nbck, cmax, graph);

disp(sprintf(' True number of factors %d', r));
disp(sprintf(' Estimated number of factors with large window %d', rhat1));
disp(sprintf(' Estimated number of factors with small window %d', rhat2));
disp(sprintf(' Estimated number of factors average %f', (rhat1+rhat2)/2));





