% ABC_Criteria - Determines the Number of Common Factors in the Static
% Approximate Factor Model
% 
%  This function implement the information criteria described in Alessi,
%  Barigozzi and Capasso (2010) 
% 
% [rhat1 rhat2]=ABC_crit(X, kmax, nbck, cmax)
% 
% Inputs:
% 	   X - (T x n) stationary data matrix
%   kmax - maximum number of factors
%   nbck - number of sublocks to be used (default floor(n/10))
%   cmax - maximum value for the penalty constant (default = 3) 
%   graph - if set to 1 show graphs as in the paper (default = 0)
% Outputs:
%  rhat1 - determines the number of shocks using a large window
%  rhat2 - determines the number of shocks using a small window
% 
% Written by Matteo Barigozzi 
% 
% Reference: Alessi, L., M. Barigozzi, and M. Capasso (2010). 
% Improved penalization for determining the number of factors 
% in approximate static factor models. 
% Statistics and Probability Letters 80, 1806?1813.

function [rhat1 rhat2] = ABC_crit(X, kmax, nbck, cmax, graph)

npace=1; 
step=500; 

[T,n] = size(X);                                                            % Size of the datatset

if nargin < 2
    disp(sprintf('Too few input arguments, must provide a value for kmax'))
elseif nargin == 2
    nbck = floor(n/10);
    cmax = 3;
    graph = 0;
elseif nargin == 3
    cmax = 3;
    graph = 0;
elseif nargin == 4
    graph = 0;
end

x = (X - ones(T,1)*mean(X))./(ones(T,1)*std(X,1));
    
% Running the Criteria %%
s=0;
for N = n-nbck:npace:n
    s=s+1; 
    [~, Ns]=sort(rand(n,1));
    xs = x(1:T,Ns(1:N));
    xs = (xs - ones(T,1)*mean(xs))./(ones(T,1)*std(xs,1));
    eigv = flipud(eig(cov(xs)));                                            % Eigenvalues of the Covariance Matrix
    
    for k=1:kmax+1
        IC1(k,1) = sum(eigv(k:N)); 
    end
    
    p = ((N+T)/(N*T))*log((N*T)/(N+T));                                     % penalty
    
    T0=repmat((0:kmax)',1).*p;
    for c = 1:floor(cmax*step);            
        cc = c/step;
        IC = (IC1./N) + T0*cc;   [~, rr]=min(IC);                           % criterion            
        abc(s,c)=rr-1;                                  
    end
end    

%%% ----------------------------------------- %%%
%%% Select Automatically the Number of Shocks %%%
%%% ----------------------------------------- %%%

cr=(1:floor(cmax*500))'/500;

for ll=1:2; 
    ABC(1,1)=kmax;
    ABC(1,2:3)=0;
end
        
sabc = std(abc);
c1=2;
for ii=1:size(cr,1);
    if sabc(1,ii)==0;                                                       % If the number of factors is always the same across sub-blocks
        if abc(end,ii)==ABC(c1-1,1);
            ABC(c1-1,3)=cr(ii);
        else
            ABC(c1,1)=abc(end,ii);
            ABC(c1,2:3)=cr(ii);
            c1=c1+1;
        end;
    end;
end;
ABC(:,4) = ABC(:,3)-ABC(:,2);                                               % Computes Window Size
q = ABC(find(ABC(2:end,4)>.05)+1,1);                                         % Number of Factors with Large Window
rhat1 = q(1);                                                               
q = ABC(find(ABC(2:end,4)>.01)+1,1);                                        % Number of Shocks with Small Window
rhat2 = q(1);                                              

if graph == 1
    set(0,'DefaultLineLineWidth',1.5);
    figure
    plot(cr,abc(end,:),'r-')
    axis tight
    hold all
    plot(cr,5*sabc,'b--')
    xlabel('c')
    axis tight
    xlim([0 1])
    legend('r^{*T}_{c;N}','S_c')
    title('ABC estimated number of factors')
    grid on
end
