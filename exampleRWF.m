%% Example of the RWF and IRWF algorithm under 1D Gaussian designs
% The code below is adapted from implementation of the TWF desinged by Y. Chen and E. Candes, Wirtinger Flow algorithm designed and implemented by E. Candes, X. Li, and M. Soltanolkotabi
clear;
%% Set Parameters
if exist('Params')                == 0,  Params.n2          = 1;    end
if isfield(Params, 'n1')          == 0,  Params.n1          = 1024; end             % signal dimension
if isfield(Params, 'm')           == 0,  Params.m           = 8* Params.n1;  end     % number of measurements
if isfield(Params, 'cplx_flag')   == 0,  Params.cplx_flag   = 0;    end             % real: cplx_flag = 0;  complex: cplx_flag = 1;
if isfield(Params, 'grad_type')   == 0,  Params.grad_type   = 'TWF_Poiss';  end     % 'TWF_Poiss': Poisson likelihood

if isfield(Params, 'alpha_lb')    == 0,  Params.alpha_lb    = 0.5;  end
if isfield(Params, 'alpha_ub')    == 0,  Params.alpha_ub    = 5;    end
if isfield(Params, 'alpha_h')     == 0,  Params.alpha_h     = 5;    end
if isfield(Params, 'alpha_y')     == 0,  Params.alpha_y     = 3;    end 
if isfield(Params, 'T')           == 0,  Params.T           = 150;  end    	% number of iterations
if isfield(Params, 'mu')          == 0,  Params.mu          = 0.2;  end		% step size / learning parameter 0.3 is best for twf, 0.2 is best fwf; 0.12 is best for nesterov
if isfield(Params, 'npower_iter') == 0,  Params.npower_iter = 30;   end		% number of power iterations

n           = Params.n1;    
m           = Params.m;         
cplx_flag	= Params.cplx_flag;  % real-valued: cplx_flag = 0;  complex-valued: cplx_flag = 1;    
display(Params)
        
%% make the data


x = randn(n,1)+ cplx_flag * 1i * randn(n,1);
xnorm=norm(x)

Amatrix = (randn(m,n)+ cplx_flag * 1i * randn(m,n)) / (sqrt(2)^cplx_flag);
A  = @(I) Amatrix  * I;
At = @(Y) Amatrix' * Y;
y  = abs(A(x)); % y_i=|a_i x|

%% Initialization
npower_iter = Params.npower_iter;           % Number of power iterations 
z0 = randn(n,Params.n2); z0 = z0/norm(z0,'fro');    % Initial guess 
%normest = sqrt(pi/2)*sum(y(:))/numel(y(:));
    tol = 1e-14;

 normest = (sqrt(pi/2)*(1-Params.cplx_flag)+sqrt(4/pi)*Params.cplx_flag)*sum(y(:))/numel(y(:))
ytr=y.* (abs(y) > 1 * normest );% truncated version
for tt = 1:npower_iter,                     
    z0 = At( ytr.* (A(z0)) ); z0 = z0/norm(z0,'fro');
end
z0 = normest * z0;                   % Apply scaling

%% reshaped Wirtinger flow
Relerrs=zeros(Params.T+1,1);
z=z0;
Relerrs(1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error

mu=0.8+0.4*cplx_flag;% real step size 0.8/ complex step size1.2
t=1;
while t<=Params.T
    yz=Amatrix*z;
 %   ang   =  Params.cplx_flag*exp(1i * angle(yz)) +(1 - Params.cplx_flag) * sign(yz);
    z = z - mu* (m\At(yz-y.*yz./abs(yz)));
  Relerrs(t+1)=norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
    t=t+1;
end
T = Params.T;
fprintf('Relative error after initialization: %f\n', Relerrs(1))
fprintf('RWF Relative error after %d iterations: %14f\n', t, Relerrs(t))
 
figure, h1=semilogy(0:T,Relerrs,'-r'); 
xlabel('Iteration'), ylabel('Relative error (log10)'), ...
     title('Relative error vs. iteration count')
 hold on;
%% --- minibatch Incremental Reshaped Wirtinger Flow (IRWF) ---%
z=z0;
Relerrs=zeros(Params.T+1,1);

Relerrs(1) = norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro'); % Initial rel. error
mu=1;
sgd=Params.m;
batch=64; %batch=1 if for IRWF, 
tic;
for t = 1: Params.T,
        for i=1:batch:sgd-batch+1

            Asub=Amatrix(i:i+batch-1,:);
            ysub=y(i:i+batch-1);
            Asubz=Asub*z;
            z=z-(Asub'*(Params.n1\(Asubz-ysub.*Asubz./abs(Asubz))));
        end
  Relerrs(t+1)=norm(x - exp(-1i*angle(trace(x'*z))) * z, 'fro')/norm(x,'fro');
       
end
h2=semilogy(0:T,Relerrs,'-b');
hold off;