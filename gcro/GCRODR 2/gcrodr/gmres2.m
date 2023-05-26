% Arnoldi Iteration
%
% Generates relation (I - C*C') V(:,1:m) = V(:,1:m+1) H
%
% INPUT:  A      N-by-N matrix
%         X      current solution vector
%         R      N-by-1 preconditioned residual vector
%         M      number of GMRES iterations to perform
%         M1     left preconditioner for A
%         M2     right preconditioner for A
%         C
%         tol    specifies the tolerance of the method
% OUTPUT: V      N-by-M+1 matrix containing orthogonal basis for Krylov subspace
%         H      M+1-by-M upper Hessenburg reduction of matrix operator
%         B      the matrix C'*A*V(:,1:k)
%         K      number of GMRES iterations actually performed
%         RESVEC vector containing norm of residual at each iteration
function [V,H,B,k,resvec] = gmres2(A,x,r,m,M1,M2,C,tol)

if(isempty(M1))
    existM1 = 0;
else
    existM1 = 1;
end
if(isempty(M2))
    existM2 = 0;
else
    existM2 = 1;
end

% Initialize V

%alp = sqrt(0.5);
V(:,1) = r / norm(r);
L = eye(m+size(C,2), m+size(C,2));
L(1:size(C,2), 1:size(C,2)) = tril(C'*C);
T = L(1:size(C,2), 1:size(C,2));
for i=1:size(C,2)
L(i,i) =1;
end

wt = 1;


for k = 1:m
    % Find w using preconditioning if available.
    if(existM2)
        w = M2 \ V(:,k);
    else
        w = V(:,k);
    end
    if (isreal(w))
        Aw = double(py.integrators.ros2.ros2matvec(w)).';
    else
        w_r = py.numpy.array(real(w));
        w_i = py.numpy.array(imag(w));
        dual = py.integrators.ros2.ros2matvec_complex(w_r, w_i);
        new_r = double(dual{1}).';
        new_i = double(dual{2}).';
        Aw = complex(new_r, new_i);
    end
    w = Aw;
    if(existM1)
        w = M1 \ w;
    end
    %%% KS: replace this stuff ***
    % Create next column of V and H
    
    % Apply (I-C*C') operator to Aw
    %      B(:,k) = C' * w;
    %      w = w - C * B(:,k);
    
    WW = [C V(:, 1:k)];
    kk = size(C,2)+k;

    
    for ii=1:kk
        col1(ii) = WW(:, ii)'*V(:,k)*wt;
        rv(ii) = WW(:,ii)'*w*wt;
    end
    if size(col1,2)>size(col1,1)
    col1=col1'; rv = rv';
    end

col1 = WW(:,1:kk)'*V(:,k)*wt;   % r2 = V^Tu - he last row of the L matrix in T = (I + L)^-1
rv = WW(:, 1:kk)'*w*wt;         % r0 = W^Tw.    
L(size(C,2)+k, 1:kk) = col1';
L(kk,kk) = 0; j = size(C,2)+1;
 
%V(:,k+1) = w - WW(:, 1:kk)*(rv); %old version

 ll(1:j-1) = rv(1:j-1); 
 ll(j:kk) = wt*( eye(kk-j+1,kk-j+1) + L(j:kk, j:kk) ) \ ( rv(j:kk) - L(j:kk, 1:j-1)*rv(1:j-1) );
 r1 = ll(1:kk)'; %<=====. r1 = T W^T r0 
 V(:,k+1) = w - WW(:, 1:kk)*(r1); %<==== V = w - W r1. Gauss-Seidel MG
 
 r2 = WW(:, 1:kk)'*V(:,k+1)*wt; % second projection. (CGS - Jacobi step)
 %r3 = ll(1:kk)';
 %r3(1:j-1) = rv(1:j-1); 
 %r3(j:kk) = ( eye(kk-j+1,kk-j+1) + L(j:kk, j:kk) ) \ ( r2(j:kk) - L(j:kk, 1:j-1)*r2(1:j-1) );
 V(:,k+1) = V(:,k+1) - WW(:, 1:kk)*(r2);

 H(1:k, k) = ll(size(C,2)+1:end) + r2(size(C,2)+1:end)'; % update the Hessenberg where r2 is small correction
 B(:,k) = rv(1:size(C,2));
    H(k+1,k) = norm(V(:,k+1));
    V(:,k+1) = V(:,k+1)/(norm(V(:,k+1)));
    %fprintf('V: %g \n', norm(eye(k+1,k+1) - V(:,1:k+1)'*V(:,1:k+1), 'Fro') );
    %fprintf('W: %g \n', norm(eye(kk,kk) - WW(:,1:kk)'*WW(:,1:kk), 'Fro') );
    
    % this is where the stuff ends
    
    % Initialize right hand side of least-squares system
    rhs = zeros(k+1,1);
    rhs(1) = norm(r);
    
    % Solve least squares system; Calculate residual norm    
    y = (1/wt)*(H \ rhs); 
    %x = x + V(:,1:k) * y;
    %tr = norm(b - A*x) / norm(b)
    tr = norm(r);
    res = rhs - H * y;
    resvec(k) = norm(res);
    if resvec(k) < tol
        x = x + V(:,1:k) * y;
        %tr = norm(b - A*x) / norm(b);
        r = V * res;
        %trueres = norm(r) / tr;
        return
    end
end
