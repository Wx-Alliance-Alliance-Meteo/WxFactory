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

%alp = sqrt(1e-7);
V(:,1) = r / norm(r);
L = eye(m+size(C,2), m+size(C,2));
L(1:size(C,2), 1:size(C,2)) = tril(C'*C);
T = L(1:size(C,2), 1:size(C,2));
for i=1:size(C,2)
L(i,i) =1;
end

min_it = 1;
num_it = 1;

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
        col1(ii) = WW(:, ii)'*V(:,k);
        rv(ii) = WW(:,ii)'*w;
    end
    if size(col1,2)>size(col1,1)
    col1=col1'; rv = rv';
    end

col1 = WW(:,1:kk)'*V(:,k);
rv = WW(:, 1:kk)'*w; 
L(size(C,2)+k, 1:kk) = col1';
L(kk,kk) = 0; j = size(C,2)+1;
 
 V(:,k+1) = w - WW(:, 1:kk)*(rv);
 %size(V(:,k+1))
 %fprintf('Here\n');
 ll(1:j-1) = rv(1:j-1); 
 ll(j:kk) = 1.0e-7 * ( eye(kk-j+1,kk-j+1) + L(j:kk, j:kk) ) \ ( rv(j:kk) - L(j:kk, 1:j-1)*rv(1:j-1) );
 
 H(1:k, k) = ll(size(C,2)+1:end);
 B(:,k) = rv(1:size(C,2));
    H(k+1,k) = norm(V(:,k+1));
    V(:,k+1) = V(:,k+1)/norm(V(:,k+1));
    %fprintf('V: %g \n', norm(eye(k+1,k+1) - V(:,1:k+1)'*V(:,1:k+1), 'Fro') );
    %fprintf('W: %g \n', norm(eye(kk,kk) - WW(:,1:kk)'*WW(:,1:kk), 'Fro') );
    
    % this is where the stuff ends
    
    % Initialize right hand side of least-squares system
    rhs = zeros(k+1,1);
    rhs(1) = norm(r);
    
    % Solve least squares system; Calculate residual norm    
    y = H \ rhs;  
    %D = eig(H(1:k,1:k))
    %figure(3)
    %semilogy(D,'+','LineWidth',2)
    %hold on
    tr = norm(r);
    res = rhs - H * y;
    resvec(k) = norm(res);
    
    if resvec(k) < tol
        x = x + V(:,1:k) * y;
        r = V * res;
        trueres = norm(r) / tr
        if (num_it >= min_it)
            return
        end
    end

    num_it = num_it + 1;
end
