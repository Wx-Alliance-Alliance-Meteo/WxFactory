%GCRODR   GCRO with Deflated Restarting
%    X = GCRODR(A,B,M,K) attempts to solve the system of linear equations A*X = B for
%    X.  The N-by-N coefficient matrix A must be square and the right hand side
%    column vector B must have length N. M is the maximum subspace dimension used
%    by GCRODR. K is the number of approximate eigenvectors kept from one cycle to the 
%    next. 
% 
%    GCRODR(A,B,M,K,X0) specifies the first initial guess.  If X0 is [] then 
%    GCRODR uses the default, an all zero vector.
% 
%    GCRODR(A,B,X0,TOL) specifies the tolerance of the method.  If TOL is []
%    then GCRODR uses the default, 1e-6.
% 
%    GCRODR(A,B,X0,TOL,M1,M2) precondition system as inv(M1)*A*inv(M2)
%    If M1 is [] then a left preconditioner is not applied. If M2 is [] then a 
%    right preconditioner is not applied.
% 
%    GCRODR(A,B,X0,TOL,M1,M2,NAME) specifies a string associated with the subspace
%    that will be recycled from this call. A call with 'nopreconditioning' will 
%    recycle the subspace saved the last time a call was made with the string 
%    'nopreconditioning'. Likewise, a call with 'IC(0)' will recycle the subspace
%    saved the last time a call was made with the string 'IC(0)'. If NAME is [],
%    a default string 'default' is used.
% 
%    [X,RESVEC] = GCRODR(A,B,X0,TOL,M1,M2,NAME) also returns a vector of the 
%    preconditioned residual norms at each inner iteration, including NORM(B-A*X0).
%
%    [X,RESVEC,R] = GCRODR(A,B,X0,TOL,M1,M2,NAME) also returns the preconditioned 
%    residual vector 
%    
%    [X,RESVEC,R,NMV] = GCRODR(A,B,X0,TOL,M1,M2,NAME) also returns the number of 
%    matrix vector multiplications required to solve the system
%
%    [X,RESVEC,R,NMV,RELRES] = GCRODR(A,B,X0,TOL,M1,M2,NAME) also returns 
%    the preconditioned relative residual NORM(B-A*X)/NORM(B).
% 
% SUBFUNCTIONS: gmres1.m, gmres1.m, getHarmVecs1.m, getHarmVecs2.m
%
% NOTES:        Run 'clear all' between calls to GCRO-DR to ensure no subspace is recycled
%
% EXAMPLE:      A = gallery('poisson',20);  b = sum(A,1)';
%               tol = 1e-10;
%               [x,resvec,r,nmv,relres] = gcrodr(A,b,10,4,[],tol);
%               semilogy(resvec);
function [x,resvec,r,nmv,relres] = gcrodr(A,b,m,k,x0, dt,tol,M1,M2,reuse_name)

% Initialize optional variables.
if(nargin < 7 | isempty(tol))
   tol = 1e-6;
end
if(nargin < 8 | isempty(M1))
   existM1 = 0;
   M1 = [];
else
   existM1 = 1;
end
if(nargin < 9 | isempty(M2))
   existM2 = 0;
   M2 = [];
else
   existM2 = 1;
end
if(nargin < 10 | isempty(reuse_name))
   reuse_name = 'default';
end

py.integrators.ros2.make_matvec(x0, dt)
% initialize solution vector
x0 = x0.';
b = b.';
x = zeros(size(x0));

% initialize matvec count
nmv = 1;

% Calculate initial preconditioned residual.
Ax0 = double(py.integrators.ros2.ros2matvec(x0)).';
r = b - Ax0;
if(existM1)
   r = M1 \ r;
end

% Calculate initial preconditioned residual norm.
resvec = zeros(2,1);
resvec(1) = norm(r);
disp(sprintf('1: ||r|| = %e\t\tnmv = %d',resvec(nmv)./resvec(1),nmv-1));

% Precondition rhs if available.
if(existM1)
   bnorm = norm(M1 \ b);
else
   bnorm = norm(b);
end

persistent U_persist;

%%%%%%%%%%%%%%%%%%%  Initialize U (Recycled subspace) %%%%%%%%%%%%%%%%%%%
if(isfield(U_persist,reuse_name))
   % Initialize U with information recycled from previous call to solver.
   eval(sprintf('U = U_persist.%s;',reuse_name));

   % C = A * U (with preconditioning)
   % We can frequently represent A_new = A_old + deltaA. Note that
   % A_new * U = A_old*U + delta_A*U
   %           = C_old   + delta_A*U
   % where we already have C_old. Computing deltaA*U is generally much less 
   % expensive than computing A_new*U, so we do not record these (k) matvecs
   if(existM2)
      C = M2 \ U;
   else
      C = U;
   end
   C_r = py.numpy.array(real(C));
   C_i = py.numpy.array(imag(C));
   dual = py.integrators.ros2.ros2matvec_complex(C_r, C_i);
   new_r = double(dual{1}).';
   new_i = double(dual{2}).';
   C = complex(new_r, new_i);
   if(existM1)
      C = M1 \ C;
   end

   % Orthonormalize C and adjust U accordingly so that C = A*U
   [C,R] = qr(C,0);
   U = U / R;

   % Set residual norm to be the initial residual norm for this case.
   x = x + U*(C'*r);
   r = r - C*(C'*r);
   resvec(1) = norm(r);
   %keyboard
else
   % I have no subspace to recycle from a previous call to the solver
   % So, do one cycle of GMRES to "prime the pump"

   % Perform m GMRES iterations to produce Krylov subspace of dimension m
   [x,r,V,H,p,resvec_inner] = gmres1(A,x,r,m,M1,M2,tol*bnorm);

   % Record residual norms and increment matvec count
   resvec(2:p+1) = resvec_inner;
   nmv = nmv + p;
   disp(sprintf('2: ||r|| = %e\t\tnmv = %d',resvec(nmv)./resvec(1),nmv-1));

   % Find the k smallest harmonic Ritz vectors.
   % Check to be sure GMRES took more than k iterations. Else, I can't compute
   % k harmonic Ritz vectors
   if k < p
       [p k]
      P = getHarmVecs1(p,k,H);

     % Form U (the subspace to recycle)
      U = V(:,1:p) * P;
  
      % Form orthonormalized C and adjust U accordingly so that C = A*U
      [C,R] = qr(H*P,0);
      C = V * C;
      U = U / R;
   end

   % If p < m, early convergence of GMRES
   if p < m
      % Assign all (unassigned) outgoing values and return
      if(existM2)
         x = x0 + M2 \ x;
      else
         x = x0 + x;
      end      
      nmv = nmv - 1;
      relres = resvec(p+1) / bnorm;
      Ax = double(py.integrators.ros2.ros2matvec(x0)).';
      rr = norm(b - Ax) / bnorm;
      % Save information to be carried to next call to solver.
      if k < p
         eval(sprintf('U_persist.%s = U;',reuse_name));
      end
      return
   end

end

%%%%%%%%%%%%%%%%%%%%%%%%%  Main Body of Solver  %%%%%%%%%%%%%%%%%%%%%%%%%

min_it = 1;
num_it = 0;
while(resvec(nmv) / bnorm > tol || num_it < min_it)
  
   num_it = num_it + 1;
   % Do m-k steps of Arnoldi

   [V,H,B,p,resvec_inner] = gmres2(A,x,r,m-k,M1,M2,C,tol*bnorm);
   resvec(nmv+1:nmv+p) = resvec_inner;
   nmv = nmv + p;
   disp(sprintf('3: ||r|| = %e\t\tnmv = %d',resvec(nmv)./resvec(1),nmv-1));

   % Rescale U; Store inverses of the norms of columns of U in diagonal matrix D
   for i = 1:k
      d(i) = norm(U(:,i));
      U(:,i) = U(:,i) / d(i);
   end
   D = diag(1 ./ d);

   % Form large H
   H2 = zeros(p+k+1,p+k);
   H2(1:k,1:k) = D;
   H2(1:k,k+1:p+k) = B;
   H2(k+1:p+k+1,k+1:p+k) = H;

   % Calculate solution update
   rhs = [C V]' * r;
   y = H2 \ rhs; 
   x = x + [U V(:,1:p)] * y;

   % Calculate new residual
   nr1 = norm(r);
   r = r - [C V] * (H2 * y);

   % If p < m-k, early convergence of GMRES
   if (p < m-k && num_it >= min_it)
      % Assign all (unassigned) outgoing values and return
      if(existM2)
         x = x0 + M2 \ x;
      else
         x = x0 + x;
      end      
      relres = resvec(nmv) / bnorm;
      nmv = nmv - 1;
      % Save information to be carried to next call to solver.
      eval(sprintf('U_persist.%s = U;',reuse_name));
      return
   end

   % Calculate Harmonic Ritz vectors.
   P = getHarmVecs2(p+k,k,H2,V,U,C);

   % Form new U and C.
   U = [U V(:,1:p)] * P;

   % Form orthonormalized C and adjust U accordingly so that C = A*U
   [Q,R] = qr(H2*P,0);
   C = [C V] * Q;
   U = U / R;

end

% Save information to be carried to next call to solver.
eval(sprintf('U_persist.%s = U;',reuse_name));

% Calculate final solution and residual.
if(existM2)
   x = x0 + M2 \ x;
else
   x = x0 + x;
end

% Calculate relative residual.
relres = resvec(nmv) / bnorm;

% Correct matvec count
nmv = nmv - 1;