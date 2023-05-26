% getHarmVecs2     For use with GCRODR
%
% Determines harmonic Ritz vectors using matrices computed from
% GMRES iteration. 
% 
% INPUT:  M        dimension of upper Hessenburg matrix H
%         K        select and return basis for space spanned by K harmonic 
%                  Ritz vectors corresponding to K harmonic Ritz values 
%                  of smallest magnitude
%         H2       upper Hessenburg matrix computed GCRODR relations
%         V        N-by-M+1 matrix containing Krylov basis computed by GMRES
%         U        basis for recycled subspace
%         C        C = A*U, as per GCRODR relations
%         kmax     max  allowable for k
%         korig    original specified....   
% OUTPUT: HARMVECS basis for span of K harmonic Ritz vectors
function [harmVecs, new_k] = getHarmVecs2(m,k,H2,V,U,C, kmax, korig)

complex_cnt = 0;
real_cnt = 0;

B = H2' * H2;

%keyboard

% A = | C'*U        0 |
%     | V_{m+1}'*U  I |
A = zeros(m+1,m);
A(1:k,1:k) = C' * U;
A(k+1:m+1,1:k) = V' * U;
A(k+1:m,k+1:m) = eye(m-k);
A = H2' * A;

%sum(sum(isnan(A) == true));
%sum(sum(isnan(B) == true));
% Compute k smallest harmonic Ritz pairs.
[harmVecs, harmVals] = eig(A,B);
%fprintf('HRV2- is hVecs real? %d\n', isreal(harmVecs)) 
dv = diag(harmVals);
dv_mag = abs(dv);
TT = [dv_mag dv harmVecs'];
%AB - we want descending order to keep largest of (1./harmonic Ritz value)
TT = sortrows(TT, 'descend');
harmVecs = (TT(1:k,3:m+2)');
harmVals = TT(1:k,2);
magVals = TT(1:k,1);
new_harmVecs = [];
num = size(harmVecs,2); %this is k
v_cnt = 0;

%keyboard

%if complex, we need to include the real and imag part
%if two complex of same mag, skip 2nd one
prev_comp = 0.0;
for j = 1:num
    if (isreal(harmVecs(:,j))) %real
        new_harmVecs = [new_harmVecs harmVecs(:,j)];
        %disp('real')
        v_cnt = v_cnt + 1;
        real_cnt = real_cnt + 1;
    else %complex
         %disp('complex')
        if (magVals(j) == prev_comp)
            %fprintf('Skipping = %d\n', magVals(j));
             continue;
        end 
        new_harmVecs = [new_harmVecs, real(harmVecs(:,j)), imag(harmVecs(:,j))];    
        v_cnt = v_cnt + 2;
        prev_comp = magVals(j);
        complex_cnt = complex_cnt + 2;
    end 
    if v_cnt >= k
        break
    end
end
%fprintf('k = %d\n', k)
%fprintf('kmax = %d\n', kmax)
%fprintf('v_cnt = %d\n', v_cnt)
%fprintf('real_cnt = %d\n', real_cnt)
%fprintf('complex_cnt = %d\n', complex_cnt)

if v_cnt > k   %last one is complex 
    %avoid splitting the last complex evec (but don't go over kmax)
    if v_cnt > kmax
        num = kmax -1;
    else %add 1
        num = v_cnt;
    end
    fprintf('adjusting k ... new k = %d\n', num)

    %WARNING: temp to allow splitting complex evcs for testing or
    %let the last be complex
    %num = k;
    %new_harmVecs(:,k) = harmVecs(:,k);
    %disp('Using a complex evec')
end

%disp('change num?')
%keyboard

new_k = num;
harmVecs = new_harmVecs(:, 1:num); 
size(harmVecs);
%fprintf('new_k = %d\n', new_k)

%keyboard

% k smallest harmonic ritz values
% Actually, k largest of (1./harmonic Ritz value)