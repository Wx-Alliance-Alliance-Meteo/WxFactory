% getHarmVecs2  
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
%         opt      1 = increase k to not split complex pair
%                  2 = keep k fixed, 
%                  3 = only let the last evec be complex if it would ...
%                           be split (keep k fixed)
%                  4 = no modification (whether real or complex)    
                       
% OUTPUT: HARMVECS basis for span of K harmonic Ritz vectors
function [harmVecs, new_k] = getHarmVecs2(m,k,H2,V,U,C, kmax, opt)

complex_cnt = 0;
real_cnt = 0;
if opt > 4
    opt = 4;
end
    

B = H2' * H2;

% A = | C'*U        0 |
%     | V_{m+1}'*U  I |
A = zeros(m+1,m);
A(1:k,1:k) = C' * U;
A(k+1:m+1,1:k) = V' * U;
A(k+1:m,k+1:m) = eye(m-k);
A = H2' * A;

% Compute k smallest harmonic Ritz pairs.
[harmVecs, harmVals] = eig(A,B);
dv = diag(harmVals);
dv_mag = abs(dv); %get magnitude in case complex
TT = [dv_mag dv harmVecs'];
%AB - we want descending order to keep largest of (1./harmonic Ritz value)
TT = sortrows(TT, 'descend');
harmVecs = (TT(1:k,3:m+2)');
harmVals = TT(1:k,2);
magVals = TT(1:k,1);
new_harmVecs = [];
num = size(harmVecs,2); %this is k
v_cnt = 0;

%are any of the harmVecs complex?
isreal_hV = isreal(harmVecs);
%fprintf('Are harmVecs real? %d\n', isreal_hV) 

if ((opt == 4) || (isreal_hV)) %either all real or we are allowing complex
    %no adjustments needed
    new_k = num;
    harmVecs = harmVecs(:, 1:num); 
else %contains complex    
    %if complex, we split into the real and imag part
    %then skip the second complex of same mag
    prev_comp = 0.0;
    for j = 1:num
        if (isreal(harmVecs(:,j))) %real
            new_harmVecs = [new_harmVecs harmVecs(:,j)];
            v_cnt = v_cnt + 1;
            real_cnt = real_cnt + 1;
        else %complex
            if (magVals(j) == prev_comp)
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
    
    if v_cnt > k   %indicates that the last evec is complex 
                   %avoid splitting the last complex evec (but don't go over kmax)
        if (opt == 1) %adjust k (don't exceed max)       
            if v_cnt > kmax
                num = kmax -1;
            else %add 1
                num = v_cnt;
            end
            fprintf('adjusting k ... new k = %d\n', num)
        elseif (opt == 2) %keep k fixed - so allow the last evec
                           %to be split (i.e., only real part included)   
             num = k;
        elseif (opt == 3)
             %let the last evec only be complex (to avoid splitting)
             num = k;
             new_harmVecs(:,k) = harmVecs(:,k);
             %disp('Using a complex evec')
        end
    end

    new_k = num;
    % k largest of (1./harmonic Ritz value)
    harmVecs = new_harmVecs(:, 1:num); 

    %fprintf('new_k = %d\n', new_k)

end

