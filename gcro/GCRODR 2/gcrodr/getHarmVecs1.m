% getHarmVecs1      For use with GCRODR
%
% Determines harmonic Ritz vectors using matrix H computed from a
% GMRES iteration. For this case, the harmonic Ritz values are the
% eigenvalues of H
% 
% INPUT:  M        dimension of upper Hessenburg matrix H
%         K        select and return basis for space spanned by K harmonic 
%                  Ritz vectors corresponding to K harmonic Ritz values 
%                  of smallest magnitude
%         H        M+1-by-M upper Hessenburg matrix computed from GMRES 
% OUTPUT: HARMVECS basis for span of K harmonic Ritz vectors

function [harmVecs, new_k] = getHarmVecs1(m,k,H, kmax)

complex_cnt = 0;

% Build matrix for eigenvalue problem.
harmRitzMat = H(1:m,:)' \ speye(m);
harmRitzMat(1:m,1:m-1) = 0;
harmRitzMat = H(1:m,:) + H(m+1,m)^2 * harmRitzMat;
%fprintf('HRV1- is hrm real? %d\n', isreal(harmRitzMat)) 

[harmVecs, harmVals] = eig(harmRitzMat);
%fprintf('HRV1- is hVecs real? %d\n', isreal(harmVecs)) 

dv = diag(harmVals);
dv_mag = abs(dv);
TT = [dv_mag dv harmVecs'];
TT = sortrows(TT);
harmVecs = (TT(1:k,3:m+2)');
harmVals = TT(1:k,2);
magVals = TT(1:k,1);
%now if the evec is complex, split
%into real and complex parts
new_harmVecs = [];
num = size(harmVecs,2); %this is k
v_cnt = 0;

%if complex, we need to include the real and imag part
%if two complex of same mag, skip 2nd one
prev_comp = 0.0;
for j = 1:num
    if (isreal(harmVecs(:,j))) %real
        %disp('real')
        new_harmVecs = [new_harmVecs harmVecs(:,j)];
        v_cnt = v_cnt + 1;
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
    if v_cnt >= num
        break
    end
end
%disp('k = ')
%disp(num)
%disp('v_cnt = ')
%disp(v_cnt)
%fprintf('complex_cnt = %d\n', complex_cnt)

if v_cnt > num   %last one is complex
    %avoid splitting the last complex evec (but don't go over kmax)
    if v_cnt > kmax
        num = kmax -1;
    else %add 1
        num = v_cnt;
    end
    fprintf('adjusting k ... new k = %d\n', num)
end
%k smallest
new_k = num;
harmVecs = new_harmVecs(:, 1:num); 
size(harmVecs);

