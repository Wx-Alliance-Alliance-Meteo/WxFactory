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
%         kmax     max  allowable for k
%         opt      1 = increase k to not split complex pair
%                  2 = keep k fixed, 
%                  3 = only let the last evec be complex if it would ...
%                           be split (keep k fixed)
%                  4 = no modification (whether real or complex)    
  
% OUTPUT: HARMVECS basis for span of K harmonic Ritz vectors


function [harmVecs, new_k] = getHarmVecs1(m,k,H, kmax, opt)

complex_cnt = 0;
if opt > 4
    opt = 4;
end
    
% Build matrix for eigenvalue problem.
harmRitzMat = H(1:m,:)' \ speye(m);
harmRitzMat(1:m,1:m-1) = 0;
harmRitzMat = H(1:m,:) + H(m+1,m)^2 * harmRitzMat;

[harmVecs, harmVals] = eig(harmRitzMat);

dv = diag(harmVals);
dv_mag = abs(dv); %get magnitude in case complex
TT = [dv_mag dv harmVecs'];
TT = sortrows(TT);
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
else  %contains complex    
    %if complex, we split into the real and imag part
    %then skip the second complex of same mag
    prev_comp = 0.0;
    for j = 1:num
        if (isreal(harmVecs(:,j))) %real
            new_harmVecs = [new_harmVecs harmVecs(:,j)];
            v_cnt = v_cnt + 1;
        else %complex
            if (magVals(j) == prev_comp)
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
    
    if v_cnt > num    %indicates that the last evec is complex 
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
    harmVecs = new_harmVecs(:, 1:num); 
end

