function [output,dlamda,dbeta]= my3DBatchNormalization(A, lamda, beta, forward_or_backward,Loss)
    %MY3DBATCHNORMALIZATION Summary of this function goes here
    %   A Batch Normalization function for 3D voxel grid
    %   'A' is a 4-D matrix,the 4th dimension of the matrix is the number of
    %   the batch

    % e is a constant added to the mini-batch variance for numerical stability.
    epsilon = 1e-4;

    %compute the mean of the batch of Input 3D matrix;
    E_A = mean(A(:));

    %compute the variance of the batch of Input 3D matrix;
    V_A = var(A(:));

    %nomalize the data of the batch of the input 3D matix;
    my_std=sqrt(V_A+epsilon);
    my_A_star = (A-E_A)/my_std;

    if strcmp(forward_or_backward,'forward')
    %compute the output of the Batch Normalization layer;output is a 4-D
    %matrix;
        output = lamda*my_A_star+beta;
        dlamda=0;
        dbeta=0;
    elseif strcmp(forward_or_backward,'backward')
        %Loss and dx_star is a 4 dimensional Matrix;
        dx_star=Loss*lamda;

        %tmp is a 4-D matrix;
        tmp1=(dx_star.*(A-E_A));
        dV_A=sum(tmp1(:))*(-1/2)*(my_std^(-1/3));
        
        %compute the derivation of means;
        tmp2=dx_star-E_A;
        dE_A=(sum(dx_star(:))*(-1/my_std))+dV_A*(-2)*(sum(tmp2(:))/size(A,4));
        
        %compute the derivation of input x;dx is a 4-D matrix;
        dx = dx_star*(1/my_std)+dV_A*2*((A-E_A)/size(A,4))+dE_A/size(A,4);
        
        tmp3=Loss.*my_A_star;
        dlamda=sum(tmp3(:));
        
        dbeta=sum(Loss(:));
        
        output=dx;
    end

end

