function [output,dlamda,dbeta]= my3DBatchNormalization(A, lamda, beta, forward_or_backward,Loss)
    %MY3DBATCHNORMALIZATION Summary of this function goes here
    %   A Batch Normalization function for 3D voxel grid
    %   'A' is a 4-D matrix,the 1th dimension of the matrix is the number of
    %   the batch

    % e is a constant added to the mini-batch variance for numerical stability.
    epsilon = 1e-4;

    %compute the mean of the batch of Input 3D matrix;
    mu = mean(A,1);

    %compute the variance of the batch of Input 3D matrix;
    x_mu = bsxfun(@minus,A,mu);
    x_sigma2 = mean(x_mu.^2,1);
    norm_factor = lamda./sqrt(x_sigma2+epsilon);

    %nomalize the data of the batch of the input 3D matix;
    x_hat = bsxfun(@times,A,norm_factor);

    if strcmp(forward_or_backward,'forward')
    %compute the output of the Batch Normalization layer;output is a 4-D
    %matrix;
        output = bsxfun(@plus,x_hat,beta-norm_factor.*mu);
        dlamda=0;
        dbeta=0;
    elseif strcmp(forward_or_backward,'backward')
        %Loss and dx_star is a 4 dimensional Matrix;
        d_hat = bsxfun(@times,Loss,lamda);
        % a 4-D matrix;
        inv_sqrt_sigma = 1./sqrt(x_sigma2+epsilon);
        d_sigam2 = -0.5*sum((d_hat.*x_mu),1).*(inv_sqrt_sigma.^3);
        
        %compute the derivation of means;
        d_mu = bsxfun(@times,d_hat,inv_sqrt_sigma);
        d_mu = -1 * sum(d_mu,1)-2.*d_sigam2.*mean(x_mu,1);
        
        %compute the derivation of input x;dx is a 4-D matrix;
        di1 = bsxfun(@times,d_hat,inv_sqrt_sigma);
        di2 = 2/size(A,1) * bsxfun(@times,d_sigam2,x_mu);
        
        tmp = (Loss.*x_hat);
        dlamda=sum(tmp(:));
        
        dbeta=sum(Loss(:));
        
        output=di1+di2+1/size(A,1)*repmat(d_mu,1,1,1,size(A,1));
    end

end

