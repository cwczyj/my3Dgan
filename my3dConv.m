function D= my3dConv( A,B,stride,padding, T)
%MY3DCONV Summary of this function goes here
%   if T == 'C' means this operator convolute matrix A and matrix B;
%   if T == 'T' means this operator Transpose convolute matrix A and matrix
%   B;
%   B is the kernel;

    sizeA=size(A,1);
    sizeB=size(B,1);
    k=stride;
    p=padding;

    
    
    if T == 'C'
        %% 3D convolution
        sizeC=(sizeA-sizeB+2*p)+1;

        C=zeros(sizeC,sizeC,sizeC);
        
        %add padding to the A matrix
        tmpA=zeros((sizeA+2*p),(sizeA+2*p),(sizeA+2*p));
        tmpA((p+1:end-p),(p+1:end-p),(p+1:end-p))=A;
        A=tmpA;
     
        for i=1:sizeC
            for j=1:sizeB
                C(:,:,i)=C(:,:,i)+conv2(A(:,:,i+j-1),rot180(B(:,:,j)),'valid');
            end
        end
        
        %use stride here
        D=C((1:k:end),(1:k:end),(1:k:end));
  
    
    elseif T == 'T'
    %% DeConvolution
        %use stride here for Transpose Convolution
        tmpx=(sizeA-1)*(k-1)+sizeA;
        tmpA=zeros(tmpx,tmpx,tmpx);
        tmpA((1:k:end),(1:k:end),(1:k:end))=A;
        A=tmpA;
        sizeA=size(A,1);

        %3D Transpose Convolution
            sizeC=(sizeA+sizeB)-1;

            C=zeros(sizeC,sizeC,sizeC,'single');

            if sizeA>sizeB

                for i=1:sizeC
                    if i<sizeB
                        for j=1:i
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,i-j+1),B(:,:,j),'full');
                        end
                    elseif i<sizeA
                        for j=1:sizeB
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,i-j+1),B(:,:,j),'full');
                        end
                    else
                        for j=(i-sizeA+1):sizeB
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,i-j+1),B(:,:,j),'full');
                        end
                    end
                end

            else

                for i=1:sizeC
                    if i<sizeA
                        for j=1:i
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,i-j+1),B(:,:,j),'full');
                        end
                    elseif i<sizeB
                        for j=1:sizeA
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,j),B(:,:,i-j+1),'full');
                        end
                    else
                        for j=(i-sizeB+1):sizeA
                            C(:,:,i)=C(:,:,i)+conv2(A(:,:,j),B(:,:,i-j+1),'full');
                        end
                    end
                end

            end

           %use padding here
           D=C((p+1:end-p),(p+1:end-p),(p+1:end-p));
       
    end
    
    function tmp = rot180(X)
        tmp = flip(flip(X, 1), 2);
    end
end

