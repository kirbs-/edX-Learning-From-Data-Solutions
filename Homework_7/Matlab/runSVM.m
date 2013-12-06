function [ alpha,learnedWeight,b] = runSVM( trainSampleX,trainSampleY,kernelFunc )
%runSVM run support vector machine
%   trainSampleX: X vectors of training samples in the form of [1;x1;x2];
%   trainSampleY: resposes Y
%   kernelFunc: a function handle for kernel function in quadratic
%   programming

sampleNumber=length(trainSampleY);

Qmtx=zeros(sampleNumber);
for i=1:sampleNumber
    for j=1:sampleNumber
        Qmtx(i,j)=trainSampleY(i)*trainSampleY(j)*kernelFunc(trainSampleX(2:end,i),trainSampleX(2:end,j));
    end
end

opts = optimoptions('quadprog','Algorithm','interior-point-convex','Display','off');

alpha=quadprog(Qmtx,-ones(sampleNumber,1),[],[],trainSampleY',0,zeros(sampleNumber,1),[],[],opts);

learnedWeight=zeros(2,1);

for i=1:sampleNumber
    learnedWeight=learnedWeight+alpha(i)*trainSampleY(i)*trainSampleX(2:end,i);
end

%solve b
%yn(w'xn+b)=1
t=abs(alpha)>1e-4;
supportX=trainSampleX(:,t);
supportY=trainSampleY(t);

if isempty(supportY)~=true
    b=1./supportY(1)-learnedWeight'*supportX(2:end,1);
else
    learnedWeight=[1;1];
    b=1;
end

end

function K=kernel(x1,x2)
    K=(1+transpose(x1)*x2)^2;
end

function K=basicKernel(x1,x2)
    K=x1'*x2;
end

function K=RBFKernel(x1,x2)
    gamma=1.5;
    K=exp(-gamma*(sum((x1-x2).^2)));
end

