classdef twoDimProb<handle
    %twoDimProb A 2D problem object. This object generates random test points
    % in [-1,1]x[-1,1] space
    %   Detailed explanation goes here
    
    properties
        targetWeight=zeros(1,3);
        learnedWeight=[0,0,1];
        sampleX; %sample sets for learning
        sampleNumber; %number of samples for learning
        sampleY;
        sampleSetPlus;
        sampleSetMinus;
        ax;
        targetLineArray;
    end
    
    methods
        function prob=twoDimProb(learningSampleNumber)
            prob.sampleNumber=learningSampleNumber;  %number of samples for learning
            sampleCoords=rand(2,prob.sampleNumber)*2-1; % scale [0,1] space to [-1,1];
            prob.sampleX=[ones(1,prob.sampleNumber);sampleCoords];
            
            %Randomly select a target function
            targetFuncPoints=rand(2,2)*2-1;
            prob.targetWeight=prob.getTargetWeight(targetFuncPoints(1,1),targetFuncPoints(2,1),...
                targetFuncPoints(1,2),targetFuncPoints(2,2));
            
            
            prob.sampleY=transpose(sign(prob.targetWeight'*prob.sampleX));
            
            
            %sort the points into sampleSetPlus and sampleSetMinus
            prob.sortingPoints();
            
            
            prob.targetLineArray=prob.getArrayForPlot(prob.targetWeight);
        end
        
        function genSampleBySetTarget(prob,noiseRatio)
           %assign a pre-defined function and generate some noice into it
           prob.sampleNumber=1000;
           
           sampleCoords=rand(2,prob.sampleNumber)*2-1; % scale [0,1] space to [-1,1];
           prob.sampleX=[ones(1,prob.sampleNumber);sampleCoords];
           %target function: f(x1,x2)=sign(x1^2+x2^2-0.6)
           
           x=[ones(1,prob.sampleNumber)*(-0.6);sampleCoords];
           y=prob.sampleX.*x;
           prob.sampleY=transpose(sign(sum(y,1)));
           
           noiseIndex=randi(prob.sampleNumber,1,ceil(noiseRatio*prob.sampleNumber)); % the indexes that assign which sample need to be flipped
           
           %making some noises
           for i=1:length(noiseIndex)
               prob.sampleY(noiseIndex(i))=prob.sampleY(noiseIndex(i))*(-1);
           end      
        end
        
        function genSampleBySetTarget2(prob)
           %assign a pre-defined function and generate some noice into it
           prob.sampleNumber=1000;
           
           sampleCoords=rand(2,prob.sampleNumber)*2-1; % scale [0,1] space to [-1,1];
           prob.sampleX=[ones(1,prob.sampleNumber);sampleCoords];
           
           %set the function f(x)=sign(x2-x1+0.25*sin(pi*x));
           prob.sampleY=transpose(sign(prob.sampleX(3,:)-prob.sampleX(2,:)+0.25*sin(pi*prob.sampleX(2,:))));         
        end
        
        function plotLearningPoints(prob)
            ax=axes();
            for i=1:prob.sampleNumber
                %h=prob.targetWeight*prob.sampleX(i,:)';
                h=prob.sampleY(i);
                if h>0
                    plot(ax,prob.sampleX(2,i),prob.sampleX(3,i),'+','color','blue');
                    hold on;
                elseif h<0
                    plot(ax,prob.sampleX(2,i),prob.sampleX(3,i),'o','color','blue');  
                    hold on;
                elseif h==0;
                    plot(ax,prob.sampleX(2,i),prob.sampleX(3,i),'*','color','blue');
                    hold on;
                end
            end
            
            plot(ax,prob.targetLineArray(:,1),prob.targetLineArray(:,2),'color','m');
            
            prob.learnedWeight=prob.learnedWeight/prob.learnedWeight(3);

            learnedLineArray=prob.getArrayForPlot(prob.learnedWeight);

            plot(learnedLineArray(:,1),learnedLineArray(:,2),'color','r');

            xlim([-1,1]);
            ylim([-1,1]);
            hold off;
        end
        
        function sortingPoints(prob)
            prob.sampleSetPlus=prob.sampleX(:,prob.sampleY>=0);
            prob.sampleSetMinus=prob.sampleX(:,prob.sampleY<=0);        
        end
        
        
        function Eout=calcEout(prob,testSampleNumber,mode)
             %number of samples for learning
            sampleCoords=rand(2,testSampleNumber)*2-1; % scale [0,1] space to [-1,1];
            testSampleX=[ones(1,testSampleNumber);sampleCoords];
            
            %calculate Eout
            switch mode
                case 'abs'
                    EoutSum=(sign(prob.learnedWeight'*testSampleX)-sign(prob.targetWeight'*testSampleX));
                    Eout=sum((EoutSum.*EoutSum))/testSampleNumber;
                case 'frac'
                    EoutSum=(sign(prob.learnedWeight'*testSampleX)~=sign(prob.targetWeight'*testSampleX));
                    Eout=sum(EoutSum)/testSampleNumber;
            end
            
        end
        
        function Ein=calcEin(prob,mode)
            if strcmp(mode,'abs')
                EinSum=(sign(prob.learnedWeight'*prob.sampleX)-prob.sampleY');
                Ein=sum((EinSum.*EinSum))/prob.sampleNumber;
            elseif strcmp(mode,'frac') %fraction of in-sample points which got classifed incorrectly
                EinSum=(sign(prob.learnedWeight'*prob.sampleX)~=prob.sampleY');
                Ein=sum(EinSum)/prob.sampleNumber;
            end
        end
        
        function weight=getTargetWeight(prob,x1,x2,y1,y2)
            m=(y1-y2)/(x1-x2);
            w2=1;
            w1=-m;
            w0=-y1+x1*m;
            weight=[w0;w1;w2];
        end
        
        function arrays=getArrayForPlot(prob,w)
            lx=-1;
            rx=1;
            by=-1;
            ty=1;
            x1=-(w(1) + w(3)*by)/w(2);
            x2=-(w(1) + w(3)*ty)/w(2);

            y1=-(w(1) + w(2)*lx)/w(3);
            y2=-(w(1) + w(2)*rx)/w(3);

            arrays=[x1,by;x2,ty;lx,y1;rx,y2];
        end
        
    end
    
end

