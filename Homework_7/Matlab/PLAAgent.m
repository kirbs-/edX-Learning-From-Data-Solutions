classdef PLAAgent<handle
    %PLAAgent An agent that solves prolem using perceptron learning
    %algorithm
    
    
    properties
        convergedIteration=0;
        learnedWeight;
    end
    
    methods
        function ag=PLAAgent(prob)
            %Constructor: user problem as function input
           ag.learnedWeight=[0;0;0];
        end

        function ag=learningFromData(ag,prob,maxLearningIterations,initialWeight)
            %This function runs perceptron learning algorithm, with initial
            %weight function [0;0;0];
            
            
            %Initialise a matrix that records the weights at each iteration
            weightHistory=zeros(3,maxLearningIterations);
            
            learningWeight=initialWeight; %initial condition
            ag.convergedIteration=maxLearningIterations;
            for k=1:maxLearningIterations
                converged=true;
                
                % run through each learning points in a radom order
                % randomise the order of learning points
                p=randperm(prob.sampleNumber);
                for i=1:prob.sampleNumber
                    y=prob.sampleY(p(i));
                    h=sign(learningWeight'*prob.sampleX(:,p(i)));
                    if sign(h*y)~=1;
                       learningWeight=learningWeight+sign(y)*prob.sampleX(:,p(i));
                       converged=false;
                    end
                end
                weightHistory(:,k)=learningWeight;
                
                %convergence test, if the hypothesis satisfies every
                %learning point, break the iteration
                if converged==true;
                    ag.convergedIteration=k;
                    ag.learnedWeight=learningWeight;
                    break;
                end
            end
            %output the result
            ag.learnedWeight=learningWeight;
        end
        
        
    end
    
end

