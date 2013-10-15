%Done by Pedro Cisneros, 2013 ! :)

%Perceptron:
aver_var  = 0;
summ = 0;
summ_p = 0;

%Number of total samples:
% N = 10;
N = 100;
xn1 = zeros(N,1); %First component of samples
xn2 = zeros(N,1); %Second component of samples
yn = zeros(N,1); %Outcome of samples

%Number of total tests we want to do
N_iter = 1000;


for i = 1:N_iter
    
%=======================================================
%=== INITIALIZATION OF INITIAL DATA, SUPERVISED LEARNING

%==
%Generating the funtion f
N_iter_inner = 0; %This will count the number of inner iterations
xo1 = unifrnd(-1,1);
yo1 = unifrnd(-1,1);
xo2 = unifrnd(-1,1);
yo2 = unifrnd(-1,1);
%Line equation
m = (yo1 - yo2)/(xo1 - xo2);
b = yo1 - (m*xo1);
%==

%==
%Getting sample data (as random numbers)
for j = 1:N
   xn1(j) = unifrnd(-1,1);
   xn2(j) = unifrnd(-1,1);
   tempp = xn2(j) - (m*xn1(j)) - b;
   if( tempp >= 0 )
       yn(j) = 1;
   else
       yn(j) = -1;
   end
end
x = [xn2 xn1 ones(N,1)];
%==

miss_classx = zeros(N,3); %Will hold misclassified points
miss_classy = zeros(N,1); %Will hold misclassified points outcomes
w = zeros(3,1);

xt1 = unifrnd(-1,1,50,1); %Outer random data.
xt2 = unifrnd(-1,1,50,1); %Outer random data.


%=== ITERATIONS OF PLA ======================================!!!!!!!!!!!!!
while(1==1) %Infinite loop until convergence
        
    %Producing set misclasfied points
    count_miss = 0;
    for k = 1:N
        signo = sign(w'*(x(k,:))');
        if(signo ~= yn(k))
            count_miss = count_miss + 1;
            miss_classx(count_miss,:) = x(k,:);
            miss_classy(count_miss) = yn(k);
        end
    end
    
    %Choose a random misclassified point
    if (count_miss ~= 0)
        indice = unidrnd(count_miss);
        xnn = miss_classx(indice,:);
        w = w + miss_classy(indice)*(xnn'); %Updating paramerters "w"
    else
        break; %NO misclassified points
    end
    
    N_iter_inner = N_iter_inner + 1;
end
%=========================================================================

%=========================================================================
%Calculating Probability of misclassfication:

%=FORM 1: Just testing one outer point per test, which will be added in all
%of the tests done and averaged after
% x11 = unifrnd(-1,1);
% x22 = unifrnd(-1,1);
% z22 = 1;
% 
% tempp = x22 - (m*x11) - b;
% if( tempp >= 0 )
%     y22 = 1;
% else
%     y22 = -1;
% end
% 
% signe = sign(w'*([x22; x11; 1]));
% if(signe ~= y22)
%     summ_p = summ_p + 1;
% end

%=FORM 2: Just calculating the error probability Ein by using 50 outer sata
%samples and its classsification
summ_p_inter = 0;
for gg = 1:50
    tempp = xt2(gg) - (m*xt1(gg)) - b;
    if( tempp >= 0 )
        y22 = 1;
    else
        y22 = -1;
    end

    signe = sign(w'*([xt2(gg); xt1(gg); 1]));
    if(signe ~= y22)
        summ_p_inter = summ_p_inter + 1;
    end
    
end
summ_p_inter = summ_p_inter / 50;
summ_p = summ_p + summ_p_inter;
%=========================================================================

%====

summ = summ + N_iter_inner; %This will count the number of iterations per 
                            %test.
end

aver_var  = summ / N_iter %Average number of the iterations in all tests
aver_var  = summ_p / N_iter %Average probability error
