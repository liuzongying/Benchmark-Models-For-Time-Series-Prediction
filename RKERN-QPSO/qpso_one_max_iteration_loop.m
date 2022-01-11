 


clear
clc
disp('RKERM with QPSO Start')
%%%%%%%%%%%%%%%%%%%%%Determine the parameters of the algorithm and the problem %%%%%%%%%%%%%%%%% 

%% dimension is how many parameter need to set
%% irange_1 & irange_r is the range of parameter for initialization
%% xmax & xmin is the min and max value for search scope
%% popsize is population size
%% runno is reported time for algorithm

popsize=10;% population size
MAXITER=50;  % Maximum number of iterations
dimension=2;  % Dimensionality of the problem
irange_l=0.6e-10; % Lower bound of initialization scope
irange_r=100;  % Upper bound of initialization scope
xmax=100;   % Upper bound of the search scope
xmin=0.6e-10;  % Lower bound of the search scope
M=(xmax-xmin)/2; % The middle point of the search cope on each dimension
alpha=0.6; % The value of alpha if the fixed value is used
runno=50; % The runno is the times that the algorithm runs 
data=zeros(runno,MAXITER); % The matrix record the objective functioin value of gbest position at each iteration during a single run of the algorithm

%%%%%%%%%%%%%%% The following is that the algorithm runs for runno times%%%%%%%%%%%%%


  
%%%%%%%%%%%%%%% Initialization of the particle swarm %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    x=(irange_r-irange_l)*rand(popsize,dimension,1) + irange_l;% Initialize the particle's current poistion  
   for i = 1:size(x,1)
        if x(i,2)<10 && x(i,2)>=1
            n = x(i,2)/10;
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;    
            clear n
        elseif x(i,2)>=10 && x(i,2)<100
            n = x(i,2)/100;
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;  
            clear n
        elseif x(i,2)>=100;
            x(i,2) = n; 
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n; 
            clear n
        end
   end
    
    %% dimension is how many parameter need to set in function
    pbest=x;    %Set the pbest position as the current position of the particle
  %  gbest=zeros(1,dimension); % Initialize the gbest poistion vector
    
    for i=1:popsize
        f_x(i)=Recurrent_KELM_with_reservoir_computing_2('norPalm_oil.csv', 0, 'RBF_kernel',x(i,1),x(i,2));   %Calculate the fitness value of the current position of the particle
        f_pbest(i)=f_x(i);% Set the fitness value of the particle's pbest position to be that of its current position
    end
    
    g=min(find(f_pbest==min(f_pbest(1:popsize))));  % Find the index of the particle with gbest position
    gbest=pbest(g,:);  % Determine the gbest position
    f_gbest=f_pbest(g); % Determinet the fitness value of the gbest position
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%% The following is the loop of the QPSO's search process%%%%%%%


    for t=1:MAXITER
        alpha=(1-0.5)*(MAXITER-t)/MAXITER+0.5; % Determine the value of
%         alpha when the linearly descreasing alpha is used
        mbest=sum(pbest)/popsize; % Calculate the mbest position(mean position of the whole particles)
        
        for i=1:popsize  %The following is the update of the particle's poistion 
            fi=rand(1,dimension); % create random number fi
            p=fi.*pbest(i,:)+(1-fi).*gbest; % center position equation��5��
            u=rand(1,dimension);  % create random number u
            x(i,:)=p+((-1).^ceil(0.5+rand(1,dimension))).*(alpha*abs(mbest-x(i,:)).*log(1./u));
            x(i,:)=x(i,:)-(xmax+xmin)/2; % This and the next two lines are to restrict the position within the search scope
            x(i,:)=sign(x(i,:)).*min(abs(x(i,:)),M);
            x(i,:)=x(i,:)+(xmax+xmin)/2; 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The above 7 lines of codes use matrix operation, which can be
%% replaced by the following equivalent codes. The martix operation
%%%%%%%%%%%%%%% accelerates the running of the codes%%%%%%%%%%%%%%%%%%%%%%%%%%
      
%          for d=1: dimension
%              fi=rand;
%              p=fi*pbest(i,d)+(1-fi)*gbest(d);
%              u=rand;
%              if rand>0.5
%                  x(i,d)=p+alpha*abs(mbest(d)-x(i,d))*log(1/u);
%              else
%                  x(i,d)=p-alpha*abs(mbest(d)-x(i,d))*log(1/u);
%              end
%              if x(i,d)>xmax;
%                  x(i,d)=xmax;
%              end
%              if x(i,d)<-xmax;
%                  x(i,d)=-xmax;
%              end
%          end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%     
            if x(i,2)>0 && x(i,2)<=1
            n = x(i,2)*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;    
            clear n    
            elseif x(i,2) == 0
            x(i,2) = mean(x(:,2));
            n = x(i,2)*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;    
            clear n    
            elseif x(i,2)<10 && x(i,2)>=1
            n = x(i,2)/10;
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;    
            clear n
        elseif x(i,2)>=10 && x(i,2)<100
            n = x(i,2)/100;
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);
            x(i,2) = n;  
            clear n
        elseif x(i,2)>=100;
            x(i,2) = n; 
            n = n*(10^2);
            n = round(n);
            n = n/(10^2);          
            x(i,2) = n; 
            clear n
            end
        
            f_x(i)=Recurrent_KELM_with_reservoir_computing_2('norPalm_oil.csv', 0, 'RBF_kernel',x(i,1),x(i,2));  % Calculate the fitness value of the particle's current position
            if (f_x(i)<f_pbest(i)) 
                pbest(i,:)=x(i,:);% Update the pbest position of the particle
                f_pbest(i)=f_x(i);% Update the fitness value of the particle's pbest position
            end

            if f_pbest(i)<f_gbest
                gbest=pbest(i,:); % Update the gbest position
                f_gbest=f_pbest(i); % Update the fitness value of the gbes position
            end
        end
%     data(1,t)=f_gbest;  % Record the fitness value of the gbest at each iteration at this run
     % Display Iteration Information
             disp(['Iteration ' num2str(t) ': Best Cost = ' num2str(f_gbest) ', Best position1 =' num2str(gbest)]);
      BestCost.Iteration(t,1) = f_gbest;
%     f_gbest
    end


BEST_COST = f_gbest;
BEST_POSITION = gbest;
        
 