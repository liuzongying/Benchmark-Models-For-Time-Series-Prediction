function [final_result,time,average_test_SMAPE]...
    = Recurrent_KELM(Data_File, Elm_Type, Regularization_coefficient, Kernel_type, Kernel_para)
%[final_result,time,average_test_SMAPE] = Recurrent_KELM('nonE2.csv', 0,1, 'RBF_kernel',8.1836)


%%%%%%%%%%% Macro definition
REGRESSION=0;
CLASSIFIER=1;

d =load(Data_File);
data = d(1:end,:); %data = d(1:100,1:14)�� ozone=2909 mg=2987
% data for investment data
% data = d(1:end,1:20);
predictsize =18;         %the number of training X (cloumn)
period = 18;              %the number of predict size
%period = size(data,2)-predictsize
% dd = data(:,1:(size(data,2)-predictsize)+1)   %price data dd = data(:,1:8)
pd=1;
ntrain = fix(size(data,1)*0.7);
ntest = size(data,1) - ntrain;
train = data(1:ntrain,:);
test = data(ntrain+1:end,:);


%%%%%%%%%%% Load training dataset
for m = 0:period-1
    
train_data=train;
T=train_data(:,predictsize+1)';
P=train_data(:,1:predictsize)';
                                %   Release raw training data array

%%%%%%%%%%% Load testing dataset
test_data=test;
TV.T=test_data(:,predictsize+1)';
TV.P=test_data(:,1:predictsize)';                             %   Release raw testing data array

C = Regularization_coefficient;
NumberofTrainingData=size(P,2);
NumberofTestingData=size(TV.P,2);

if Elm_Type~=REGRESSION
    %%%%%%%%%%%% Preprocessing the data of classification
    sorted_target=sort(cat(2,T,TV.T),2);
    label=zeros(1,1);                               %   Find and save in 'label' class label from training and testing data sets
    label(1,1)=sorted_target(1,1);
    j=1;
    for i = 2:(NumberofTrainingData+NumberofTestingData)
        if sorted_target(1,i) ~= label(1,j)
            j=j+1;
            label(1,j) = sorted_target(1,i);
        end
    end
    number_class=j;
    NumberofOutputNeurons=number_class;
    

    %%%%%%%%%% Processing the targets of training
    temp_T=zeros(NumberofOutputNeurons, NumberofTrainingData);
    for i = 1:NumberofTrainingData
        for j = 1:number_class
            if label(1,j) == T(1,i)
                break; 
            end
        end
        temp_T(j,i)=1;
    end
    T=temp_T*2-1;

    %%%%%%%%%% Processing the targets of testing
    temp_TV_T=zeros(NumberofOutputNeurons, NumberofTestingData);
    for i = 1:NumberofTestingData
        for j = 1:number_class
            if label(1,j) == TV.T(1,i)
                break; 
            end
        end
        temp_TV_T(j,i)=1;
    end
    TV.T=temp_TV_T*2-1;
                                              %   end if of Elm_Type
end
start_training_time(m+1,1) = cputime;   
%%%%%%%%%%% Training Phase %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n = size(T,2);
Omega_train = kernel_matrix(P',Kernel_type, Kernel_para,P');
OutputWeight=((Omega_train+speye(n)/C)\(T')); 

%%%%%%%%%%% Calculate the training output
Y=(Omega_train * OutputWeight);                             %   Y: the actual output of the training data
end_training_time(m+1,1) = cputime;

%%%%%%%%%%% Calculate the output of testing input

start_testing_time(m+1,1) = cputime;
Omega_test = kernel_matrix(P',Kernel_type, Kernel_para,TV.P');
TY=(Omega_test' * OutputWeight);                            %   TY: the actual output of the testing data
end_testing_time(m+1,1) = cputime;


%%%%%%%%%% Calculate training & testing classification accuracy

if Elm_Type == REGRESSION
%%%%%%%%%% Calculate training & testing accuracy (RMSE) for regression case
T = T';
    for i = 1:size(Y,1)
        up1(i,1) = abs(T(i,1)-Y(i,1));
        down1(i,1) = (abs(T(i,1))+abs(Y(i,1)))/2;
        MAPE1(i,1) = up1(i,1)/down1(i,1);
        MSE1(i,1) = (T(i,1)-Y(i,1))^2;
        variance1(i,1) = (Y(i,1)-mean(T))^2;
        %MAPE1(i,1) = abs((T(i,1)-Y(i,1))/T(i,1));
    end;
Training_RMSE(1,m+1) = sqrt(sum(MSE1)/size(MSE1,1));
TrainingAccuracy_SMAPE(1,m+1) = sum(MAPE1)/size(MAPE1,1); 
Training_MSE(1,m+1) = sum(MSE1)/size(MSE1,1);
Training_NMSE(1,m+1) = sum(MSE1)/sum(variance1);
Training_MAPE(1,m+1) = sum(MAPE1)/size(MAPE1,1);

    TV.T = TV.T';
    for i = 1:size(TY,1)
        up2(i,1) = abs(TV.T(i,1)-TY(i,1));
        down2(i,1) = (abs(TV.T(i,1))+abs(TY(i,1)))/2;
        MAPE2(i,1) = up2(i,1)/down2(i,1);
        MSE2(i,1) = (TV.T(i,1)-TY(i,1))^2;
        variance2(i,1) = (TY(i,1)-mean(TV.T))^2;
        %MAPE2(i,1) = abs((TV.T(i,1)-TY(i,1))/TV.T(i,1));
    end;
Testing_RMSE(1,m+1) = sqrt(sum(MSE2)/size(MSE2,1));
TestingAccuracy_SMAPE(1,m+1) = sum(MAPE2)/size(MAPE2,1);  
Testing_MSE(1,m+1) = sum(MSE2)/size(MSE2,1);
Testing_NMSE(1,m+1) = sum(MSE2)/sum(variance2);
Testing_MAPE(1,m+1) = sum(MAPE2)/size(MAPE2,1);
end



if Elm_Type == CLASSIFIER
%%%%%%%%%% Calculate training & testing classification accuracy
    MissClassificationRate_Training=0;
    MissClassificationRate_Testing=0;

    for i = 1 : size(T, 2)
        [x, label_index_expected]=max(T(:,i));
        [x, label_index_actual]=max(Y(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Training=MissClassificationRate_Training+1;
        end
    end
    TrainingAccuracy=1-MissClassificationRate_Training/size(T,2)  
    for i = 1 : size(TV.T, 2)
        [x, label_index_expected]=max(TV.T(:,i));
        [x, label_index_actual]=max(TY(:,i));
        if label_index_actual~=label_index_expected
            MissClassificationRate_Testing=MissClassificationRate_Testing+1;
        end
    end
    TestingAccuracy=1-MissClassificationRate_Testing/size(TV.T,2);
end
right1 = train(:,predictsize+2:end);
P = P';
P = P(:,2:end);
left1 = [P,Y];
train = [left1,right1];
right2 = test(:,predictsize+2:end);
TV.P = TV.P';
TV.P = TV.P(:,2:end);
left2 = [TV.P,TY];
test = [left2,right2];
% Calculate time
step_training_time(m+1,1) =  end_training_time(m+1,1) - start_training_time(m+1,1);
step_testing_time(m+1,1) = end_testing_time(m+1,1) - start_testing_time(m+1,1);
end;   
average_train_SMAPE = mean(TrainingAccuracy_SMAPE);
average_train_RMSE = mean(Training_RMSE);
average_train_MSE = mean(Training_MSE);
average_train_NMSE = mean(Training_NMSE);
average_test_SMAPE = mean(TestingAccuracy_SMAPE);
average_test_RMSE = mean(Testing_RMSE);
average_test_MSE = mean(Testing_MSE);
average_test_NMSE = mean(Testing_NMSE);
training_time = sum(step_training_time);
testing_time = sum(step_testing_time);
result(1,:) =  TrainingAccuracy_SMAPE;
result(2,:) = TestingAccuracy_SMAPE;
result(3,:) = Training_MAPE;
result(4,:) = Testing_MAPE;
result(5,:) = Training_MSE;
result(6,:) = Testing_MSE;
result(7,:) = step_training_time;
result(8,:) = step_testing_time;
final_result(1,:) = Testing_MSE;
final_result(2,:) = TestingAccuracy_SMAPE;
time = training_time;
    
%%%%%%%%%%%%%%%%%% Kernel Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
function omega = kernel_matrix(Xtrain,kernel_type, kernel_pars,Xt)

nb_data = size(Xtrain,1);


if strcmp(kernel_type,'RBF_kernel'),
if nargin<4,
    XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
    omega = XXh+XXh'-2*(Xtrain*Xtrain');
    omega = exp(-omega./kernel_pars(1));
else
    XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
    XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
    omega = XXh1+XXh2' - 2*Xtrain*Xt';
    omega = exp(-omega./kernel_pars(1));
end

elseif strcmp(kernel_type,'lin_kernel')
if nargin<4,
    omega = Xtrain*Xtrain';
else
    omega = Xtrain*Xt';
end

elseif strcmp(kernel_type,'poly_kernel')
if nargin<4,
    omega = (Xtrain*Xtrain'+kernel_pars(1)).^kernel_pars(2);
else
    omega = (Xtrain*Xt'+kernel_pars(1)).^kernel_pars(2);
end

elseif strcmp(kernel_type,'wav_kernel')
if nargin<4,
    XXh = sum(Xtrain.^2,2)*ones(1,nb_data);
    omega = XXh+XXh'-2*(Xtrain*Xtrain');

    XXh1 = sum(Xtrain,2)*ones(1,nb_data);
    omega1 = XXh1-XXh1';
    omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));

else
    XXh1 = sum(Xtrain.^2,2)*ones(1,size(Xt,1));
    XXh2 = sum(Xt.^2,2)*ones(1,nb_data);
    omega = XXh1+XXh2' - 2*(Xtrain*Xt');

    XXh11 = sum(Xtrain,2)*ones(1,size(Xt,1));
    XXh22 = sum(Xt,2)*ones(1,nb_data);
    omega1 = XXh11-XXh22';

    omega = cos(kernel_pars(3)*omega1./kernel_pars(2)).*exp(-omega./kernel_pars(1));
end
end
