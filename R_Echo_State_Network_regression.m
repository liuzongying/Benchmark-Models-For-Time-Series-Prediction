% A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
% in "plain" Matlab.
% by Mantas Lukosevicius 2012
% http://minds.jacobs-university.de/mantas

% load the data


data = load('norDOM36.csv');
trainLen = fix(size(data,1)*0.7);%500;
testLen = size(data,1)-trainLen;%500;
initLen = 100;
% plot some of it
% figure(10);
% plot(data(1:1000));
% title('A sample of data');

% generate the ESN reservoir
inSize = 18; outSize = 1;
a = 0.95; % leaking rate
% resSize = 10;
period = 18;
predictsize = 18;
train=data(1:trainLen,:);
test=data(trainLen+1:end,:);
resSize = size(train, 1);

for m = 0:period-1
    

train_data = train;
T=train_data(:,predictsize+1)';
P=train_data(:,1:predictsize)';
                                %   Release raw training data array
test_data=test;
TV.T=test_data(:,predictsize+1)';
TV.P=test_data(:,1:predictsize)';    
    
rand( 'seed', 45 );
Win = (rand(resSize,inSize+1)-0.5);
W = rand(resSize,resSize)-0.5;
% Option 1 - direct scaling (quick&dirty, reservoir-specific):
% W = W .* 0.13;
% Option 2 - normalizing and setting spectral radius (correct, slower):
disp 'Computing spectral radius...';
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
disp 'done.'
W = W .* ( 1.25 /rhoW);

% allocated memory for the design (collected states) matrix
X = zeros(1+inSize+resSize,trainLen-initLen);
% set the corresponding target matrix directly
Yt = data(initLen+2:trainLen+1,inSize+1:inSize+outSize)';

% run the reservoir with the data and collect X
x = zeros(resSize,1);
for t = 1:trainLen
	u = data(t,1:inSize);
	x = (1-a)*x + a*tanh( Win*[1,u]' + W*x );
	if t > initLen
		X(:,t-initLen) = [1,u,x'];
	end
end

% train the output
reg = 1e-3;  % regularization coefficient
X_T = X';
Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
% Wout = Yt*X_T* reg;% * reg*eye(1+inSize+resSize)^(-1);

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.

%%% Training
Y = zeros(trainLen,outSize);
% u = data(trainLen+1);
for t = 1:trainLen 
    u = data(t,1:inSize);
	x = (1-a)*x + a*tanh( Win*[1,u]' + W*x );
	y = Wout*[1,u,x']';
	Y(t,:) = y';
    clear y;
	% generative mode:
	% u = y;
	% this would be a predictive mode:
	% u = data(trainLen+t+1);
end

%%% Testing
TY = zeros(testLen,outSize);
% u = data(trainLen+1);
for t = 1:testLen 
    u = data(trainLen+t,1:inSize);
	x = (1-a)*x + a*tanh( Win*[1,u]' + W*x );
	y = Wout*[1,u,x']';
	TY(t,:) = y';
	% generative mode:
	% u = y;
	% this would be a predictive mode:
	% u = data(trainLen+t+1);
end
%%%% Calculate SMAPE and MSE
%Traiing data

    T = T';
    for i = 1:size(T,1)
        up1(i,1) = abs(T(i,1)-Y(i,1));
        down1(i,1) = (abs(T(i,1))+abs(Y(i,1)))/2;
        MAPE1(i,1) = up1(i,1)/down1(i,1);
        MSE1(i,1) = (T(i,1)-Y(i,1))^2;
        %variance1(i,1) = (Y(i,1)-mean(T))^2;
    end;
TrainingAccuracy_SMAPE(1,m+1) = sum(MAPE1)/size(MAPE1,1); 
Training_RMSE(1,m+1) = sqrt(sum(MSE1)/size(MSE1,1));
Training_MSE(1,m+1) = sum(MSE1)/size(MSE1,1);
%Training_NMSE(1,m+1) = sum(MSE1)/sum(variance1);
Training_NMSE(1,m+1) = Training_MSE(1,m+1)/var(T);    


% Testing data        
      
TV.T = TV.T';
     for i = 1:size(TY,1)
        up2(i,1) = abs(TV.T(i,1)-TY(i,1));
        down2(i,1) = (abs(TV.T(i,1))+abs(TY(i,1)))/2;
        MAPE2(i,1) = up2(i,1)/down2(i,1);
        MSE2(i,1) = (TV.T(i,1)-TY(i,1))^2;
        %variance2(i,1) = (TY(i,1)-mean(TV.T))^2;
        
    end;

TestingAccuracy_SMAPE(1,m+1) = sum(MAPE2)/size(MAPE2,1);
Testing_RMSE(1,m+1) = sqrt(sum(MSE2)/size(MSE2,1));
Testing_MSE(1,m+1) = sum(MSE2)/size(MSE2,1);
%Testing_NMSE(1,m+1) = sum(MSE2)/sum(variance2);
Testing_NMSE(1,m+1) = Testing_MSE(1,m+1)/var(TV.T);
     
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
clear beta

end
average_train_SMAPE = mean(TrainingAccuracy_SMAPE);
average_train_RMSE = mean(Training_RMSE);
average_train_MSE = mean(Training_MSE);
average_train_NMSE = mean(Training_NMSE);
average_test_SMAPE = mean(TestingAccuracy_SMAPE);
average_test_RMSE = mean(Testing_RMSE);
average_test_MSE = mean(Testing_MSE);
average_test_NMSE = mean(Testing_NMSE);


result(1,:) = Testing_MSE;
result(2,:) = TestingAccuracy_SMAPE;

actural = TV.T;


%%plot
plot(actural,'b')
hold on
plot(TY, 'r')
ylabel('Value')
xlabel('Time series')
hold off
legend('Actual Values','Forecasting Values')
