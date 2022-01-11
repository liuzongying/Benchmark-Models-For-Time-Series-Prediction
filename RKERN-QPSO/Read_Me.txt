recurrent kernel reservoir machine with QPSO for time series prediction model

run the algorithm in MATLAB using the following command:
1. run recurrent kernel reservoir machine
[average_test_SMAPE,average_test_RMSE,average_test_MSE,average_test_NMSE,average_train_SMAPE,average_train_RMSE,average_train_MSE,average_train_NMSE,Training_RMSE,TrainingAccuracy_SMAPE, Training_MSE,Training_NMSE,Testing_RMSE,TestingAccuracy_SMAPE, Testing_MSE,Testing_NMSE,training_time, testing_time,result,time] = Recurrent_KELM_with_reservoir_computing_2('norlorenz.csv', 0, 'RBF_kernel',9.138,0.02)


In this funtion, Recurrent_KELM_with_reservoir_computing_2(Data_File, Elm_Type, kernel_type, kernel_para,a)


where, Data_File = 'the file of your data set';
       Elm_Type = 0, which is for regression;
       kernel_type = the type of method;
       kernel_para = kernel parameter;
       a = leaking rate of reservoir;

2. Run recurrent kernel reservoir machine with QPSO

   2.1, open file named 'qpso_one_max_iteration_loop'
   2.2, fill in the specific data_file name for change data set in line 63 and line 146 before runing code;
   3.3, direct run code;