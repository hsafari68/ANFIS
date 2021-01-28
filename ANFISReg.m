clc;
clear;
close all;


%% Load Data
% load Data_Krw;
Data = xlsread('Data.xlsx');
% Data = Data_Krw;
X = Data(:,1:end-1);
Y = Data(:,end);

DataNum = size(X,1);
InputNum = size(X,2);
OutputNum = size(Y,2);

%% Normalization 
MinX = min(X);
MaxX = max(X);

MinY = min(Y);
MaxY = max(Y);

XN = X;
YN = Y;

% for ii = 1:InputNum
%     XN(:,ii) = Normalize(X(:,ii),MinX(ii),MaxX(ii));
% end
% 
% for ii = 1:OutputNum
%     YN(:,ii) = Normalize(Y(:,ii),MinY(ii),MaxY(ii));
% end


%% Seperate Train and Test data
TrPercent = 70;
ChkPercent=15;
TrNum = round(DataNum * TrPercent / 100);
ChkTsNum = DataNum - TrNum;
ChkNum=round(DataNum * ChkPercent / 100);
TsNum=ChkTsNum-ChkNum;

R = randperm(DataNum);
TrIndex = R(1 : TrNum);
ChkIndex = R(1+TrNum : TrNum+ChkNum);
TsIndex= R(1+TrNum+ChkNum:end);

TrainInputs = XN(TrIndex,:);
TrainTargets = YN(TrIndex,:);

TrainData=[TrainInputs TrainTargets];

ChkInputs = XN(ChkIndex,:);
ChkTargets = YN(ChkIndex,:);

ChkData=[ChkInputs ChkTargets];


TestInputs = XN(TsIndex,:);
TestTargets = YN(TsIndex,:);
TestData=[TestInputs TestTargets];


%% Design ANFIS

Option{1}='Grid Part.(genfis1)';
Option{2}='Sub. Clustering(genfis2)';
Option{3}='FCM(genfis3)';
Answer=questdlg('Select FIs Generation Method?','Select GENFIS'...
    ,Option{1},Option{2},Option{3},Option{3});

switch Answer
    case Option{1}
        prompt={'Number of MFs','Input MF Type','Output Mf Type'};
        title='Enter genfis1 Variables';
        defaults={'10','gaussmf','linear'};
        Params=inputdlg(prompt,title,1,defaults);
        nMFs=str2num(Params{1}); %#ok
        InputMF=Params{2};
        OutputMF=Params{3};
        fis=genfis1(TrainData,nMFs,InputMF,OutputMF);
    case Option{2}
        prompt={'Effective Radius'};
        title='Enter genfis2 Variables';
        defaults={'0.2'};
        Params=inputdlg(prompt,title,1,defaults);
        Radius=str2num(Params{1}); %#ok
        fis=genfis2(TrainInputs,TrainTargets,Radius);   
    case Option{3}
        prompt={'Number of Clusters:',...
                'Partition Matrix Component:',...
                'Maximum Number of Iterations:',...
                'Minimum Improvement:'};
        title='Enter genfis3 Variables';
        defaults={'10','2','100','1e-5'};
        Params=inputdlg(prompt,title,1,defaults);
        Cluster_n=str2num(Params{1}); %#ok
        Exponent=str2num(Params{2}); %#ok
        MaxIt=str2num(Params{3}); %#ok
        MinImprovment=str2num(Params{4}); %#ok
        DisplayInfo=1;
        FCMOptions=[Exponent MaxIt MinImprovment DisplayInfo];
        fis=genfis3(TrainInputs,TrainTargets,'sugeno',Cluster_n,FCMOptions);        
end

prompt={'Maximum Number of Epochs:',...
'Error Goal:',...
'Initial Step Size:',...
'Step Size Decrease Rate:',...
'Step Size Increase Rate'};
title='Enter ANFIS Variables';
defaults={'100','0','0.01','0.9','1.1'};
Params=inputdlg(prompt,title,1,defaults);
MaxEpoch=str2num(Params{1}); %#ok
ErrorGoal=str2num(Params{2}); %#ok
InitialStepSize=str2num(Params{3}); %#ok
StepSizeDecreaseRate=str2num(Params{4}); %#ok
StepSizeIncreaseRate=str2num(Params{5}); %#ok
TrainOptions=[MaxEpoch ErrorGoal InitialStepSize ...
    StepSizeDecreaseRate StepSizeIncreaseRate]; %NaN default

DisplayInfo=true;  % or false
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;

DisplayOptions=[DisplayInfo DisplayError ...
    DisplayStepSize DisplayFinalResult];

OptimizationMethod=1;
% 0: Backpropagation
% 1: Hybrid


[fis,error,stepsize,chkFis,chkErr]=anfis(TrainData,fis,TrainOptions,DisplayOptions,ChkData,OptimizationMethod);


%% Applying ANFIS to Train Data

TrainOutputs=evalfis(TrainInputs,fis);
TrainErrors=TrainTargets-TrainOutputs;
TrainMSE=mean(TrainErrors(:).^2);
TrainRMSE=sqrt(TrainMSE);
TrainErrorMean=mean(TrainErrors);
TrainErrorSTD=std(TrainErrors);

AvgRltvErr_Train = (TrainTargets - TrainOutputs) ./ TrainTargets;
AvgRltvErr_Train = 100 * mean(AvgRltvErr_Train);
% Finding the average absolute error for different data divisions
AvgAbsErr_Train = abs(TrainTargets - TrainOutputs) ./ abs(TrainTargets);
AvgAbsErr_Train = 100 * mean(AvgAbsErr_Train);

figure;
PlotResults(TrainTargets,TrainOutputs,'Train Data');

plotregression(TrainTargets,TrainOutputs,'Train Data');
set(gcf,'Toolbar','figure');

%% Applying ANFIS to Check Data

ChkOutputs=evalfis(ChkInputs,fis);
ChkErrors=ChkTargets-ChkOutputs;
ChkMSE=mean(ChkErrors(:).^2);
ChkRMSE=sqrt(ChkMSE);
ChkErrorMean=mean(ChkErrors);
ChkErrorSTD=std(ChkErrors);

AvgRltvErr_Chk = (ChkTargets - ChkOutputs) ./ ChkTargets;
AvgRltvErr_Chk = 100 * mean(AvgRltvErr_Chk);
% Finding the average absolute error for different data divisions
AvgAbsErr_Chk = abs(ChkTargets - ChkOutputs) ./ abs(ChkTargets);
AvgAbsErr_Chk = 100 * mean(AvgAbsErr_Chk);

figure;
PlotResults(ChkTargets,ChkOutputs,'Chk Data');

plotregression(ChkTargets,ChkOutputs,'Chk Data');
set(gcf,'Toolbar','figure');

%% Applying ANFIS to Test Data

TestOutputs=evalfis(TestInputs,fis);
TestErrors=TestTargets-TestOutputs;
TestMSE=mean(TestErrors(:).^2);
TestRMSE=sqrt(TestMSE);
TestErrorMean=mean(TestErrors);
TestErrorSTD=std(TestErrors);

AvgRltvErr_Test = (TestTargets - TestOutputs) ./ TestTargets;
AvgRltvErr_Test = 100 * mean(AvgRltvErr_Test);
% Finding the average absolute error for different data divisions
AvgAbsErr_Test = abs(TestTargets - TestOutputs) ./ abs(TestTargets);
AvgAbsErr_Test = 100 * mean(AvgAbsErr_Test);

figure;
PlotResults(TestTargets,TestOutputs,'Test Data');


plotregression(TestTargets,TestOutputs,'Test Data');
set(gcf,'Toolbar','figure');

%% Applying ANFIS to Total Data

TotalInputs=XN;
TotalTargets=YN;

TotalOutputs=evalfis(TotalInputs,fis);
TotalErrors=TotalTargets-TotalOutputs;
TotalMSE=mean(TotalErrors(:).^2);
TotalRMSE=sqrt(TotalMSE);
TotalErrorMean=mean(TotalErrors);
TotalErrorSTD=std(TotalErrors);

AvgRltvErr_Total = (TotalTargets - TotalOutputs) ./ TotalTargets;
AvgRltvErr_Total = 100 * mean(AvgRltvErr_Total);
% Finding the average absolute error for different data divisions
AvgAbsErr_Total = abs(TotalTargets - TotalOutputs) ./ abs(TotalTargets);
AvgAbsErr_Total = 100 * mean(AvgAbsErr_Total);

figure;
PlotResults(TotalTargets,TotalOutputs,'Total Data');


plotregression(TotalTargets,TotalOutputs,'Total Data');
set(gcf,'Toolbar','figure');


%% Check overfitting
figure;
plot(chkErr,'r');
hold on
plot(error);
hold off


%% Outlier detection

OrdinaryRes=TotalTargets-TotalOutputs;
RSS=sum(OrdinaryRes.^2);
dfE=DataNum-InputNum-1;
stdDev2=RSS/dfE;
h=(1/DataNum)+[((TotalTargets-mean(TotalTargets)).^2)/sum(((TotalTargets-mean(TotalTargets)).^2))];
standardizedresiduals=OrdinaryRes./(sqrt(1-h)*sqrt(stdDev2));
% d={'Actual','Predicted','Hat2','Residual'};
% xlswrite('Lith.xls', d, 1, 'A1');
% xlswrite('Lith.xls', [TotalTargets,TotalOutputs,h,standardizedresiduals], 1,'A2')
figure;
plot(h,standardizedresiduals,'*r')
xlim([0,0.3]);
hold on
plot(0:0.00001:1,3,'b','LineWidth',3)
hold on
plot(0:0.00001:1,-3,'b','LineWidth',3)
hold off


%% Visualization

figure;
[ntr,ytr] = hist(TrainErrors,30);
bar(ytr,ntr,'r');
hold on
[nts,yts] = hist(TestErrors,30);
bar(yts,nts,'b');
xlabel('\bfRelative Error');
ylabel('\bfFrequency');
legend('Train','Test');


figure;
plot(TrainTargets,TrainOutputs,'b*');
hold on
plot(TestTargets,TestOutputs,'r^')
hold on
plot(ChkTargets,ChkOutputs,'go')
hold on
plot([0,1],[0,1],'k-','LineWidth',2);
hold off
ylim([0 1]);
xlabel('\bfK_{rw} ^{Experimental}');
ylabel('\bfK_{rw} ^{ANFIS}');
legend('Train','Test','Check');



figure;
plot(TrainTargets,'-^k');
hold on
plot(TrainOutputs,'-or');
hold off
xlabel('\bfTrain Data Index');
ylabel('\bfK_{rw}');
legend('Experimental','ANFIS');
ylim([0 1]);

figure;
plot(TestTargets,'-^k');
hold on
plot(TestOutputs,'-or');
hold off
xlabel('\bfTest Data Index');
ylabel('\bfK_{rw}');
legend('Experimental','ANFIS');
ylim([0 1]);

figure;
plot(ChkTargets,'-^k');
hold on
plot(ChkOutputs,'-or');
hold off
xlabel('\bfCheck Data Index');
ylabel('\bfK_{rw}');
legend('Experimental','ANFIS');
ylim([0 1]);