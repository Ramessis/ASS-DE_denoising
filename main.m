function main()
clc
close all
format long
format compact

%% Read image and add noise
imn = zeros(550,1000,3,3);

% Normalize pixel values to 0-1
img = (im2double((imread('remote.bmp'))));

imn1 = imnoise(img,'salt & pepper',0.2);
imn2 = imnoise(img,'gaussian',0,0.2);
imn3 = imnoise(img,'speckle',0.2);

% noised image selection
noised_Num = 3;
imn(:,:,:,1) = imn1;imn(:,:,:,2) = imn2;imn(:,:,:,3) = imn3;

% parameter setting
N = 100;%population size
TEV=Error(Dim(1));
runmax = 1;
TestFitness=[];
TestResult=[];
TestValue={};
TestTime=[];%TestTime: the time spent by each algorithm 
TestRatio=[];
TestFES=[];%TestFES: the FES required to satisfy the conditions
TestOptimization={};
TestParameter={};%TestParameter:the optimal value produced by each 10000 FES

%% ASS-DE
name = 'ASS_DE';
for desnosingProblem= 1:noised_Num
    [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=ASS_DE(img,imn(:,:,:,desnosingProblem),desnosingProblem,N,runmax);
    
    sign=(RunResult<=TEV(desnosingProblem));
    Ratio=sum(sign)/runmax;
    FES=sign.*RunFES;
    TestFitness=[TestFitness;RunResult];
    TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
    TestValue=[TestValue;mean(RunValue)];
    TestTime=[TestTime;mean(RunTime)];
    TestRatio=[TestRatio;Ratio];
    TestFES=[TestFES;mean(FES)];
    TestOptimization=[TestOptimization;RunOptimization];
    TestParameter=[TestParameter;RunParameter];
    
    % experimental result
    TestResult=[TestResult;min(RunResult) max(RunResult) median(RunResult) mean(RunResult) std(RunResult)];
    TestValue=[TestValue;mean(RunValue)];
    
end

Test=sprintf('%s.mat',[name,'_TestFitness']);
save(Test,'TestFitness');
Test=sprintf('%s.mat',[name,'_TestResult']);
save(Test,'TestResult');
Test=sprintf('%s.mat',[name,'_TestValue']);
save(Test,'TestValue');
Test=sprintf('%s.mat',[name,'_TestTime']);
save(Test,'TestTime');
Test=sprintf('%s.mat',[name,'_TestRatio']);
save(Test,'TestRatio');
Test=sprintf('%s.mat',[name,'_TestFES']);
save(Test,'TestFES');
Test=sprintf('%s.mat',[name,'_TestOptimization']);
save(Test,'TestOptimization');
Test=sprintf('%s.mat',[name,'_TestParameter']);
save(Test,'TestParameter');


