%num trials to run
nTrials=5000;

%initialize empty vectors
choice=zeros(nTrials,1);
correct=zeros(nTrials,1);
QC=zeros(nTrials,1);
delta=zeros(nTrials,1);

alpha = 0.2; % learning rate

%%
%%Generate auditory trials

DV=zeros(nTrials,1);

for i = 1:nTrials
    AuditoryAlpha=1;
    LeftBiasAud=0.5;
    BetaRatio = (1 - min(0.9,max(0.1,LeftBiasAud))) / min(0.9,max(0.1,LeftBiasAud));
    %use a = ratio*b to yield E[X] = LeftBiasAud using Beta(a,b) pdf
    %cut off between 0.1-0.9 to prevent extreme values (only one side) and div by zero

    BetaA =  (2*AuditoryAlpha*BetaRatio) / (1+BetaRatio); %make a,b symmetric around AuditoryAlpha to make B symmetric
    BetaB = (AuditoryAlpha-BetaA) + AuditoryAlpha;

    AuditoryOmega=betarnd(max(0,BetaA),max(0,BetaB),1,1);


    LeftClickRate = round(AuditoryOmega.*300); 
    RightClickRate= round((1-AuditoryOmega).*300);


    LeftClickTrain = GeneratePoissonClickTrain(LeftClickRate, 0.3);
    RightClickTrain= GeneratePoissonClickTrain(RightClickRate, 0.3);


    DV(i)= (length(LeftClickTrain) - length(RightClickTrain))./(length(LeftClickTrain) + length(RightClickTrain));
end

%%  to assign variable rewards in each trial 
%%Reward Trials
%variable rewards
% [leftReward, leftSplit]=generate_rewardTrain(25, 80, 15, 18, 2.8);
% [rightReward, rightSplit]=generate_rewardTrain(25, 80, 15, 18, 2.8);
% 
% 
% leftSplit=leftSplit(1:nTrials);
% rightSplit=rightSplit(1:nTrials);
% 
% leftReward=leftReward(1:nTrials);
% rightReward=rightReward(1:nTrials);


%%
%Simulate Task

rightReward=25; %constant reward amount for left and right port
leftReward=25;

VL=sum(rightReward+leftReward)/2; %inital value estimate is the average reward for the session
VR=sum(rightReward+leftReward)/2;
%VL=sum(vertcat(leftReward,rightReward))/numel(vertcat(leftReward,rightReward));
%VR=sum(vertcat(leftReward,rightReward))/numel(vertcat(leftReward,rightReward));

for i=1:nTrials
    
    x=linspace(-1,1,21);
    likelihood=pdf('Normal',x,DV(i)+normrnd(0,0.35),0.35); %noisy percept estimat
    prior=unifpdf(x,-1,1); %prior is uniform
    
    posterior=likelihood.*prior/sum(likelihood.*prior);
    
    %figure;
    %bar(x, posterior)
    %xline(DV(i), 'linewidth',2, 'color', 'r')
    
    
    pL=sum(posterior(x>=0));
    pR=1-pL;

    QL=pL*VL;
    QR=pR*VR;
    
    if QL >= QR && DV(i)>=0 %correct left
        choice(i)=1;
        correct(i)=1;
        QC(i)=QL;
    elseif QR > QL && DV(i)<0 %correct right
        choice(i)=0;
        correct(i)=1;
        QC(i)=QR;
        
    elseif QL >= QR && DV(i)<0 %incorrect left
        choice(i)=1;
        correct(i)=0;
        QC(i)=QL;
    elseif QR > QL && DV(i)>=0 %incorrect right
        choice(i)=0;
        correct(i)=0;
        QC(i)=QR;
    end
    
    if choice(i) == 1 && correct(i)==1  %if choice is left
        delta(i)=leftReward-QC(i);
        VL=VL+delta(i)*alpha;
    elseif choice(i) == 0 && correct(i)==1 %if choice is right
        delta(i)=rightReward-QC(i);
        VR=VR+delta(i)*alpha;
        
    elseif choice(i) == 1 && correct(i)==0 %if left choice and incorrect - this version does not store Q of incorrect stimulus-action pair
        delta(i)=0-QC(i);
    elseif choice(i) == 0 && correct(i)==0
        delta(i)=0-QC(i);
    end        
end
    
%%
%Plot Psychometric
%figure;
AudBin = 8;
BinIdx = discretize(DV,linspace(-1,1,AudBin+1));
PsycY = grpstats(choice,BinIdx,'mean');
PsycX = unique(BinIdx)/AudBin*2-1-1/AudBin;
OutcomePlot=scatter(PsycX, PsycY);
hold on;

PsycAudFit_XData = linspace(min(DV),max(DV),100);
PsycAudFit_YData = glmval(glmfit(DV,choice,'binomial'),linspace(min(DV),max(DV),100),'logit');

plot(PsycAudFit_XData, PsycAudFit_YData);
xlabel('DV')
ylabel('%LeftChoice')
%%
%%Plot QC vs DV
figure;
AudBin = 3;
BinIdx = discretize(abs(DV(correct==0)),linspace(0,0.4,AudBin+1));
PsycY = grpstats(QC(correct==0),BinIdx,'mean');
X_dummy=linspace(0,0.4,AudBin+1);
PsycX = X_dummy(1:end-1)+(X_dummy(2)-X_dummy(1))/2;
QCPlot=plot(PsycX(1:length(PsycY)), PsycY);
hold on;

xlabel('DV')
ylabel('QC')

BinIdx = discretize(abs(DV(correct==1)),linspace(0,0.4,AudBin+1));
X_dummy=linspace(0,0.4,AudBin+1);
PsycX = X_dummy(1:end-1)+(X_dummy(2)-X_dummy(1))/2;
PsycY = grpstats(QC(correct==1),BinIdx,'mean');
plot(PsycX, PsycY);
%% 
%%Plot delta vs. DV

figure; 
AudBin = 5;
BinIdx = discretize(abs(DV(correct==0)),linspace(0,0.5,AudBin+1)); %for error 
PsycY = grpstats(delta(correct==0),BinIdx,'mean');
X_dummy=linspace(0,0.5,AudBin+1);
PsycX = X_dummy(1:end-1)+(X_dummy(2)-X_dummy(1))/2;
QCPlot=plot(PsycX(1:length(PsycY)), PsycY);
hold on;

xlabel('DV')
ylabel('Delta')

BinIdx = discretize(abs(DV(correct==1)),linspace(0,0.5,AudBin+1)); %for correct
X_dummy=linspace(0,0.5,AudBin+1);
PsycX = X_dummy(1:end-1)+(X_dummy(2)-X_dummy(1))/2;
PsycY = grpstats(delta(correct==1),BinIdx,'mean');
plot(PsycX, PsycY);
%%



function [trialReward,trialSplit]=generate_rewardTrain(waterReward, trialsperbout, boutSD, numBouts, banditSigma)
   
    boutLen=num2cell(round(normrnd(trialsperbout,boutSD,numBouts,1))); %generate n bouts per session of trial len 80 ~gaussian distrib
    trialSplit=cellfun(@(x)  ones(x,1), boutLen, 'UniformOutput', false); %generate separate trials

    idx = rand(length(boutLen),1); %randomly assign splits to low, med, high reward

       for i = 1:length(idx)
            if idx(i)<=0.3
                rewardSize=waterReward*0.50;
                trialSplit(i)=cellfun(@(x) x*rewardSize,trialSplit(i),'un',0);
            elseif idx(i)<=0.6
                rewardSize=waterReward*0.75;
                trialSplit(i)=cellfun(@(x) x*rewardSize,trialSplit(i),'un',0);
            else
                rewardSize=waterReward;
                trialSplit(i)=cellfun(@(x) x*rewardSize,trialSplit(i),'un',0);
            end
       end

    trialSplit=vertcat(trialSplit{:});

    decay=0.9836; %add noise centered around the average reward
    noise=normrnd(0,banditSigma,[length(trialSplit),1]);
    trialReward=trialSplit*decay+(1-decay)*mean(trialSplit)+noise;
end

