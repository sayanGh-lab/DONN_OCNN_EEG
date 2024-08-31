% epilepcy vs. normal
%%%%%%%%%%% REM*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear all
close all
load('dataset') 
% filtering %
fs=256;

% normalize_signals1_ep = zeros(144,1,512);
 ep_sig=EEG_ep;

%%normal
 EEG_he=double(EEG_he);
cutoff_freq = 40; % Cutoff frequency of the low-pass filter (Hz)
filter_order = 3; % Order of the filter

% Design the low-pass filter
[b, a] = butter(filter_order, cutoff_freq/(fs/2), 'low');
        
filtered_signals1 = zeros(size(EEG_ep,1),17,512);
    for i=1:size(EEG_ep,1)
        for j=1:17
            filtered_signals1(i,j,:) = filter(b, a, EEG_he(i,j,:));
%             filtered_signals1(i,j,:) = filtered_signals1(i,j,:)/max(abs(filtered_signals1(i,j,:)));
%             filtered_signals1(i,j,:) = filtered_signals1(i,j,:) - mean(filtered_signals1(i,j,:));
        end
      % total signal
    end
he_sig=filtered_signals1;
he2=squeeze(he_sig(2,:,:));   

%%%%%%%% labeling%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % create label
fs=256;
dt = 1/fs;
 t =[0:1:64-1]*dt;
 
 a=8*ones(1,64);
 Ydp=t.*a;



% label
Yd11=[Ydp;zeros(1,64)]; % label for REM
Yd11=Yd11';
Yd_f1=zeros(size(EEG_ep,1),64,2);
% label for nrem3(test)


Yd2=[zeros(1,64);Ydp]; % label for NREMN3
Yd2=Yd2';
Yd_f2=zeros(size(EEG_ep,1),64,2);

for ii=1:size(EEG_ep,1)
Yd_f1(ii,:,:)=Yd11;
% Yd_f2(ii,:,:)=Yd2;

end

for ii=1:size(EEG_ep,1)
% Yd_f1(ii,:,:)=Yd1;
Yd_f2(ii,:,:)=Yd2;

end

Yd=[Yd_f1;Yd_f2];  % main labelling 
EEG=[ep_sig;he_sig];
 EEG1=EEG(:,:,:);

%%%%%%% shuffling

points=zeros(size(EEG_ep,1)+size(EEG_ep,1),1);
for iii=1:length(points)
    points(iii,:)=iii;
end
index = shuffle(points);  % generate random index

Yd1=Yd(index,:,:);        % reorganize the label
EEG1=EEG1(index,:,:); % main data (training)

%% 8 sec window
num_sample=512;
num_win=8;
itration=num_sample/num_win;

for i=1:size(EEG_ep,1)+size(EEG_ep,1)
    eeg4=EEG1(i,:,:);
    eeg5=squeeze(eeg4);
    for ii=1:itration
        eeg8=eeg5(:,(ii-1)*num_win+1:ii*num_win);
        eeg9=mean(eeg8,2);
        eeg9=eeg9';
        eeg10(ii,:)=eeg9;
        
    end
%     eeg10=eeg10';
      eeg11(i,:,:)=eeg10;
end

EEG_main=eeg11;
label=Yd1;


 save model_dataset EEG_main label index
