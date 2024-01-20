function [data_out] = clean2(eegdata)

t_seg = 4;

srate = 128;
Ts = 1/srate; % Hz

[channels, N] = size(eegdata);

[BBB,AAA] = butter(4,[2*(49.8)/srate, 2*(50.2)/srate],'stop');
eegdata = filter(BBB,AAA,eegdata);

maxTime = Ts*N;

T = linspace(0,maxTime,N);

% find the number of windows
% L = 0:floor(maxTime/t_seg);
% 
% EEE = zeros([length(L),3,channels]);

% % apply feature extraction
% sFE = signalTimeFeatureExtractor(SampleRate=srate,PeakValue=true,ShapeFactor=true,RMS=true);
% 
% for i=1:length(L)
%     l=L(i);
% 
%     % segment the data (this is not really windowing)
%     data_seg = eegdata(:,T>=l*t_seg & T<=(l+1)*t_seg);
% 
%     % apply ICA on each segments
%     B = jader(data_seg);
%     Data_ICA = B*data_seg;
%     
%     % to avoid dimension errors
%     if numel(data_seg)==channels
%         continue
%     end
% 
%     EEE(i,:,:) = squeeze(extract(sFE,Data_ICA'));
% 
%     I = find(squeeze(EEE(i,1,:)<1.3 & EEE(i,2,:)<5 & EEE(i,3,:)>1.5)');
%     Data_ICA( I,:) = mean(Data_ICA(I,:),2) + tanh(Data_ICA(I,:)-mean(Data_ICA(I,:),2));
% 
%     % return back ICA component domain to original channel domain
%     eegdata(:,T>=l*t_seg & T<=(l+1)*t_seg) = inv(B) * Data_ICA;
% 
% end

% segmentation time for windowing
t_seg = 20;

% find the number of windows
L = 0:floor(maxTime/t_seg);

data_out = cell(length(L),1);

% segment the data into 20-sec length windows
for i=1:length(L)
    l=L(i);

    % take the current window
    d = eegdata(:,T>=l*t_seg & T<=(l+1)*t_seg);

    % initialize signal power for each channel and each frequency band
    eegpow = zeros([channels,5]);

    for j=1:channels
        % multiresolution analysis
        mra = ewt(d(j,:),'MaxNumPeaks',5);
        % calculate the power within each frequency band
        watts = abs( fft(mra,srate)/size(mra,1)).^2;

%         figure
%         subplot(511)
%         title('EEG Signal Power Per Frequency Intervals')
%         hold on
%         plot(0:63,watts(1:64,1),'LineWidth',2)
%         subplot(512)
%         hold on
%         plot(0:63,watts(1:64,2),'LineWidth',2)
%         subplot(513)
%         hold on
%         plot(0:63,watts(1:64,3),'LineWidth',2)
%         subplot(514)
%         hold on
%         plot(0:63,watts(1:64,4),'LineWidth',2)
%         subplot(515)
%         hold on
%         plot(0:63,watts(1:64,5),'LineWidth',2)

        % remove the structured noise
        watts(40:60,:) = smoothdata(watts(40:60,:),'rloess',10);
        
%         subplot(511)
%         hold on
%         plot(0:63,watts(1:64,1),'LineWidth',2)
%         legend('Original Signal','Smoothed Signal')
%         subplot(512)
%         hold on
%         plot(0:63,watts(1:64,2),'LineWidth',2)
%         legend('Original Signal','Smoothed Signal')
%         subplot(513)
%         hold on
%         plot(0:63,watts(1:64,3),'LineWidth',2)
%         legend('Original Signal','Smoothed Signal')
%         subplot(514)
%         hold on
%         plot(0:63,watts(1:64,4),'LineWidth',2)
%         legend('Original Signal','Smoothed Signal')
%         subplot(515)
%         hold on
%         plot(0:63,watts(1:64,5),'LineWidth',2)
%         legend('Original Signal','Smoothed Signal')
        
        watts = rms(watts,1);
        if numel(watts)<5
            continue
        end
        eegpow(j,:) = watts;
    end
    data_out(i) = {eegpow};

end

% avoid complex parts
data_out = cellfun(@real,data_out,'UniformOutput',false);

end