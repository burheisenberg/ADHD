function [eegdata] = clean(eegdata)

t_seg = 4;

srate = 128;
Ts = 1/srate; % Hz

[channels, N] = size(eegdata);

maxTime = Ts*N;

T = linspace(0,maxTime,N);

% find the number of windows
L = 0:floor(maxTime/t_seg);

EEE = zeros([length(L),3,channels]);

% apply feature extraction
sFE = signalTimeFeatureExtractor(SampleRate=srate,PeakValue=true,ShapeFactor=true,RMS=true);

for i=1:length(L)
    l=L(i);

    % segment the data (this is not really windowing)
    data_seg = eegdata(:,T>=l*t_seg & T<=(l+1)*t_seg);

    % apply ICA on each segments
    B = jader(data_seg);
    Data_ICA = B*data_seg;
    
    % to avoid dimension errors
    if numel(data_seg)==channels
        continue
    end

    EEE(i,:,:) = squeeze(extract(sFE,Data_ICA'));

    I = find(squeeze(EEE(i,1,:)<1.3 & EEE(i,2,:)<5 & EEE(i,3,:)>1.5)');
    Data_ICA( I,:) = tanh(Data_ICA(I,:));

    % return back ICA component domain to original channel domain
    eegdata(:,T>=l*t_seg & T<=(l+1)*t_seg) = inv(B) * Data_ICA;

end

% avoid complex parts
eegdata = real(eegdata);

end