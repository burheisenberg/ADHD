close all, clear all, clc

srate = 128;
Ts = 1/srate; % sampling rate of 128 Hz

% import the patient's data
name = 'v30p';
struct = load([name '.mat']);
eegdata = getfield(struct,name);

use_jader = 1;
use_notch = 1;

[N, channels] = size(eegdata);
channels = 19;
q = 19;

maxTime = Ts*N;

T = linspace(0,maxTime,N);

% Kmax=floor(maxTime/2);


% HFD_Array = HFD(eegdata,Kmax);


figure(1)
for i=1:channels
        subplot(4,ceil(channels/4),i)
        plot(T,eegdata(:,i))
        title(['channel#' num2str(i)])
end

if use_notch
    % bandstop filter at 48-52 Hz
    [BBB,AAA] = butter(4,[2*(49.8)/srate, 2*(50.2)/srate],'stop');
    eegdata = filter(BBB,AAA,eegdata);
    % fvtool(BBB,AAA)
end

% PCA
% reference: https://youtu.be/GgLaP4Des1Q
% matlab's pca automatically normalizes the data
% 
% [coeff, Data_PCA, latent, tsquared, explained, mu] = pca(eegdata,'NumComponents',q);
% 
% disp(strcat("Top ", string(q), " principle components explain ", ...
%     string(sum(explained(1:q))), " of variation."))

%% ICA
if ~use_jader
    % compute the independent components from original data
    Mdl = rica(eegdata, q);
    % apply the transformation
    Data_ICA = transform(Mdl,eegdata);
%     HFD_ICA_Array = HFD(Data_ICA,Kmax);

%     I = HFD_ICA_Array<1.0;

%     Data_ICA(:,I) = zeros(size(Data_ICA(:,I)));

    reconstructed = Data_ICA * inv(Mdl.TransformWeights);
else
    B = jader(eegdata');
    Data_ICA = eegdata*B';

    % nonlinear feature extraction for ocular activity
%     HFD_ICA_Array = HFD(Data_ICA,Kmax);
%     I = HFD_ICA_Array<1.7;
%     Data_ICA(:,I) = zeros(size(Data_ICA(:,I)));


    % bandstop filter at 48-52 Hz
%     [BBB,AAA] = butter(4,[2*(49.8)/srate, 2*(50.2)/srate],'stop');
%     Data_ICA = filter(BBB,AAA,Data_ICA);

    reconstructed = Data_ICA * inv(B');
end


figure(1)
for i=1:channels
        subplot(4,ceil(channels/4),i)
        hold on
        plot(T,reconstructed(:,i))
        title(['channel#' num2str(i)])
end

figure(2)
for i=1:q
    subplot(4,ceil(q/4),i)
    plot(T,Data_ICA(:,i))
    title(['component#' num2str(i)])
end

%% energies of the components

hz = linspace(0,srate/2,floor(N/2)+1);
eegpow = abs( fft(reconstructed)/N).^2;
I = find(hz>40 & hz<srate);
%eegpow(I,:) = filloutliers(eegpow(I,:)','makima',1e3)';
%eegpow(I,:) = smoothdata(eegpow(I,:),'movmedian',1e3);
% for i=1:channels
%     p = polyfit(hz(I),eegpow(I,i),1);
%     eegpow(I,i) = polyval(p,hz(I));
% end
hilb = abs(hilbert(eegpow));


% figure(3)
% for i=1:channels
%     subplot(channels,1,i)
%     hold on
%     plot(hz,eegpow(1:length(hz),i))
% %     plot(hz,hilb(1:length(hz),i))
%     set(gca,'xlim',[0 srate/2], 'yscale', 'log')
% end


figure(4), clf
for i=1:channels
        subplot(4,ceil(channels/4),i)
        hold on
        plot(hz,eegpow(1:length(hz),i))
        plot(hz,hilb(1:length(hz),i))
%         plot([hz,hz(end)+hz(1:end-2)],eegpow(:,i))
%         plot([hz,hz(end)+hz(1:end-2)],hilb(:,i))
        set(gca,'xlim',[0.1 srate/2], 'YScale','log')
        title(['component#' num2str(i)])
        legend('Signal Power', 'Power Envelope','location','best')
end

%% generalized eigendecomposition for feature extraction

covR = cov(eegdata(6200:10000,:));
covS = cov(eegdata(14000:15000,:));

figure(5), clf
% S matrix
subplot(131)
imagesc(covS)
title('S matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R matrix
subplot(132)
imagesc(covR)
title('R matrix')
axis square, set(gca,'clim',[-1 1]*1e6)

% R^{-1}S
subplot(133)
imagesc(inv(covR)*covS)
title('R^-^1S matrix')
axis square, set(gca,'clim',[-10 10])

[evecs,evals] = eig(covS,covR);

figure(6), clf
plot(max(evals),'ms-')

% plot the eigenspectrum
% figure(6), clf
% subplot(231)
% plot(evals./max(evals),'s-','markersize',15,'markerfacecolor','k')
% axis square
% set(gca,'xlim',[0 20.5])
% title('GED eigenvalues')
% xlabel('Component number'), ylabel('Power ratio (norm-\lambda)')

% component time series is eigenvector as spatial filter for data
% comp_ts = evecs(:,end)'*eegdata(1,:)';

%% APPLY NONLINEAR FEATURE EXTRACTION or SPECTRAL FEATURE EXTRACTION

