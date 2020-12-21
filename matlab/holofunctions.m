% 1- set screen/projector and 'object' geometry
% 1- place a 'real' object in front of the screen (one light point) and calculate intensity and phase distribution at the 'pixels' of the screen
% 2- create
% randomization mask will be used

clear all;
%tstart=tic;
disp('init params')

project_phase = 1;                % bool to toggle projector type (amp+phase or amp+referece)
projection = 1;                   % bool to toggle projection simulation
add_reference = 1 - project_phase % bool to add reference wave (in case of projector without phase)

screenXstep=7e-6;         % pixel size in X and Y
screenYstep=7e-6;

% wavelength in vacuum: 632.8 nm (HeNe laser)
% Labda=0.6328e-6;     %wavelength
Labda=0.65e-6;     %wavelength

% % screen properties
% NscreenXpoints=300;        % number of screen pixels in x
% NscreenXpoints=256;        % number of screen pixels in x
NscreenXpoints = 512;
NscreenYpoints = NscreenXpoints;

screenvectlen=NscreenXpoints*NscreenYpoints;
screenXvect(1:screenvectlen)=0;
screenYvect(1:screenvectlen)=0;
% important to remember that X is the 'slow counting MSB' and that Y is the
% 'fast counting LSB'. This to put the values in a properly oriented matrix
% later on.
% in this for loop a sample point/point light source is placed within a pixel area. At the
% center or at a random position
for x=1:NscreenXpoints
    Xtmp=(x-1)*screenXstep; % +0.5*screenXstep;
    for y=1:NscreenYpoints
        Ytmp=(y-1)*screenYstep; % +0.5*screenYstep;
        %Xtmp=(x-1)*screenXstep+0.05*screenXstep+(0.9*screenXstep)*rand();
        %Ytmp=(y-1)*screenYstep+0.05*screenYstep+(0.9*screenYstep)*rand();
        screenXvect(y+(x-1)*NscreenYpoints)=Xtmp;
        screenYvect(y+(x-1)*NscreenYpoints)=Ytmp;
    end
end


% define center of screen and an object distance Zc

Zholo = 0.35     % distance to holo points

PROJECTOR_WIDTH = NscreenXpoints*screenXstep % print
Xc0 = 0.5 * NscreenXpoints*screenXstep;
Yc0 = 0.5 * NscreenYpoints*screenYstep;
Zc=Zholo;       % scar tissue code. Zholo was Zc first...

% define some holographic points
n_objects = 1;
width = 1e-2;
% Xc = linspace(Xc0 - width/2, Xc0 + width/2, n_objects);
Xc(1) = Xc0 + Labda;
Yc(1:n_objects) = Yc0;

nHoloPoints = length(Xc)
% nHoloPoints = n_objects;
sourcepower(1:nHoloPoints)=1;    % power from holographic points.
% sourcepower = linspace(0, 1, nHoloPoints);
hphase(1:nHoloPoints)=0;

tstart=tic;   % start recording time

screenvect(1:screenvectlen)=lightfieldcalcinv(screenXvect, screenYvect, hphase, sqrt(sourcepower), Xc, Yc, Zc, Labda);
% screenvect=(max(abs(screenvect))/max(abs(screenvect)))*screenvect;

% % next section for adding a plane reference wave, comment if not needed
% maxfield=max(abs(screenvect));
% screenvect(1:screenvectlen)=screenvect(1:screenvectlen)+maxfield; % add a 'reference' plane field (phase 0 over entire projector plane) as strong as the maximum from the hologram

if add_reference
    disp('add_reference')
    % % calculate field at the projector from a point source
    Zref = 0.24 % reference wave % 0.32 0.24
    screenvect2(1:screenvectlen)=lightfieldcalc(screenXvect, screenYvect, 0, 1, Xc0, Yc0, Zref, Labda);
                                                                                                        % %% screenvect2 = abs(screenvect2) .* exp(1i * 0); % set phase to zero
                                                                                                        % % scale/normalize phase
    screenvect2=(max(abs(screenvect))/max(abs(screenvect2)))*screenvect2;
    screenvect=screenvect+screenvect2;
    Holophase=angle(screenvect2); % use phase of reference, NOT the original
else
    Holophase=angle(screenvect); % use phase of original
end

Holopixelpowers=screenXstep*screenYstep*abs(screenvect).^2; % power arriving at pixel area. Scales with area (this is 'fishy' since area is poorly defined in a random distribution) and with the square of the fieldstrength.
Holofield=sqrt(Holopixelpowers);
% Holofield = abs(screenvect);

SimTime1=toc(tstart) % stop timekeeping and show how long it took to run the whole program.
tstart=tic;   % restart recording time

% figure(10)
% histogram(Holopixelpowers,50)
% figure(11)
% histogram(Holophase,50)
% plot intensity and phase in two images.

% figure(12)
% [histarray, bincentersarray]=scatter2hist(Holophase, Holopixelpowers, 50);
% bar(bincentersarray, histarray)
% h=scatter(Holophase,Holopixelpowers,1)
disp('plot projector')
testim=matrixmaker(NscreenXpoints, NscreenYpoints, Holopixelpowers./(max(Holopixelpowers)));
testimphase=matrixmaker(NscreenXpoints, NscreenYpoints, (pi+Holophase)./(2*pi));
testim2=uint8(255*testim(:,:));
figure(1)
title('im2')
imwrite(flip(testim2, 1), '../tmp/m-y-amp2.png') % gcf = get current figure
% saveas(gcf, '../tmp/y-phase2.png') % gcf = get current figure
figure(2)
title('phase')
% imshow(testimphase)
% savefig('../tmp/m-phase2.fig')
imwrite(flip(testimphase, 1), '../tmp/m-y-phase.png') % gcf = get current figure
testim=[];
testim2=[];


if projection
    disp('projection (z)')
    % sample the light field with dense sampling so we 1- don't 'miss' the
    % tiny hologram, 2- make detailed observations of the light field.

    Zobserve = Zc; % when an offset is added you look behind or in front of the 'point hologram'/focal spot
                 % define area in which to sample
    %Zobserve = Zc + 0.2;
    Nx = 1024;
    Ny = Nx;
    % define sample density in units of wavelengths and determine the required
    % number of samples to fill the sample area
    % Samplestep=2.019*Labda;
    % Samplestep=8.019*Labda;
    projection_width = 1e-3;
    Samplestep = projection_width / (Nx - 1);
    Xleft =  Xc(1) - projection_width/2.;
    Xright = Xc(1) + projection_width/2.;
    Xrange = Xright - Xleft;
    Ybottom = Yc0 - projection_width/2.;
    Ytop =   Yc0 + projection_width/2.;
    Yrange = Ytop - Ybottom;
    % Ny=round(Yrange/Samplestep);
    % Nx=round(Xrange/Samplestep);
    screenWidth = screenXstep * NscreenXpoints
    screenHeight = screenYstep * NscreenYpoints
    projectionWidth = Xright - Xleft
    projectionHeight = Ytop - Ybottom
    sprintf('N: %i x %i = %0.3e\n', Nx, Ny, Nx * Ny)
    (projectionWidth / screenWidth)

    %put sample positions in X and Y vectors. This is dense sampling so a
    %periodic sample distribution is fine.
    holvectlen=Nx*Ny;
    holXvect(1:holvectlen)=0;
    holYvect(1:holvectlen)=0;

    for x=1:Nx
        Xtmp=(x-1)*Samplestep+0.5*Samplestep+Xleft;
        for y=1:Ny
            Ytmp=(y-1)*Samplestep+0.5*Samplestep+Ybottom;
            holXvect(y+(x-1)*Ny)=Xtmp;
            holYvect(y+(x-1)*Ny)=Ytmp;
        end
    end
    tstart=tic;
    % calculate the light field (phase and intensity distribution), resulting from all the light sources in the 'screen', in the
    % sampling plane.
    %lightfield=lightfieldcalc(ObserveXvect, ObserveYvect, Holoangles, Holopowers, Xvectscreen, Yvectscreen, Zobserve, Labda)
    sampleresult=lightfieldcalc(holXvect, holYvect, Holophase, Holofield, screenXvect, screenYvect, Zobserve, Labda);
    samplepower=abs(sampleresult).^2; % back to power densities
    samplearea=Samplestep^2;
    samplepower=samplearea*samplepower;   % power arriving in sample areas
    samplephase = angle(sampleresult);

    SimTime=toc(tstart) % stop timekeeping and show how long it took to run the whole program.

    % show images of the light intesities in the sampling area
    testim=matrixmaker(Nx, Ny, samplepower);
    testim2=uint8(255*(testim(:,:)./max(samplepower)));
    figure(5)
    imwrite(flip(testim2, 1), '../tmp/m-z-amp2.png');

    % add pi to avoid negative values
    testimphase2=matrixmaker(Nx, Ny, (pi+samplephase)./(2*pi));
    testim2=uint8(255*testim(:,:));
    title('phase')
    imwrite(flip(testimphase2, 1), '../tmp/m-z-phase.png')


    testim3=log10((1e-5*max(samplepower))*(1+round(testim./(1e-5*max(samplepower)))));
    figure(6)
    s = surf(testim,'EdgeColor','none');
    saveas(s, '../tmp/m-z-surf-amp2.png');
        figure(7);
    s = surf(testim3,'EdgeColor','none');
    saveas(s, '../tmp/m-z-surf-log-amp2.png');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Functions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function lightfield=lightfieldcalc(ObserveXvect, ObserveYvect, Holoangles, Holofield, Xvectscreen, Yvectscreen, Zobserve, Labda)
  disp('lightfieldcalc')
HoloangGPU=gpuArray(single(Holoangles));
HolofieldGPU=gpuArray(single(Holofield));
screenXGPU=gpuArray(single(Xvectscreen));
screenYGPU=gpuArray(single(Yvectscreen));

screenvectlen=uint32(length(Xvectscreen));
simvectlen=uint32(length(ObserveXvect));
Gz(1:simvectlen)=gpuArray(single(Zobserve));
Glabda(1:simvectlen)=gpuArray(single(Labda));

GholXvect=gpuArray(single(ObserveXvect));
GholYvect=gpuArray(single(ObserveYvect));
Gsamplevect(1:simvectlen)=gpuArray(single(0));
parfor k=1:screenvectlen

        Gdistances=sqrt((GholXvect-screenXGPU(k)).^2+(GholYvect-screenYGPU(k)).^2+Gz.^2);
        cosfact=Gz./Gdistances; % this function assumes that the samples have the same Z position and the inclination with the XY plane is taken into account
        Gsamplevect=Gsamplevect+cosfact.*(HolofieldGPU(k)./Gdistances).*exp(1i*((2*pi*Gdistances./Glabda)+HoloangGPU(k))); % minus for the angle. light starts to 'lag' when moving away from the source


end
lightfield=gather(Gsamplevect);
end

function lightfield=lightfieldcalcinv(ObserveXvect, ObserveYvect, Holoangles, Holofield, Xvectscreen, Yvectscreen, Zobserve, Labda)
  disp('lightfieldcalcinv')
HoloangGPU=gpuArray(single(Holoangles));
HolofieldGPU=gpuArray(single(Holofield));
screenXGPU=gpuArray(single(Xvectscreen));
screenYGPU=gpuArray(single(Yvectscreen));

screenvectlen=uint32(length(Xvectscreen));
simvectlen=uint32(length(ObserveXvect));

GholXvect=gpuArray(single(ObserveXvect));
GholYvect=gpuArray(single(ObserveYvect));
Gsamplevect(1:simvectlen)=gpuArray(single(0));
parfor k=1:screenvectlen

        % Gdistances=sqrt((GholXvect-screenXGPU(k)).^2+(GholYvect-screenYGPU(k)).^2+Gz.^2);
  Gdistances=sqrt((GholXvect-screenXGPU(k)).^2+(GholYvect-screenYGPU(k)).^2+ Zobserve^2);
        Gsamplevect=Gsamplevect + (HolofieldGPU(k)./Gdistances).*exp(1i*((-2*pi/Labda*Gdistances)+HoloangGPU(k))); % minus for the angle. light starts to 'lag' when moving away from the source


end
lightfield=gather(Gsamplevect);
end

function matout=matrixmaker(Xpoints, Ypoints, inputvect)
matout(1:Ypoints,1:Xpoints)=0;
    for x=1:Xpoints
        for y=1:Ypoints
            matout(y,x)=inputvect(y+(x-1)*Ypoints);
        end
    end
end

