clear all;
disp('init params')

distance_point = 0.5;
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
screenZvect(1:screenvectlen)=0;
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


% define screen
PROJECTOR_WIDTH = NscreenXpoints*screenXstep % print
Xc0 = 0.5 * NscreenXpoints*screenXstep;
Yc0 = 0.5 * NscreenYpoints*screenYstep;

% define some holographic points
n_objects = 1;
width = 1e-2;
% Xc = linspace(Xc0 - width/2, Xc0 + width/2, n_objects);
Xc(1) = Xc0 + Labda;
Yc(1:n_objects) = Yc0;

nHoloPoints = length(Xc)

% obj x, projector, y
x_amp(1:nHoloPoints) = 1.;    % power from holographic points.
x_phase(1:nHoloPoints) = 0.;
x_pos = flatten_positions([0.], [0.], [distance_point]);
y_pos = flatten_positions(screenXvect, screenYvect, screenZvect);

save_array(x_amp, '../tmp/x0_amp.dat');
save_array(x_phase, '../tmp/x0_phase.dat');
save_array(x_pos, '../tmp/u0.dat');
save_array(y_pos, '../tmp/v0.dat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% define projection

projectionZvect(1:screenvectlen) = distance_point;
z_pos = flatten_positions(screenXvect, screenYvect, projectionZvect);
save_array(z_pos, '../tmp/w0.dat');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = flatten_positions(x, y, z)
% concatenate, then flatten
result = reshape(cat(1, x, y, z), 1, []);
end

function out = save_array(array, filename)
file_id = fopen(filename, 'w'); 
fwrite(file_id, array, 'double'); 
fclose(file_id);
end

