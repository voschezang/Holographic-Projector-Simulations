clear all;
disp('plot projector')

TWO_PI = 2 * pi

% obj x of N datapoints, projector y of M datapoints
N = 1;
M = 512 ^ 2;
x_amp = read_array('../tmp/x0_amp.dat', N);
x_phase = read_array('../tmp/x0_phase.dat', N);
x_pos = read_array('../tmp/u0.dat', N * 3);
y_amp = read_array('../tmp/y0_amp.dat', M);
y_phase = read_array('../tmp/y0_phase.dat', M);
y_pos = read_array('../tmp/v0.dat', M * 3);

%y_amp(:, :) = y_amp / max(y_amp)


n_sqrt = sqrt(length(y_amp))
% assume y is a square matrix
gray = reshape(uint8(255 .* y_amp ./ max(y_amp)), n_sqrt, n_sqrt);
figure(1)
imwrite(gray, '../tmp/y_amp.png');
gray = reshape(uint8(255 .* y_phase ./ TWO_PI), n_sqrt, n_sqrt);
figure(2)
imwrite(gray, '../tmp/y_phase.png');
%figure(3)
%s = surf(gray, 'EdgeColor', 'none');
%saveas(s, '../tmp/y_amp_surf.png');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = unflatten_positions(array)
% concatenate, then flatten
result = reshape(array, 3, []);
end

function array = read_array(filename, length)
file_id = fopen(filename, 'r'); 
array = fread(file_id, length, 'double'); 
fclose(file_id);
end

