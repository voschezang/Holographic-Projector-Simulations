clear all;
disp('plot projection (using projector)')

TWO_PI = 2 * pi
N = 512 ^ 2;
z_amp = read_array('../tmp/z0_0_amp.dat', N);
z_phase = read_array('../tmp/z0_0_phase.dat', N);
z_pos = read_array('../tmp/w0_0.dat', N * 3);

n_sqrt = sqrt(length(z_amp))
% assume z is a square matrix
gray = reshape(uint8(255 .* z_phase ./ TWO_PI), n_sqrt, n_sqrt);
figure(1)
imwrite(gray, '../tmp/z_phase.png');
A = reshape(z_amp ./ sum(z_amp), n_sqrt, n_sqrt);
figure(2)
imwrite(uint8(255 .* A), '../tmp/z_amp.png');
figure(3)
log_A = log(A);
imwrite(uint8(255 .* log_A ./ max(abs(log_A))), '../tmp/z_amp_log.png');
figure(4)
s = surf(A, 'EdgeColor', 'none');
saveas(s, '../tmp/z_amp_surf.png');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function result = flatten_positions(x, y, z)
% concatenate, then flatten
result = reshape(cat(1, x, y, z), 1, []);
end

function result = unflatten_positions(array)
% concatenate, then flatten
result = reshape(array, 3, []);
end

function array = read_array(filename, length)
file_id = fopen(filename, 'r');
assert(file_id > 0, 'Error; cannot open file');
array = fread(file_id, length, 'double'); 
fclose(file_id);
end

function out = save_array(array, filename)
file_id = fopen(filename, 'w'); 
assert(file_id > 0, 'Error; cannot open file');
fwrite(file_id, array, 'double'); 
fclose(file_id);
end


