% === Configuration ===
input_folder = 'recon';        % Folder containing input .png images
output_folder = 'matlab';    % Folder to save output segmentations
ext = '*.tif';                        % Change if using .jpg, .tif, etc.
% Create output folder if it doesn't exist
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% Get list of files
image_files = dir(fullfile(input_folder, ext));

% === Loop over each image ===
for i = 1:length(image_files)
    file_name = image_files(i).name;
    input_path = fullfile(input_folder, file_name);
    
    % Read and normalize image
    img = im2double(imread(input_path));
    
    % Segment
    seg = pseudoSegment(img);

    % Convert segmentation to color
    seg_rgb = label2rgb(seg);

    % Save result
    [~, base_name, ~] = fileparts(file_name);
    output_path = fullfile(output_folder, [base_name, '_alter.tif']);
    imwrite(seg_rgb, output_path);

    fprintf('Saved: %s\n', output_path);
end

fprintf('Done processing %d files.\n', length(image_files));

