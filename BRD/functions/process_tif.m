%{
Purpose: Load in raw raster data from eBird in tif format.  Adjust the
window to only include US (eBird collects observations worldwide).
Note: This has already been run for all our data.  Results are compiled in
the bird_matrix.
Inputs:
    f   - the tif file to read in
Outputs:
    zoomed_in   - A matrix representing bird distribution in the US
%}
function zoomed_in = process_tif(f)

    % Read in raw data.
    img = imread(f);

    % Adjust object type.
    imgMatrix = double(img);

    % Zoom in on US (determining coordinates from a good ol' eyeballin')
    zoomed_in = imgMatrix(850:1700, 2000:3800);
end