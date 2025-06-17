%{
Purpose: Place birdwatchers within (or technically near) the US, under the
stipulation that they must be on land, and all pixels within a specified
radius are contained with the borders of the map.
Implementation: Available bird maps encode ocean tiles with NaN and empty
land tiles with 0. The first condition may therefore be enforced by
ensuring no birders are placed on a NaN tile. The second condition is
enforced by precising birder positions to be beyond a fixed distance from
the border of the map. Since outstanding birders are removed, the initial
number of positions is artificially raised by 30, allowing for that many
removals.
Inputs:
    num_watchers    - desired number of birdwatchers
    radius          - radius defining birder neighborhood
    birds           - map of bird distribution
Output:
    birders     - matrix of (x,y) coordinates for birders
%}
function birders = collect_birders(num_watchers,radius,birds)

    % Define borders of region.
    [height, width] = size(birds);

    % Randomly populate the map with birders.
    rows_w = randi([radius,height-radius], num_watchers + 30, 1);
    cols_w = randi([radius,width-radius], num_watchers + 30, 1);
    
    % Check for NaNs at chosen positions.
    isValid = ~isnan(birds(sub2ind(size(birds), rows_w, cols_w)));

    % Prune chosen positions of all invalid positions.
    rows_w = rows_w(isValid);
    cols_w = cols_w(isValid);

    birders = [cols_w(1:num_watchers), rows_w(1:num_watchers)];  % x = col, y = row

end