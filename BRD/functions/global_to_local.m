%{
Purpose: From global data, compute local counts and perceived abundance
ratings.
Inputs:
    radius      - radius of neighborhood
    positions   - positions of birdwatchers
    birds       - matrix of observed birds
Outputs:
    averages        - average count over a neighborhood
    score_matrix    - perceived abundance ratings over a neighborhood
    sums            - total number of birds observed in a neighborhood
%}
function [averages score_matrix sums] = global_to_local(radius,positions,birds)

    % Collect parameters.
    [height width] = size(squeeze(birds(1,:,:)));
    num_watchers = size(positions,1);

    % Store bird counts in each neighborhood.
    averages = zeros(num_watchers, size(birds,1));
    sums = zeros(num_watchers,1);

    % Proceed bird by bird.
    for j = 1:size(birds,1)

        A = squeeze(birds(j,:,:));

        % Proceed birder by birder
        for i = 1:num_watchers

            % Get the center of each neighborhood.
            cx = positions(i, 1);  % x = column
            cy = positions(i, 2);  % y = row

            % Compute neighborhood boundaries.
            x1 = max(cx - radius, 1);
            x2 = min(cx + radius, width);
            y1 = max(cy - radius, 1);
            y2 = min(cy + radius, height);

            % Extract the submatrix.
            square_region = A(y1:y2, x1:x2);

            % Determine average over the neighborhood.
            vals = square_region(:);
            if all(isnan(vals))
                averages(i,j) = NaN;
            else
                averages(i,j) = mean(vals, 'omitnan');
                sums(i) = sums(i) + averages(i,j);
            end

        end % of birdwatcher cycle.

    end % of bird species cycle.

    % Store perceived abundance ratings.
    scaled_matrix = zeros(size(averages));

    for i = 1:size(averages, 1)
        row = averages(i, :);
        minVal = min(row);
        maxVal = max(row);

        % Use interp1 for linear interpolation between min and max count.
        if ~any(isnan(row))
            if minVal ~= maxVal
                scaled_matrix(i, :) = interp1([minVal, maxVal], [1, 5], row);
            else
                scaled_matrix(i, :) = -1;
            end            
        end

    end

    % Mimic survey by accepting only integer values.
    score_matrix=round(scaled_matrix);

end