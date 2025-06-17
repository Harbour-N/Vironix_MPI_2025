%{
Purpose: From abundance ratings and macroscopic count data, reconstruct
local counts for each bird in each neighborhood
Inputs:
    scores      - abundance ratings
    total_birds - vector of total birds observations in each neighborhood
Outputs:
    local_reconstruction    - matrix of local bird counts
%}
function local_reconstruction = scores_to_local(scores,total_birds)

    % Invert the linear interpolation.
    row_sums = sum(scores, 2);
    percentages = scores ./ row_sums;

    % Scale by total bird observations.
    local_reconstruction=percentages.*total_birds;

end