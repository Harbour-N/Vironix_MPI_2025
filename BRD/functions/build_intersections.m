%{
Purpose: Given a collectoin of binary matrices, construct a binary matrix
representing intersections by a_ij = 1 iff matrices i and j have nontrivial
intersection.
Inputs:
    birders - collection of binary matrices to intersect
Outputs:
    intersections   - single binary matrix representing intersections
%}
function intersections = build_intersections(birders)
   
    % Set for loop bounds.
    num_watchers = size(birders,1);

    % Check all birders pairs for intersection.
    intersections = zeros(num_watchers,num_watchers);
    for i = 1:num_watchers
        for j = 1:num_watchers
            if sum(birders(i,:,:).*birders(j,:,:),'all') > 0
                intersections(i,j) = 1;
            end
        end
    end

end