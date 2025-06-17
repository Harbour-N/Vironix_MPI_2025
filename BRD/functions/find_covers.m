%{
Purpose: Given a collection of sets, find disjoint subsets.
Implementation: Encode a matrix as follows:
    a_ij = 1    if sets i and j intersect
    a_ij = 0    if sets i and j do not intersect
Now determine which rows (and corresponding columns because the matrix
will be symmetric) to remove such that what remains is an identity
matrix. Example: from [1 1 0; 0 1 1; 0 0 1] removing either the first or
second index yields [1 0; 0 1].
Inputs:
    matrix  - the matrix begin pruned
    removed - the indices removed thus far
    row     - the next row which may need to be removed
    f       - file to write to
Outputs: (This does not play a role in the code, it's just for MatLab. The
        actual data is written to an external file.)
    kept    - vector of retained indices in last recursive step
%}
function kept = find_covers(matrix,removed,row,f)

    % Find the next row with multiple 1s.
    % This is the stopping condition for recursion.
    next = row;
    while ismember(next,removed) && next < length(matrix) + 1
        next = next + 1;
    end

    % If there are still 'problem rows', proceed
    if next ~= length(matrix) + 1

        current_row = matrix(next,:);

        % Find all indices that could be removed to fix this row.
        problems = find(current_row);
        problems = problems(~ismember(problems,removed));

        % Construct matrix of options for indices to remove such that only
        % one remains.
        to_remove = zeros([length(problems),length(problems)-1]);
        for i = 1:length(problems)
            problems_temp = problems;
            problems_temp(i) = [];
            to_remove(i,:) = problems_temp;
        end

        % Remove the problem indices one by one, and recurse.
        for i = 1:size(to_remove,1)
            find_covers(matrix,[removed to_remove(i,:)],next+1,f);
        end

        % If there was nothing to be done, continue to the next row.
        if isempty(to_remove)
            find_covers(matrix,removed,next+1,f);
        end

    end

    % If there are no more 'problem rows', report the remaining indices to
    % an external file and reset the '
    if next == length(matrix) + 1
        kept = 1:length(matrix);
        kept(removed) = [];
        fprintf(f,'%g ',kept);
        fprintf(f,'\n');
    end

end