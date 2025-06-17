%{
Purpose: From a collection of sets, prune the redundant entries.
Inputs:
    covers_file - file containing collection of sets
Outputs:
    covers  - file containing collection of sets with duplicates removed
%}
function covers = collect_covers(covers_file)

    % Read in file.
    covers_step1 = readlines(covers_file);

    % Convert strings to arrays for processing. I suspect this step could
    % be removed or streamlined.
    covers_step2= {};
    for i = 1:size(covers_step1,1)
        covers_step2 = [covers_step2 str2num(covers_step1(i))];
    end

    % Remove duplicates.
    covers_step3 = cellfun(@mat2str,covers_step2,'UniformOutput',false);
    [~, ia, ~] = unique(covers_step3,'stable');
    covers = cellfun(@str2num,covers_step3(ia),'UniformOutput',false);

end