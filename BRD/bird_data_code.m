%% To Begin: Load in data.

clc
load('bird_data/birds.mat');
addpath('functions');

%% To Begin: Prepare data.
% This step has already been completely and the results compiled in the
% bird_matrix.  An example is included here for the sake of a complete
% report.

clc

data_for_bird_matrix = process_tif( ...
    'bird_data/norcar/norcar_abundance_seasonal_year_round_mean_2023.tif');

% This cheeky use of another plotting function is to save on code, hence
% the seemingly vacuous variable.
plot_birders(1,data_for_bird_matrix,{[]},0,[]);

%% For Goal 1: Construct abundance ratings from local distributions

clc

nw = 3;
r = 100;

pos = collect_birders(nw,r,squeeze(bird_matrix(1,:,:)));

[cm sm ss] = global_to_local(r,pos,bird_matrix);

cm(1:3,:,:)
sm(1:3,:,:)
ss.'

%% For Goal 2: Place birders and visualize neighborhoods.

clc

nw = 10;    % number of birders
r = 100;    % neighborhood radius

% Place birders.
pos = collect_birders(nw,r,squeeze(bird_matrix(1,:,:)));

% Visualize birders.
plot_birders(1,squeeze(bird_matrix(1,:,:)),{1:nw},r,pos);

%% For Goal 2: Find collection of birders with disjoint neighborhoods.

clc

% Calculate neighborhoods.
nbhd = collect_neighborhoods( ...
            r,size(bird_matrix,2),size(bird_matrix,3),pos);
ins = build_intersections(nbhd);

% Calculate disjoint subcollections of birders.
file = fopen('raw_covers.txt','w');
find_covers(ins,[],1,file);
fclose(file);
covers = collect_covers('raw_covers.txt');

% Find largest subcollections (for easier visualization) and plot.
ind = find(cellfun(@length,covers)==max(cellfun(@length,covers)));
plot_birders(2,squeeze(bird_matrix(1,:,:)),{covers{ind(1)}},r,pos);

%% For Goal 3: Construct local counts from abundance ratings.
% One can run the Goal 1 section first for some data to pass to this
% section.

clc

l = scores_to_local(sm,ss);

cm(1:3,:,:)
l(1:3,:,:)