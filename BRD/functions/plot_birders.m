%{
Purpose: This is a helpful graphing method. It plots birder neighborhoods
over some background image.
Inputs:
    fig_num - which figure to plot in
    bottom  - the background image
    covers  - which birders to plot AS CELL ARRAY
    radius  - the radius of birder neighborhoods
    birders - positions of birders
Outputs:
    plot    - a dummy variable
%}
function plot = plot_birders(fig_num,bottom,covers,radius,birders)

    figure(fig_num)

    % Plot background image.
    imagesc(bottom);
    colormap gray;
    colorbar;
    daspect([1,1,1])
    hold on

    % Plot neighborhoods.
    for i = covers{1}
        x = birders(i,1) - radius;
        y = birders(i,2) - radius;
        rectangle('Position', [x, y, 2*radius, 2*radius], ...
                  'EdgeColor', 'w', 'LineWidth', 0.6);
    end

end