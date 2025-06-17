%{
Purpose: Given a collection of birders, collect the regions (neighborhoods)
which they observe.
Inputs:
    radius      - radius of birder neighborhood
    height      - one dimension of map
    width       - other dimension of map
    positions   - positions of birders
Outputs:
    neighborhoods   - collection of birder neighborhoods, stored (x,y,z)
    (birder number, x coordinate, y coordinate)
%}
function neighborhoods = collect_neighborhoods(radius,height,width,positions)

    neighborhoods=zeros(size(positions,1), height, width);

    for i = 1:size(positions,1)

        % Get the center of the square.
        cx = positions(i, 1);
        cy = positions(i, 2);

        % Compute square boundaries, ensuring they stay within matrix
        % bounds. I suspect this could be removed or streamlined since we
        % modified birder position generated (see collect_birders).
        x1 = max(cx - radius, 1);
        x2 = min(cx + radius, width);
        y1 = max(cy - radius, 1);
        y2 = min(cy + radius, height);

        % Calculate birder neighborhoods in binary (0/1) format. Compile
        % in booklet.
        outside_matrix=zeros(height,width);
        outside_matrix(y1:y2,x1:x2)=1;
        neighborhoods(i,:,:)=outside_matrix;

    end

end