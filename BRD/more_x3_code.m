function w = which_to_remove(v)
    if sum(v) > 1
        x = find(v);
        w = zeros([length(x),length(x)-1]);
        for i = 1:length(x)
            r = x;
            r(i) = [];
            w(i,:) = r;
        end
    else
        w = [];
    end
end

function f = step_thru1(a, r, x)
    if x <= length(a)
        a1 = a(x,:);
        r;
        a2 = a1(~ismember(a1==1,r));

        
        w = which_to_remove(a2);


        if ~isempty(w)
            for i = 1:size(w,1)
                wr = w(i,:);
                %wr = wr(~ismember(wr,r))
                if ~isempty(wr)
                    step_thru1(a,[r wr],x+1);
                else
                    step_thru1(a,r,x+1);
                end
            end
        else
            step_thru1(a,r,x+1);
        end
    else
        f = 1:length(a);
        f(r) = [];
        f
    end
end

clc

disp('begin')

m = [1 1 0;
     1 1 1;
     0 1 1
     ];

step_thru1(m,[],1);
disp('end')


function r = step_thru(a, r, x)
    if x <= length(a)
        a(x,:)
        w = which_to_remove(a(x,:))
        if ~isempty(w)
            for i = 1:size(w,1)
                wr = w(i,:);
                wr = wr(~ismember(wr,r))
                if ~isempty(wr)
                    step_thru(a,[r wr],x+1);
                else
                    step_thru(a,r,x+1);
                end
            end
        else
            step_thru(a,r,x+1);
        end
    else
        f = 1:length(a);
        f(r) = [];
        f
    end
end