%function g = build_g1(x,a,m,s)
%    pow = -(x-m.').^2./(2*s.^2).';
%    gp = exp(pow).*a.';
%    g = sum(gp./(s*sum(a)*sqrt(2*pi)).');
%end

function g = build_g2(x,y,a,m,s)
    (x-m).^2;
    sum((x-m).^2);
    pow = -sum((x-m).^2)/(2*s^2)
    gp = exp(pow)*a
    g = sum(gp/(s*sum(a)*2*pi))
    %pow = -(x-m.').^2./(2*s.^2).'
    %gp = exp(pow).*a.';
    %g = sum(gp./(s*sum(a)*2*pi).');
end

disp('begin')

%a = [10 5 6 6];
%m = [2 2 4 4; 2 4 2 4];
%s = [1 1 1 1];

a = [10];
m = [1.5; 2];
s = [1];

dist = @(t) build_g2(t,a,m,s);

%X = linspace(0,10);
%Y = linspace(0,10);

X = [1 2];
Y = [1 2];
[X, Y] = meshgrid(X,Y)

clf
figure(1)
hold on

surf(X,Y,dist([X; Y]))

%{
a = [10 5 6 6];
m = [2 5 7 8];
s = [1 1 1 1];

dist = @(t) build_g1(t,a,m,s);

X = linspace(0,10);

clf
figure(1)
hold on

plot(X,dist(X))
%}
disp('end')