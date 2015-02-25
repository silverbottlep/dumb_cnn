function y = cnn_normalize(x, param)
[y_h,y_w,outputmaps] = size(x);
depth = param(1);
k = param(2);
alpha = param(3);
beta = param(4);

tap = depth-(depth-1)/2;

for y_y=1:y_h
for y_x=1:y_w
for out=1:outputmaps
	s = 0;
	for d=1:depth
		idx = out-tap+d;
		if idx > 0 && idx <= outputmaps
			s = s + x(y_y,y_x,idx)^2;
		end
	end
	y(y_y,y_x,out) = x(y_y,y_x,out)/((k + alpha*s)^beta);
end
end
end
