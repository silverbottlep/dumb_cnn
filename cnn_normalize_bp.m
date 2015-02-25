function dzdx = my_vl_nnnormalize_bp(x, param, dzdy)
[y_h,y_w,outputmaps] = size(x);
depth = param(1);
k = param(2);
alpha = param(3);
beta = param(4);

tap = depth-(depth-1)/2;

for y_y=1:y_h
for y_x=1:y_w
for out=1:outputmaps
	s1 = 0;
	for d=1:depth
		idx = out-tap+d;
		if idx > 0 && idx <= outputmaps
			s1 = s1 + x(y_y,y_x,idx)^2;
		end
	end
	L(y_y,y_x,out) = (k + alpha*s1);
end
end
end

for y_y=1:y_h
for y_x=1:y_w
for out=1:outputmaps
	s2 = 0;
	for d=1:depth
		idx = out-tap+d;
		if idx > 0 && idx <= outputmaps
			s2 = s2 + (dzdy(y_y,y_x,idx)/(L(y_y,y_x,idx)^(beta+1))) ...
				*x(y_y,y_x,idx)*x(y_y,y_x,out);
		end
	end
	dzdx(y_y,y_x,out) = dzdy(y_y,y_x,out)/(L(y_y,y_x,out)^(beta))-2*alpha*beta*s2;
end
end
end
