function [dzdx, dzdw, dzdb] = my_vl_nnconv_bp(x, filters, biases, dzdy, pad, stride)

% fully connected mode
fc = 0;
if size(filters,1)==size(x,1) && size(filters,2)==size(x,2) && size(filters,3)==size(x,3)
	fc = 1;
end
% outputmap height and width calculation
%y_h = ((size(x,1) - size(filters,1) + pad(1) + pad(2))/stride(1)) + 1;
%y_w = ((size(x,2) - size(filters,2) + pad(3) + pad(4))/stride(2)) + 1;
y_h = size(dzdy,1);
y_w = size(dzdy,2);
% filters height and width
f_h = size(filters,1);
f_w = size(filters,2);

% this is for the multiple machine execution, 
% no information exchange between these groups
groups = size(x,3)/size(filters,3);
% the number of inputmaps
inputmaps = size(filters,3);
% the number of outputmaps for each group
outputmaps = size(filters,4)/groups;

% padding -- should be more explicit for loop version later.
padded_x = zeros(size(x,1)+pad(1)+pad(2), size(x,2)+pad(3)+pad(4),size(x,3));
for in=1:size(x,3)
	padded_x(pad(1)+1:end-pad(2),pad(3)+1:end-pad(4),in) = x(:,:,in);
end

dzdw = zeros(size(filters));
dzdb = zeros(size(biases));

% compute dzdw
for g=1:groups
for out=1:outputmaps
for in=1:inputmaps
	in_idx = in+(g-1)*inputmaps;
	out_idx = out+(g-1)*outputmaps;
	% for 2 dim outputmaps
	for y_y=1:y_h
	for y_x=1:y_w
		for f_y=1:f_h
		for f_x=1:f_w
			x1 = (y_y-1)*stride(1)+f_y; 
			x2 = (y_x-1)*stride(2)+f_x;
			dzdw(f_y,f_x,in,out_idx) = dzdw(f_y,f_x,in,out_idx) + ...
			dzdy(y_y,y_x,out_idx)*padded_x(x1,x2,in_idx);
		end
		end
	end
	end
end
end
end

% compute dzdb
for y_y=1:y_h
for y_x=1:y_w
for out=1:outputmaps*groups
	dzdb(out) = dzdb(out) + dzdy(y_y,y_x,out);
end
end
end

sx = size(x);
fx = size(filters);
if fc
	x = zeros(1,1,sx(1)*sx(2)*sx(3));
	filters = reshape(filters,1,1,fx(1)*fx(2)*fx(3),fx(4));
	inputmaps = size(x,3);
end

x_h = size(x,1);
x_w = size(x,2);
f_h = size(filters,1);
f_w = size(filters,2);
dzdx = zeros(size(x));

% padding for dzdy to compute dzdx 
% padding -- should be translated into for loop version later.
padded_dzdy = zeros(x_h+f_h-1, x_w+f_w-1, outputmaps*groups);
t = (size(padded_dzdy,1)-size(x,1))/2;
if stride(1) > 1
	for out=1:outputmaps*groups
	for y_y=1:y_h
	for y_x=1:y_w
		padded_dzdy(1+t+(y_y)*stride(1)+1,1+t+(y_x)*stride(2)+1,out) = dzdy(y_y,y_x,out);
	end
	end
	end
else
	for out=1:outputmaps*groups
		padded_dzdy(1+t:end-t,1+t:end-t,out) = dzdy(:,:,out);
	end
end

% compute dzdx
for g=1:groups
for in=1:inputmaps
for out=1:outputmaps
	in_idx = in+(g-1)*inputmaps;
	out_idx = out+(g-1)*outputmaps;
	for x_y=1:x_h
	for x_x=1:x_w
	for f_y=1:f_h
	for f_x=1:f_w
		y1 = x_y-1+f_y; 
		y2 = x_x-1+f_x;
		dzdx(x_y,x_x,in_idx) = dzdx(x_y,x_x,in_idx) + ...
		padded_dzdy(y1,y2,out_idx)*filters(f_h-f_y+1,f_w-f_x+1,in,out_idx);
	end
	end
	end
	end
end
end
end

% reshape -- should be translated into for loop version later.
if fc
	dzdx = reshape(dzdx,sx(1),sx(2),sx(3));
end
