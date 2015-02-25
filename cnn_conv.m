function y = cnn_conv(x, filters, biases, pad, stride)

% outputmap height and width calculation
y_h = ((size(x,1) - size(filters,1) + pad(1) + pad(2))/stride(1)) + 1;
y_w = ((size(x,2) - size(filters,2) + pad(3) + pad(4))/stride(2)) + 1;

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

y = zeros(y_h, y_w, outputmaps*groups);

% padding -- should be 'for loop' version later.
if pad > 0
new_x = zeros(size(x,1)+pad(1)+pad(2), size(x,2)+pad(3)+pad(4),size(x,3));
for in=1:size(x,3)
	new_x(pad(1)+1:end-pad(2),pad(3)+1:end-pad(4),in) = x(:,:,in);
end
x = new_x;
end

% convolution operation
for g=1:groups
for out=1:outputmaps
for in=1:inputmaps
	% for 2 dim outputmaps
	for y_y=1:y_h
	for y_x=1:y_w
	% for 2 dim filters
	for f_y=1:f_h
	for f_x=1:f_w
		x1 = (y_y-1)*stride(1)+f_y; 
		x2 = (y_x-1)*stride(2)+f_x;
		x3 = in+(g-1)*inputmaps;
		out_idx = out+(g-1)*outputmaps;
		y(y_y,y_x,out_idx) = y(y_y,y_x,out_idx) + ...
		x(x1,x2,x3)*filters(f_y,f_x,in,out_idx);
	end
	end
	end
	end
end
end
end

% add bias
for out=1:outputmaps*groups
for y_y=1:y_h
for y_x=1:y_w
	y(y_y,y_x,out) = y(y_y,y_x,out) + biases(out);
end
end
end

