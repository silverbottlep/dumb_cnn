function y = cnn_conv_16bit(x, filters, biases, pad, stride)

y_h = ((size(x,1) - size(filters,1) + pad(1) + pad(2))/stride(1)) + 1;
y_w = ((size(x,2) - size(filters,2) + pad(3) + pad(4))/stride(2)) + 1;
f_h = size(filters,1);
f_w = size(filters,2);
groups = size(x,3)/size(filters,3);
inputmaps = size(filters,3);
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

% convert inputs into 16bit ieee floating point precision
temp = halfprecision(x);
x = halfprecision(temp,'single');
temp = halfprecision(filters);
filters = halfprecision(temp,'single');
temp = halfprecision(biases);
biases = halfprecision(temp,'single');

% convolution operation
for g=1:groups
for out=1:outputmaps
for in=1:inputmaps
	for y_y=1:y_h
	for y_x=1:y_w
	for f_y=1:f_h
	for f_x=1:f_w
		y(y_y,y_x,out+(g-1)*outputmaps) = y(y_y,y_x,out+(g-1)*outputmaps) + ...
		x( (y_y-1)*stride(1)+f_y, (y_x-1)*stride(2)+f_x, in+(g-1)*inputmaps ) * ...
		filters(f_y,f_x,in,out+(g-1)*outputmaps);
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

% convert outputs into 16bit ieee floating point precision
temp = halfprecision(y);
y = halfprecision(temp,'single');
