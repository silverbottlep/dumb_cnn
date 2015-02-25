%% maxpooling
function y = cnn_pool(x, pool, pad, stride)
y_h = ((size(x,1) - pool(1) + pad(1) + pad(2))/stride(1)) + 1;
y_w = ((size(x,2) - pool(2) + pad(3) + pad(4))/stride(2)) + 1;
outputmaps = size(x,3);
y = zeros(y_h, y_w, outputmaps);

for out=1:outputmaps
	for y_y=1:y_h
	for y_x=1:y_w
		max_value = -eps;
		for p_y=1:pool(1)
		for p_x=1:pool(2)
			if( max_value < x((y_y-1)*stride(1)+p_y, (y_x-1)*stride(2)+p_x, out) )
				max_value = x((y_y-1)*stride(1)+p_y, (y_x-1)*stride(2)+p_x, out);
			end
		end
		end
		y(y_y,y_x,out) = max_value;
	end
	end
end
