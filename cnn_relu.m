function y = cnn_relu(x)
[h,w,k] = size(x);

for x_y=1:h
for x_x=1:w
for x_k=1:k
	if x(x_y,x_x,x_k) <= 0
		x(x_y,x_x,x_k) = 0;
	end
end
end
end
y = x;
