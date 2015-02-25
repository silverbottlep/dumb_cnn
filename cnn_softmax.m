% preventing from overflow and underflow substract maximum value from
% nominator and denominator.
function y = cnn_softmax(x)
max_value = max(x);
denom = 0;
for i=1:size(x,3)
	x(1,1,i) = x(1,1,i) - max_value;	
	denom = denom + exp(x(1,1,i));
end
for i=1:size(x,3)
	y(1,1,i) = exp(x(1,1,i))/denom;
end
