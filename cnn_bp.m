function res = cnn_bp(net, x, dzdy)

n = numel(net.layers);
res(1).x = x;
res(n+1).dzdx = dzdy ;

for i=1:n
  l = net.layers{i};
  fprintf('ff layers: %d %s\n', i, l.type);
  switch l.type
    case 'conv'
      res(i+1).x = cnn_conv(res(i).x, l.filters, l.biases, l.pad, l.stride);
    case 'pool'
      res(i+1).x = cnn_pool(res(i).x, l.pool, l.pad, l.stride);
    case 'normalize'
      res(i+1).x = cnn_normalize(res(i).x, l.param);
    case 'softmax'
      res(i+1).x = cnn_softmax(res(i).x);
    case 'softmaxloss'
      res(i+1).x = cnn_softmaxloss(res(i).x, l.class);
    case 'relu'
      res(i+1).x = cnn_relu(res(i).x);
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
end

for i=n:-1:1
  l = net.layers{i} ;
  fprintf('bp layers: %d %s\n', i, l.type);
  switch l.type
    case 'conv'
		  [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
				cnn_conv_bp(res(i).x, l.filters, l.biases, ...   
				res(i+1).dzdx, l.pad, l.stride) ;  
    case 'pool'
       res(i).dzdx = cnn_pool_bp(res(i).x, l.pool, res(i+1).dzdx, l.pad, l.stride);
    case 'normalize'
       res(i).dzdx = cnn_normalize_bp(res(i).x, l.param, res(i+1).dzdx);
    case 'softmaxloss'
       res(i).dzdx = cnn_softmaxloss_bp(res(i).x, l.class, res(i+1).dzdx);
    case 'relu'
       res(i).dzdx = res(i+1).dzdx .* (res(i).x > single(0));
    otherwise
      error('Unknown layer type %s', l.type) ;
    end
  end
end
