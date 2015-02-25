function res = cnn_ff(net, x)

n = numel(net.layers);
res(1).x = x;

for i=1:n
  l = net.layers{i};
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
  fprintf('ff layers: %d %s\n', i, l.type);
end
