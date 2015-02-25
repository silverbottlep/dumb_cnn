net = load('/home/eunbyung/Works/data/pretrained/imagenet-caffe-ref.mat');

% evaluation for testset images
listing = dir('/home/eunbyung/Works/data/imagenet12/images/samples');
listing = listing(3:end);
n = numel(listing);
labels = [206 161 894 921 844 762 99 214 471 71];

net.layers{end} = struct('type', 'softmaxloss');
for i = 1:n
	net.layers{end}.class = labels(i);

	im = imread(['/home/eunbyung/Works/data/imagenet12/images/samples/' listing(i).name]);
	im_ = single(im) ; % note: 255 range
	im_ = imresize(im_, net.normalization.imageSize(1:2));
	im_ = im_ - net.normalization.averageImage ;

	res = cnn_bp(net, im_, single(1));
	%scores = squeeze(gather(res(end).x));
	%[bestScore, best] = max(scores);

	fpga_results{i} = res;
end

