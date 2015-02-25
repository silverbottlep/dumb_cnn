addpath('halfprecision');
net = load('/home/eunbyung/Works/data/pretrained/imagenet-caffe-ref.mat');

% evaluation for testset images
listing = dir('/home/eunbyung/Works/data/imagenet12/images/samples');
listing = listing(3:end);
n = numel(listing);


for i = 1:n
	im = imread(['/home/eunbyung/Works/data/imagenet12/images/samples/' listing(i).name]);
	im_ = single(im) ; % note: 255 range
	im_ = imresize(im_, net.normalization.imageSize(1:2));
	im_ = im_ - net.normalization.averageImage ;

	% run the CNN
	res = cnn_ff(net, im_);

	scores = squeeze(gather(res(end).x));
	[bestScore, best] = max(scores);

	%figure(1) ; clf ; imagesc(im) ;
	%title(sprintf('%s (%d), score %.3f',...
	%net.classes.description{best}, best, bestScore)) ;
	%pause;

	fpga_results{i} = res;
end
