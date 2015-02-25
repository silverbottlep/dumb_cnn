function dzdx = my_vl_nnpool_bp(x, pool, dzdy, pad, stride)
dzdy_h = size(dzdy,1);
dzdy_w = size(dzdy,2);

[dzdx_h dzdx_w outputmaps] = size(x);
dzdx = single(zeros(dzdx_h, dzdx_w, outputmaps));

for out=1:outputmaps
	for dzdy_y=1:dzdy_h
	for dzdy_x=1:dzdy_w
		y1 = max((dzdy_y-1)*stride(1)+1-pad(1), 0);
		x1 = max((dzdy_x-1)*stride(2)+1-pad(3), 0);
		y2 = min(y1+pool(1)-1,dzdx_h);
		x2 = min(x1+pool(2)-1,dzdx_w);

		maximum = -eps;
		for dzdx_y=y1:y2
		for dzdx_x=x1:x2
			if(x(dzdx_y,dzdx_x,out) > maximum)
				maximum = x(dzdx_y,dzdx_x,out);
				idx_y = dzdx_y;
				idx_x = dzdx_x;
			end
		end
		end

		dzdx(idx_y,idx_x,out) = dzdx(idx_y,idx_x,out) + dzdy(dzdy_y,dzdy_x,out);
	end
	end
end
