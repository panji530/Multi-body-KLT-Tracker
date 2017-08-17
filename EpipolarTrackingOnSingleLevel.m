%Author: Pan Ji, pan.ji@anu.edu.au, Australian National University
%All rights reserved
%Cite: Pan Ji, et al. Robust multi-body feature tracker: a
%      segmentation-free approach, in CVPR 2016
function [newLocations, u, C] = EpipolarTrackingOnSingleLevel(preImage, preLoc, currImage, u0, stackNum, Params)
% preImage: the template image
% preLoc: the feature locations on the template image
% currImage: the current image
% u0: initial displacments
% stackNum: pyramid stack/level number
% Params: parameters

ratio     = 2^(stackNum-1);
patchsize = Params.patchsize;
gamma     = Params.gamma;
lambda    = Params.lambda;

maxOuterIter = Params.maxOuterIter;
thr          = Params.thr;

preLocThisLevel = preLoc/ratio;
u0ThisLevel     = u0/ratio;

N = size(preLocThisLevel, 2);
D = patchsize*patchsize;
d = floor(patchsize/2);
[height, width] = size(preImage);

%% normalized data
x_bar_normd  =[preLocThisLevel/width ; ones(1,N)]; % third row all 1

%% compute image gradient
[currImage_Gx, currImage_Gy] = imgradientxy(currImage); % sobel by default Sobel', Prewitt', 'CentralDifference', or 'IntermediateDifference'

%% get all the patch index
all_idx_x    = zeros(patchsize, patchsize*N);
all_idx_y    = zeros(patchsize, patchsize*N);

for i = 1:N
	patchidx_x           = (preLocThisLevel(1,i)-d):(preLocThisLevel(1,i)+d);
	patchidx_y           = (preLocThisLevel(2,i)-d):(preLocThisLevel(2,i)+d);
	patchidx_x           = ((patchidx_x>=1) & (patchidx_x<=width)).*patchidx_x + (patchidx_x<1) + (patchidx_x>width);
	patchidx_y           = ((patchidx_y>=1) & (patchidx_y<=height)).*patchidx_y + (patchidx_y<1) + (patchidx_y>height);
    [idx_x, idx_y]       = meshgrid(patchidx_x, patchidx_y);
	all_idx_x(:, ((i-1)*patchsize+1):(i*patchsize))   = idx_x;
	all_idx_y(:, ((i-1)*patchsize+1):(i*patchsize))   = idx_y;
end
Ti_ALL    = fast_interp2(1:width, 1:height, preImage, all_idx_x, all_idx_y,'linear'); %mex the C++ function with your Matlab 
u         = u0ThisLevel;

for outerIter = 1:maxOuterIter
	tic
    u0 = u;
	rep_u01 = reshape(repmat(u(1,:), D, 1),patchsize, patchsize*N);
	rep_u02 = reshape(repmat(u(2,:), D, 1),patchsize, patchsize*N);
	all_u0idx_x = all_idx_x+rep_u01;
	all_u0idx_y = all_idx_y+rep_u02;
	all_u0idx_x = ((all_u0idx_x>=1) & (all_u0idx_x<=width)).*all_u0idx_x + (all_u0idx_x<1) + (all_u0idx_x>width);
	all_u0idx_y = ((all_u0idx_y>=1) & (all_u0idx_y<=height)).*all_u0idx_y + (all_u0idx_y<1) + (all_u0idx_y>height);
	
	%% get all template, image, and gradient
	Ii_ALL    = fast_interp2(1:width, 1:height, currImage, all_u0idx_x, all_u0idx_y,'linear'); %1:width, 1:height,
	IGxi_ALL  = fast_interp2(1:width, 1:height, currImage_Gx, all_u0idx_x, all_u0idx_y,'linear'); %1:width, 1:height,
	IGyi_ALL  = fast_interp2(1:width, 1:height, currImage_Gy, all_u0idx_x, all_u0idx_y,'linear'); %1:width, 1:height,  linear
	
	tau_i   = IGxi_ALL.*rep_u01 + IGyi_ALL.*rep_u02 + Ti_ALL - Ii_ALL;
	
	A   = reshape(Ii_ALL-Ti_ALL, D, N);
	A   = A';
	IGx = reshape(IGxi_ALL, D, N);
	IGx = IGx';
	IGy = reshape(IGyi_ALL, D, N);
	IGy = IGy';
	tau = reshape(tau_i, D, N);
	tau = tau';
	
	%% get big sparse matrix H
	IGxx = sum(IGx.^2,2);
	IGyy = sum(IGy.^2,2);
	IGxy = sum(IGx.*IGy,2);
	IG_all = [IGxx';IGxy';IGxy';IGyy'];
	
	I = repmat(reshape(1:(2*N), 2, N),2,1);
	J = repmat(1:(2*N), 2, 1);
	H = sparse(I(:) ,J(:), IG_all(:));
	
	%% normalized data	
	u0_bar_normd = [u/width; zeros(1,N)]; % third row all 0
	
	%% init W
	tmp1 = [repmat(x_bar_normd(1,:),3,1);repmat(x_bar_normd(2,:),3,1);repmat(x_bar_normd(3,:),3,1)];
	tmp2 = repmat(x_bar_normd+u0_bar_normd,3,1);
	W = tmp1.*tmp2;
	
	%% get big sparse matrix P
	I = 1:(9*N);
	J = repmat(reshape(1:(3*N),3,N),3,1);
	P_bar        = sparse(I', J(:), tmp1(:));
	P            = P_bar;
	P(:,3*(1:N)) = [];
	PtP = P'*P;
	b            = P_bar*x_bar_normd(:);%x_bar(:);%
	m            = P*u(:)/width;%u0ThisLevel(:);%
    
	%% parameters for ADMM, and initializations
	rho  = 1e-6; %1/computeLambda_mat(A);%
	eta  = 8; epsilon = 1e-6; max_rho = 1e10; maxIter = 1e8;
	E    = zeros(9,N); %outlier term
	Y1   = zeros(9,N);
	Y2   = zeros(N,D); %Lagrange multipliers
	y    = zeros(9*N,1);
	iter = 0;
	
	%% start main algorithm
	while(iter < maxIter)
		iter = iter+1;
		%update Z.
		tmp = A-Y2/rho;
		Z   = max(0,tmp-gamma/rho)+min(0,tmp+gamma/rho);
        
		%update C		
		lhs = eye(N)+rho*(W'*W);
		rhs = rho*W'*(W-E+Y1/rho);       
		C   = lhs\rhs;       
        		
		%update E		
		tmp = W-W*C+Y1/rho;
		E   = max(0,tmp-lambda/rho)+min(0,tmp+lambda/rho);
        
		%update u		
		gx  = sum((Y2 + rho*(tau+Z)).*IGx,2);
		gy  = sum((Y2 + rho*(tau+Z)).*IGy,2);		
        g   = [gx,gy]';
        g   = g(:);
		
		lhs = rho*(H+PtP/(width^2));
		rhs = g+P'*y/width+rho*P'*m/width;		
		u   = lhs\rhs;	
		
		%update m 		
        B = reshape(b, 9, N);
		G = reshape(y/rho-P*u/width, 9, N);
		
        ImC = eye(N)-C;
        tmp = ImC*ImC';
		lhs = tmp+eye(N);
		rhs = -(G+B*tmp+(Y1/rho-E)*(eye(N)-C'));		
		M = rhs/lhs;	
		
		
		m = M(:);
		%update W, A with new u
		tmp = repmat(u,1,D);
		ux  = tmp(1:2:end,:);
		uy  = tmp(2:2:end,:);
		A   = ux.*IGx + uy.*IGy - tau;
		
		W = reshape(b+m, 9, N);
		
		leq1 = W-W*C- E;
		leq2 = Z - A;
		leq3 = m - P*u/width;
		convResid = max( max(max(leq1(:)),max(leq2(:))), max(leq3(:)) );

		if(convResid<epsilon)
			break;
		else
			Y1 = Y1 + rho*leq1;
			Y2 = Y2 + rho*leq2;
			y = y + rho*leq3;
			rho = min(max_rho,eta*rho);
		end
	end
	
	u = reshape(u, 2, N);
    delta_u = u-u0;
	
	if(norm(delta_u)<thr)
		break;
	end
end

%% update u and location
u = u*ratio;
newLocations = preLoc + u;

end
