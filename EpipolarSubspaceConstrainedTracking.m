%Author: Pan Ji, pan.ji@anu.edu.au, Australian National University
%All rights reserved
%Cite: Pan Ji, et al. Robust multi-body feature tracker: a
%      segmentation-free approach, in CVPR 2016
function [currLocations, displacement, C] = EpipolarSubspaceConstrainedTracking(preLocations, preFrame, currFrame, Params)
prePyramids = {};
currPyramids = {};

stackNum = Params.stackNum;

%Build image pyramids
prePyramids{1} = preFrame;
for i = 2:stackNum
	prePyramids{i} = impyramid(prePyramids{i-1},'reduce');
end

currPyramids{1} = currFrame;
for i = 2:stackNum
	currPyramids{i} = impyramid(currPyramids{i-1},'reduce');
end

displaceOfLastStack = Params.init_u0; %initial displacement vector at the coarsest level
for i = stackNum:-1:1	
	[currLocations, displaceOfLastStack, C] = EpipolarTrackingOnSingleLevel(prePyramids{i}, preLocations, currPyramids{i}, displaceOfLastStack, i, Params);
end
displacement = displaceOfLastStack;

end