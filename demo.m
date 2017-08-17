%Author: Pan Ji, pan.ji@anu.edu.au, Australian National University
%All rights reserved
clear, close all
addpath(genpath(pwd))
cd 'Data'

Params.patchsize = 7; %Patch size
Params.gamma = 1.8e4; 
Params.lambda = 1.0e4;
Params.stackNum = 4; %The number of image pyramid levels (try larger values for high-sdefinition images)
Params.maxOuterIter = 10; %Maximal outer iteration number on a single level (you may increase this value for better tracking accuracy)
Params.thr = 1; %Convergence threshold (you may decrease this value for better tracking accuracy)

load('cars4_truth.mat')
[~,N,F] = size(x);
D = 2*F;
X = reshape(permute(x(1:2,:,:),[1 3 2]),D,N); %Normalized locations
Y = reshape(permute(y(1:2,:,:),[1 3 2]),D,N); %Original locations

initLocations = [Y(1,:);480-Y(2,:)]; %Initial featur location on the first frame; this step can be replaced by any feature detector
preName = ['Frame_' num2str(0) '.jpg'];
preFrame = double(imread(preName))/255;%Read the template image
preLocations = initLocations;

figure(1)
hImage = imshow(preFrame);
hold on
plot(preLocations(1,:), preLocations(2,:),'ro')

Params.init_u0 = zeros(2,N);
for frame = 1:F-1      
    currName = ['Frame_' num2str(frame) '.jpg'];
    currFrame = double(imread(currName))/255; %Read current image     
        
    %%%%%%%%%% Main Function %%%%%%%%%%%    
    [currLocations,u0] = EpipolarSubspaceConstrainedTracking(preLocations, preFrame, currFrame, Params);    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    figure(1);
    imshow(currFrame)    
    hold on
    plot(currLocations(1,:), currLocations(2,:),'ro')
    drawnow
    
    Params.init_u0 = u0;
    preFrame = currFrame;
    preLocations = currLocations;         
end
close all
cd ..