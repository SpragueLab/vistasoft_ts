function model = rmGridFit_oneGaussianNonlinear_GPU(model,prediction,data,params,t)
% rmGridFit_oneGaussianNonlinear - core of one non-linear (exponential) Gaussian fit
% CSS or compressive spatial summation model
% - updated to perform grid search on GPU, 15-40x speed improvement,
%   depending on system speed (40 mins for whole brain, 500k models)
%
% model = rmGridFit_oneGaussianNonlinear_GPU(model,prediction,data,params,trends);
%
% 2017.02.16 TCS duplicated from rmGridFit_oneGaussianNonlinear.m, replaced
% linear gridfit operations with GPU-accelerated version; requires
% gpuRegress tools from TCS (tsprague@nyu.edu; tommy.sprague@gmail.com)
%
% 2014/02 JW: duplicated from rmGridFit_oneGaussian, then exponent added to
% model

% input check 
if nargin < 5,
    error('Not enough arguments');
end

trends         = t.trends;
t_id           = t.dcid+1;

% adapted from run_gridfitgpu.m 2/16/2017
model_preds = single(nan(size(trends,1),1+size(trends,2),size(prediction,2)));
model_preds(:,1,:) = prediction;
model_preds(:,1+(1:size(trends,2)),:) = repmat(trends,1,1,size(prediction,2)); % incl 3 trends, model is ~3 GB


% only look at voxels with 'real' data (std dev or range is not 0)
% - for surface files, this should reduce computation time by ~10-15x!


goodvox = std(data,[],1)~=0;


% we compute mean rss but we need sum rss (old convention)
model.rss=single(model.rss./(size(prediction,1)- (size(trends,2)+1)));  


%------------------------------------
% offload the work of computing the model fits to gridfitgpu.m

% idx is the idx within model_preds of the best-fit model for each voxel
% b is the best-fit betas for each predictor (4)
% rss is residual sum of squares
if sum(goodvox) > 0
    % maybe if multiple GPUs available, create a parpool here (if doesn't
    % already exist, if it does and has == nGPUs workers, use that one; if
    % it has > nGPUs workers, kill it and create one w/ nGPUs workers)
    % - split data(:,goodvox) into cell array
    if gpuDeviceCount == 1
        [idx,b,rss] = gridfitgpu    (data(:,goodvox),model_preds,1); % 3rd arg: whether or not to truncate neg fits
    elseif gpuDeviceCount >= 1
        [idx,b,rss] = gridfitgpu_par(data(:,goodvox),model_preds,1); % 3rd arg: whether or not to truncate neg fits
    else
        error('vistasoft_ts:mrBOLD:rmGridFit_oneGaussianNonlinear_GPU:GPUNotFound','No GPU detected!');
    end
else
    fprintf('No signal voxels this run..., moving on....\n');
    idx = []; b = []; rss = [];
end
    
    
idx(isnan(idx)) = 1;




model.rss(goodvox)      = rss.';     % to make sure same dims as rawrss
model.b([1 t_id],goodvox) = b.';
%model.b
%warning('on', 'MATLAB:lscov:RankDefDesignMat')

% Under some conditions, the grid fit never returns an acceptable fit, For
% example for onegaussian fits with data driven DC component, when the DC
% is artificially high. In this case some of the rss values remain Inf,
% which fails to interpolate and compute correct variance explained values.
% So we check it here and reset any Inf (bad fits) to rawrss, so the
% variance explained will be 0.
model.rss(model.rss==Inf)=model.rawrss(model.rss==Inf);

goodvox_idx = find(goodvox);

% for each voxel, pull out the analysis params corresponding to
% best-fitting model
for ii = 1:length(idx)

    % now update
    model.x0(goodvox_idx(ii))       = params.analysis.x0(idx(ii));
    model.y0(goodvox_idx(ii))       = params.analysis.y0(idx(ii));
    model.s(goodvox_idx(ii))        = params.analysis.sigmaMajor(idx(ii));
    model.s_major(goodvox_idx(ii))  = params.analysis.sigmaMajor(idx(ii));
    model.s_minor(goodvox_idx(ii))  = params.analysis.sigmaMajor(idx(ii));
    model.s_theta(goodvox_idx(ii))  = params.analysis.theta(idx(ii));
    model.exponent(goodvox_idx(ii)) = params.analysis.exponent(idx(ii));
    
    % save the predictions index...used for saving out the best-fit
    % prediction timeseries...
    %model.predidx(goodvox_idx(ii))  = idx(ii); 
    
    %model.b([1 t_id],minRssIndex) = b(:,minRssIndex);
end


% Correct lscov. It returns the mean rss. To maintain compatibility with the
% sum rss this function expects, we have to multiply by the divisor. See
% the lscov docs for details. NOTE TCS 2/17/2017 - I think it's just - size
% trends, not +1, as that's the value used in line 227!!!!!
%model.rss=single(model.rss.*(size(prediction,1)-size(trends,2)+1));  
 
%drawnow;
return;


