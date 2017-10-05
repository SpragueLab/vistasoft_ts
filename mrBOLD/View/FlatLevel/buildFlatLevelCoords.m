function buildFlatLevelCoords(view,pathStr)
% buildFlatLevelCoords(view,pathStr);
%
% For flat level view, build the file
% coordsLevels, containing information about
% the location of coordinates across, as well as
% between, gray levels.
%
%  anat: 3D matrix with the following slice order: 
%       first slice -- LH, across levels,
%       second slice -- RH, across levels,
%       slices 3:numLeftLevels+2: separate left levels
%       numLeftLevels+3:end: separate right levels
%
%  map,amp,ph,co: cell of size numScans. Each entry
%  contains a map resized to be the same size as the
%  anat field. The code will therefore treat
%  overlaying fields on the anat in the same manner
%  as with inplane slices. The UI simply selects
%  the appropriate slice subset for display based on 
%  the hemisphere buttons / level controls.
%
%   coords: a 1 x 2 cell, I guess. 
%
% ROIs.coords: ROI coords will specify x,y, and slice num,
% similar to Inplane ROI coords
%
% ras 09/04. TO DO: add indices matrix.
global HOMEDIR 

imSize = [0 0];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 1) Get coords separately for each hemisphere in a cell %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
waitHandle = mrvWaitbar(0,'Computing flat coordinates.  Please wait...');
for h = 1:2
    mrvWaitbar((h-1)/2)
    
    % get left and right flat paths for this unfold from
    % the existing Flat[name]/coords.mat file
    oldCoordsPath = fullfile(HOMEDIR,view.subdir,'coords.mat');
    if exist(oldCoordsPath,'file')
        load(oldCoordsPath,'leftPath','rightPath');
    else
        leftPath = []; rightPath = [];
    end
    
    % Load gLocs2d and gLocs3d
    if h==1
        [gLocs2d,gLocs3d,curvature,leftPath] = loadGLocs('left',leftPath);
    else
        [gLocs2d,gLocs3d,curvature,rightPath] = loadGLocs('right',rightPath);
    end
    
    if isempty(gLocs2d) || isempty(gLocs3d)
        levelCoords{h} = [];
        grayCoords{h} = [];
        numLevels(h) = 0;
    else
        % Compute imSize
        imSize = max(imSize,(max(gLocs2d,[],2) - min(gLocs2d,[],2) + 1)');
        imSize = round(imSize);
        
        % Find gray nodes that are both in the inplanes and included
        % in the unfold.
        % gray.coords are the gray coords that lie in the inplanes.
        % gLocs3d are the gray coords in the unfold.
        hiddenGray = initHiddenGray;
        [grayCoordsTmp,gLocsIndices,coordsIndices] = ...                
            intersectCols(gLocs3d,hiddenGray.coords);
        
        % Flat locations corresponding to those voxels
        levelCoords{h} = gLocs2d(:,gLocsIndices);
        
        % Add third dimension for each node: the gray level
        % (this is taken from row 6 of the nodes struct
        % generated by mrFlatMesh)
        levelCoords{h}(3,:) = hiddenGray.nodes(6,coordsIndices);
        
        grayCoords{h} = grayCoordsTmp;
        
        % count the # of gray levels for this view
        numLevels(h) = length(unique(levelCoords{h}(3,:)));

        % round the coords to integer values -- this may not be 
        % strictly necessary, and in some way the precise 2D 
        % locations may be useful, but we save on space and simplicity
        levelCoords{h} = round(levelCoords{h});
        
        % there sometimes are redundant x,y positions in each level
        % (mutiple gray nodes in the first level map to the same 
        % node in a higher level -- even w/o rounding): get rid of these
        [levelCoords{h} ia] = intersectCols(levelCoords{h},levelCoords{h});
        grayCoords{h} = grayCoords{h}(:,ia);
        
        % Warning if there are any NaNs in the coords.
        NaNs = find(isnan(levelCoords{h}(1,:)) | isnan(levelCoords{h}(2,:)));
        if ~isempty(NaNs)
            myWarnDlg(['You have ',int2str(length(NaNs)),' NaNs in your flat coords.  ',...
                    'Those gray matter nodes will not be rendered in the FLAT view.']);
        end
    end
end
close(waitHandle);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 2) Get coordinates across levels for each hemisphere      %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~isempty(levelCoords{1})
	leftAvgCoords = unique(levelCoords{1}(1:2,:)','rows')';
	leftAvgCoords(3,:) = 1;    % first slice
else
    leftAvgCoords = [];
end

if ~isempty(levelCoords{2})
    rightAvgCoords = unique(levelCoords{2}(1:2,:)','rows')';
    rightAvgCoords(3,:) = 2;   % second slice
else
    rightAvgCoords = [];
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3) Combine everything for each hemisphere          %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% first, convert the 3rd row in levelCoords from gray levels
% to slice # (this takes into account the across-level indices, 
% and, for the right hemisphere, the left levels)

% L: add the acr-level slices
if ~isempty(levelCoords{1})
	levelCoords{1}(3,:) = levelCoords{1}(3,:) + 2; 
end
% R: add the acr-level slices & left levels
if ~isempty(levelCoords{2})
	levelCoords{2}(3,:) = levelCoords{2}(3,:) + 2 + numLevels(1); 
end

% merge (initial merge -- see below)
coords = [leftAvgCoords levelCoords{1} rightAvgCoords levelCoords{2}];

% build gray coords: there are no gray coordinates
% corresponding to the across-voxel locations, so 
% set these to NaN
grayCoords = [NaN*ones(size(leftAvgCoords)) ...
              grayCoords{1} ...
              NaN*ones(size(rightAvgCoords)) ...
              grayCoords{2}];
       
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 4) Fix for disk/memory limitations: subdivide      %
% fields as cells-of-slices; later save tSeries by slice  %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for slice = unique(coords(3,:))
    subInd = find(coords(3,:)==slice);
    tmpCoords{slice} = coords(1:2,subInd);
    tmpGrayCoords{slice} = grayCoords(:,subInd);
end
coords = tmpCoords;
grayCoords = tmpGrayCoords;

% make the door swing both ways (cross the streams!):
% make indices for mapping from the image maps (anat, 
% amp, map, etc) to the coordinates (tSeries)
numSlices = 2 + numLevels(1) + numLevels(2);
indices = zeros(imSize(1),imSize(2),numSlices);
for slice = 1:numSlices
    indSlice = zeros(imSize(1),imSize(2));
    tmp = round(coords{slice});
	if ~isempty(tmp)
        tmpInd = sub2ind(size(indSlice),tmp(1,:),tmp(2,:));
		indSlice(tmpInd) = 1:size(tmp,2);
	end
    indices(:,:,slice) = indSlice;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 5) Save                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% add a note in case anyone stumbles on the file
note = 'These are the coords for viewing data separately across gray levels.';

% Save to file
save(pathStr,'coords','grayCoords','indices','imSize',...
    'numLevels','leftPath','rightPath','note');
fprintf('Saved %s. \n',pathStr);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Side-step) Existing flat prefs are obsolete, need to delete %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
prefsFile = fullfile(viewDir(view),'userPrefs.mat');
if exist(prefsFile,'file')
    delete(prefsFile);
end

return



% lots of old code:


% % Now I need to do something ugly: I need to unmerge.
% % This is purely for back compatibility: previous
% % flat views separate the coords across hemispheres.
% % Sometime, soon I hope, all the code that assumes
% % it will be this way will be updated, and coords
% % and grayCoords can remain a single matrix.
% coords = cell(1,2);
% coords{1} = [leftAvgCoords levelCoords{1}];
% coords{2} = [rightAvgCoords levelCoords{2}];
% 
% % build gray coords: there are no gray coordinates
% % corresponding to the across-voxel locations, so 
% % set these to NaN
% grayCoords{1} = [NaN*ones(size(leftAvgCoords)) ...
%               grayCoords{1} ...
%               ];
% grayCoords{2} = [NaN*ones(size(rightAvgCoords)) ...
%               grayCoords{2} ...
%               ];



%         % there sometimes are redundant x,y positions in each level
%         % (mutiple gray nodes in the first level map to the same 
%         % node in a higher level -- even w/o rounding): get rid of these
%         oldCoords = levelCoords{h}; % oldCoords will be useful below
%         newCoords = [];
%         for level = 1:numLevels(h)
%             subInd = find(levelCoords{h}(3,:)==level);
%             subCoords = levelCoords{h}(1:2,subInd);
%             subCoords = unique(subCoords','rows')';
%             subCoords(3,:) = level;
%             newCoords = [newCoords subCoords];
%         end
%         levelCoords{h} = newCoords;
%         
%         % now we have to go back and figure out the corresponding
%         % gray coords (this is much trickier after removing the
%         % redundant coords, but I believe it's both correct/necessary):
%         [ignore subGLocsInd] = intersectCols(oldCoords,newCoords);
%         grayCoords{h} = gLocs3d(:,gLocsIndices(subGLocsInd));

% numSlices = 2 + numLevels(1) + numLevels(2);
% indices = zeros(imSize(1),imSize(2),numSlices);
% tmp = round(coords);
% 	if ~isempty(tmp)
%         tmpInd = sub2ind(size(indices),tmp(1,:),tmp(2,:),tmp(3,:));
% 		indices(tmpInd) = 1:size(tmp,2);
% 	end
