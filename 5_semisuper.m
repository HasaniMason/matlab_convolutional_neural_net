% Stage 5:
% If you have unlabelled data, this will attempt to classify it

% run pretrained models on new, unlabeled data
% LOAD production convnets and ALL Stage 3 vars before running this

% We will look at which two categories get the most votes for each instance
% If the difference between the number of votes between the top two classes
% is greater than TopDiff, we'll classify it as the most voted class
% otherwise, we'll say the classifiers are undecided

% We will then be nice and move the images into the right folders for you
% Name the folder that contains unlabeled images "temp", and the folder
% that contains the resized images "temp_resize" (resized to
% inputResolution, they must have the same names as the fullsize images)
% Create a temp_blahblahblah folder for each class you have
% (replace blahblahblah with the class name)
% Also, create a temp_undecided folder which will store images that we
% aren't sure about (more on that below)
% Finally, create a blahblahblah folder for each class you have (these will
% hold the resized images)

% large TopDiff suggests strong difference between top two categories
% small TopDiff suggests weak or no difference between the categories
% the larger this value, the larger the difference has to be for the diff
% to be considered strong; must be raised by an integer to have any effect
% reduce this value if too many ties (this also suggests bad convnets)
% also reduce this value if num ensembles * k folds is small
% max TopDiff possible is num ensembles * k folds
TopDiff = 2;

semiSupervised = imageDatastore(fullfile('temp_resize/.'));
predictionEnsemble = []; % clears it out if ran poorly_predicted_search
predictionEnsemble{length(semiSupervised.Files)} = [];

parfor k = 1:length(semiSupervised.Files)
    predictionEnsemble{k} = containers.Map;
    for j = 1:length(alphabetical_labels)
        predictionEnsemble{k}(alphabetical_labels{j}) = 0; 
    end
end

% predict
for j = 1:(ensembles*k_deploy)
    predictions = classify(convnet{j}, semiSupervised);
    parfor k = 1:length(predictions)
        response = char(predictions(k));
        predictionEnsemble{k}(response) = predictionEnsemble{k}(response) + 1;
    end
end

% vote
max_votes{length(semiSupervised.Files)} = [];
topDiffValues = zeros(1, length(semiSupervised.Files));
parfor j = 1:length(predictionEnsemble)
    keys = predictionEnsemble{j}.keys;
    values = cell2mat(predictionEnsemble{j}.values);
    max_label = keys{values == max(values)};
    max_votes{j} = max_label;
    
    nonMax = values(values < max(values));
    if length(nonMax) >= 1
        topDiffValues(j) = max(values) - max(nonMax);
    else % all values are tied, technically not necessary but cleaner
        topDiffValues(j) = 0;
    end
end

% move resized images
% doesn't work with abs path names, no idea why
% but that keeps things portable at least
cd 'temp_resize'
files = semiSupervised.Files;
targets{length(files)} = [];
sources{length(files)} = [];
parfor j = 1:length(predictionEnsemble)
    fullName = files{j};
    [fpath, fname, fext] = fileparts(fullName);
    source = strcat(fname, fext);
    
    if topDiffValues(j) > TopDiff
        target_path = max_votes{j};
    else
        target_path = 'undecided_r';
    end
    
    target = strcat('..\', target_path, '\', fname, fext);
    targets{j} = target;
    sources{j} = fullName;
end

parfor j = 1: length(files)
   movefile(sources{j}, targets{j}); 
end
cd '..'

% move fullsized images (names must be ident)
cd 'temp'
targets{length(files)} = [];
sources{length(files)} = [];
parfor j = 1:length(predictionEnsemble)
    fullName = files{j};
    [fpath, fname, fext] = fileparts(fullName);
    source = strcat(fname, fext);
    
    if topDiffValues(j) > TopDiff
        target_path = strcat('temp_', max_votes{j});
    else
        target_path = 'temp_undecided';
    end
    
    target = strcat('..\', target_path, '\', fname, fext);
    targets{j} = target;
    
    pathAr = split(fpath, '\');
    miniPathAr = pathAr(1:length(pathAr)-1);
    miniPath = char(join(miniPathAr, '\'));
    sources{j} = strcat(miniPath, '\temp\', fname, fext);
end

parfor j = 1: length(files)
    movefile(sources{j}, targets{j});
end
cd '..'