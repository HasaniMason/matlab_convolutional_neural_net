% Stage 2:
% After an initial set of hyperparams is chosen on the first fold in Stage
% 1, we want to make sure that set of hyperparams is good across the entire
% training set

% Performs bayesian optimization with voting consistency as a constraint
% this is to refine stage 1 because it is k times more expensive per 
% optimization run than a standard optimization search

alphabetical_labels = {'class_1', 'class_2', 'class_3'};

k = 3; % reverse cross-val: train on one fold, validate on k-1 folds
% this is so we have something to vote (a vote of 1 isn't helpful)

train_percent = 0.80;
% MUST be consistent across all stages to properly exclude test set

% number of classifiers out of k that must predict the same thing to be
% counted as strong agreement
consistency_threshold = 3;

% number of classifiers out of k that must predict the same thing to be
% counted as weakly in agreement
consistency_threshold_weak = 2;

% Percent of the data that the consisenty_threshold must be satisfied in order
% for the classifiers to be considered "consistent"
% 1 is perfect
score_threshold = 0.6;

% After the strong consistency is satisfied, we also want to make sure the
% remaining data that the classifiers are not strongly consistent on must
% at least be weakly consistent this percent of the time
score_threshold_weak = 0.5; % percent remaining after strong satisfied
% note that balanced data is required (provided below via bootstrap)

% each optimization run here will take k times as long to complete as an
% optimization run in stage 1
max_optimization_runs = 10;

max_epochs = 24;
mini_batch = 320;

validation_freq = 8192; % in iterations, also controls verbose frequency
validation_patience = inf; % early stopping if valid loss inc this many times
verbose = false;

bootstrapFactor = 8; % how big to make bootstrap compared to num images

% other vars are hard coded so param passing doesn't have to be changed
% recommended to use a more limited set of variables here than in Stage 1
% only work on what you are most unsure about
optimVars = [
    optimizableVariable('SQDF', [0.1 0.9999],'Transform','log')
    optimizableVariable('LRDF', [0.1 0.9])
    ];

augmentedResolution = [128 128];
inputResolution = augmentedResolution;
inputResolution(3) = 3; % color dimension
augmenter = imageDataAugmenter('RandRotation', [-4 4], ...
    'RandXTranslation', [-4 4], ...
    'RandYTranslation', [-4 4], ...
    'RandXShear', [-4 4], ...
    'RandYShear', [-4 4], ...
    'RandXReflection', true);

% sanity check, make sure I don't set something stupid
if consistency_threshold > k || consistency_threshold_weak > consistency_threshold
    error("Either raise k or reduce consistency threshold")
end

datastore = imageDatastore(fullfile('.'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

rng(1); % for reproducible results
[crossValSet, testingSetBase] = splitEachLabel(datastore, train_percent);
% we won't use the testingSet here

crossValSet = shuffle(crossValSet);
labelCounts = countEachLabel(crossValSet);
labelCounts = labelCounts.Count;
weights = labelCounts/sum(labelCounts);
weights = weights.^(-1); % inverse so big is small and small is big
labels = crossValSet.Labels;

weightVec = zeros(1,length(labels));
for lab = 1:length(labels)
    for labidx = 1:length(alphabetical_labels)
        if labels(lab) == alphabetical_labels(labidx)
            weightVec(lab) = weights(labidx);
        end
    end
end

partStores{k} = [];
results_tracker{k} = [];

crossValSetFiles = crossValSet.Files;
bootstrapSize = round(length(crossValSetFiles) * bootstrapFactor);
crossValSetBootstrap = datasample(crossValSetFiles, bootstrapSize, 'Weights', weightVec);
bootstrapStore = imageDatastore(crossValSetBootstrap, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

for i = 1:k
   temp = partition(bootstrapStore, k, i);
   partStores{i} = temp.Files;
end

ObjFcn = makeObjFcn(k, partStores, augmentedResolution, augmenter,...
    consistency_threshold, score_threshold, ...
    consistency_threshold_weak, score_threshold_weak, ...
    mini_batch,...
    max_epochs,...
    verbose,...
    validation_freq,...
    validation_patience,...
    inputResolution, ...
    alphabetical_labels);

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxObj',max_optimization_runs,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false, ...
    'NumCoupledConstraints', 2);

bestIdx = BayesObject.IndexOfMinimumTrace(end);
disp(BayesObject.ObjectiveTrace(bestIdx));
bestNetFName = BayesObject.UserDataTrace{bestIdx};

bayesData = load(bestNetFName);
convnets = bayesData.convnets;
allPredictions = zeros(length(alphabetical_labels),length(crossValSet.Files));
for i = 1:length(convnets)
   convnet = convnets{j};
   classifications = classify(convnet, crossValSet);
   for g = 1:length(alphabetical_labels)
       parfor h = 1:length(classifications)
           if alphabetical_labels{g} == classifications(h) %#ok<PFBNS>
               allPredictions(g,h) = allPredictions(g,h) + 1;
           end
       end
   end
end

j = 1;
weak = zeros(0);
for i = 1:length(crossValSet.Files)
   votes = allPredictions(:,i);
   if max(votes) < 1 + floor(k / 2)
       weak(j) = i;
       j = j+1;
   end
end

% gets the full sized images of the badly predicted ones and shows them
%weakImg = getWeak(weak, crossValSet.Files, allPredictions, alphabetical_labels);

% function weak = getWeak(weak, files, preds, labels)
%     j=1;
%     weak{j} = [];
%     for i = 1:length(weak)
%        [p,n,e]=fileparts(files{weak(i)});
%        splits = split(p,'\');
%        
%        keep_path = splits(1:length(splits)-2);
%        
%        votes = preds(:,weak(i));
%        which_label = max(votes) == votes;
%        label = labels{which_label};
%        
%        undsc_split = split(label, '_');
%        biglabel = undsc_split{1};
%        
%        keep_path{length(splits)-1} = biglabel;
%        
%        custom_split = split(n, " (Custom)");
%        keep_path{length(splits)} = custom_split{1};
%        showPath = join(keep_path,"\");
%        showFile = strcat(showPath{1}, e);
%        if exist(showFile, 'file') == 0
%            % weakly and incorrectly classified
%            weak{j} = showFile;
%            j = j + 1;
%        end
%     end
%     return;
% end

function ObjFcn = makeObjFcn(k, partStores, augmentedResolution, augmenter,...
    consistency_threshold, score_threshold, ...
    consistency_threshold_weak, score_threshold_weak, ...
    mini_batch,...
    max_epochs,...
    verbose,...
    validation_freq,...
    validation_patience,...
    inputResolution, ...
    alphabetical_labels)

    ObjFcn = @valErrorFun;
    function [objective,constraints,fname] = valErrorFun(optVars,~)
        
        idx = crossvalind('Kfold', k, k);
        validate_accuracy = zeros(1,k);
        
        max_len = 0;
        for i = 1:k
            max_len = max_len + length(partStores{i});
        end
        
        validatePredictions = zeros(length(alphabetical_labels), max_len);
        convnets{k} = [];
        all_options{k} = [];
        for i=1:k
            % intentionally backwards so that we can do consistenty voting with
            % more extrapolation required from each model
            % validate on everything needed to vote without mapping file names to indices
            train_idx = (idx == i);
            validate_idx = idx;

            validate_StoreBase = imageDatastore(cat(1, partStores{validate_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
            train_StoreBase = imageDatastore(partStores{train_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

            validate_Store = augmentedImageDatastore(augmentedResolution, validate_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
            train_Store = augmentedImageDatastore(augmentedResolution, train_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);

            options = trainingOptions('rmsprop', 'MiniBatchSize', mini_batch, ...
            'MaxEpochs', max_epochs, 'InitialLearnRate', 0.000999, ...
            'LearnRateSchedule', 'piecewise', ...
            'LearnRateDropPeriod', 20, ...
            'LearnRateDropFactor', optVars.LRDF, ...
            'L2Regularization', 0.00001, ...
            'SquaredGradientDecayFactor', optVars.SQDF, ...
            'Verbose', verbose, 'VerboseFrequency', validation_freq, ...
            'ValidationFrequency', validation_freq, ...
            'ValidationData', validate_Store, ...
            'ValidationPatience',validation_patience, ...
            'Shuffle', 'every-epoch');
            
            all_options{k} = options;
            
            initialNumFilters = 16;
            network_depth = 2;
            layers = [
                imageInputLayer(inputResolution);
                convBlock(3,initialNumFilters,network_depth);
                maxPooling2dLayer(2,'Stride',2);

                convBlock(3,2*initialNumFilters,network_depth);
                maxPooling2dLayer(2,'Stride',2);

                convBlock(3,4*initialNumFilters,network_depth);
                maxPooling2dLayer(2,'Stride',2);

                batchNormalizationLayer();
                reluLayer();
                dropoutLayer(0.2);
                fullyConnectedLayer(3);
                softmaxLayer();
                classificationLayer();
                ];

            convnet = trainNetwork(train_Store, layers, options);
            convnets{i} = convnet;

            classifications = classify(convnet, validate_Store);
            validateLabels = validate_StoreBase.Labels;
            validate_accuracy(i) = sum(classifications == validateLabels)/numel(validateLabels);

            for j = 1:length(classifications)
                for l = 1:length(alphabetical_labels)
                    if alphabetical_labels{l} == classifications(j)
                        validatePredictions(l,j) = validatePredictions(l,j) + 1;
                    end
                end
            end
        end
        
        consistency_score = 0;
        consistency_score_weak = 0;
        for i=1:length(validatePredictions(1,:))
            votes = validatePredictions(:,i);
            % condition on sum < k-1 in case there is a file that didn't get
            % classified by everyone
            if max(votes) >= consistency_threshold
               consistency_score = consistency_score + 1;
            elseif max(votes) >= consistency_threshold_weak
                consistency_score_weak = consistency_score_weak + 1;
            end
        end
        
        % make negative b/c pos. constraint means constraint not satisfied
        consistency_score = -consistency_score / length(validatePredictions(1,:));
        remaining = length(validatePredictions(1,:)) + (length(validatePredictions(1,:)) * consistency_score) + 1e-16;
        consistency_score_weak = -consistency_score_weak / remaining;
        
        consistency_score = consistency_score + score_threshold;
        consistency_score_weak = consistency_score_weak + score_threshold_weak;
        
        % consistency fix, if score is too high, weak is impossible
        if consistency_score >= 1 - score_threshold_weak
            consistency_score_weak = -score_threshold_weak;
        end
        
        % only needed if verbose is true
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'));
        
        objective = 1 - mean(validate_accuracy);
        
        % checkpointing
        % dicomuid generates a unique id
        fname = num2str(objective) + "_" + dicomuid + ".mat";
        save(fname,'objective','all_options','convnets','consistency_score','consistency_score_weak');
        
        constraints = [consistency_score consistency_score_weak];
    end
end

function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1);
end