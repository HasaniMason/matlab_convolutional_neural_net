% Stage 1:
% Find an optimum set of hyperparameters over a single partition of the
% dataset via Bayesian optimization

% data should be placed in subfolders listed in the alphabetical_labels
% variable below, only depth 1 should be used (no folders in folders)
% the working directory should be the parent of those subfolders

% the names of the image categories, should also be the names of the folders
% MUST be in alphabetical order (for later stages)
alphabetical_labels = {'class_1', 'class_2', 'class_3'};

k = 4; % pretend cross-validation, only which_fold is used for training
which_fold = 1; % full fledged cross val is expensive and reserved for stage 2
train_percent = 0.80; % remaining percent is not used in optimization at all
% MUST be consistent across all stages to properly exclude test set

% the number of optimization steps we are going to try
% this should be about 10 for each hyperparam you are optimizing
% so optimizing 3 vars = ~30 runs
% but in pracitce you might need to reduce it to keep the runtime
% reasonable
max_optimization_runs = 16;

% the number of times the network sees the data again and again
% one epoch = one pass through ALL the training data
% more = better accuracy, but also slower and > chance of overfitting
max_epochs = 24;

% more is faster, limited by gpu memory
% but too much takes the "stochastic" out of stachastic gradient descent
mini_batch = 320;

% we will conduct a validation check every validation_freq iterations
validation_freq = 8192; % also controls verbose frequency if verbose=true
validation_patience = inf; % early stopping if valid loss inc this many times
verbose = false; % prints results to screen, can use instead of plotting

% class imbalance problem? not enough data? no problem, we can hit two
% birds with one stone via weighted boostrapping through image augmentation
bootstrapFactor = 8; % how big to replicate the num images

% which hyperparameters do you want to optimize? place them here
% other vars are hard coded so param passing doesn't have to be changed
optimVars = [
    optimizableVariable('LRDP', [20 24], 'Type','integer')
    optimizableVariable('initFilt', [10 16], 'Type','integer')
    ];

% the target resolution of the images
% will also control the entry image size of the neural network
% note: augmentation can change the resolution on the fly, but it is slow
% recommended to copy images to the desired resolution first
augmentedResolution = [128 128];
inputResolution = augmentedResolution;
inputResolution(3) = 3; % color dimension

% controls all the agumentations we want to do when we bootstrap the sample
augmenter = imageDataAugmenter('RandRotation', [-4 4], ...
    'RandXTranslation', [-4 4], ...
    'RandYTranslation', [-4 4], ...
    'RandXShear', [-4 4], ...
    'RandYShear', [-4 4], ...
    'RandXReflection', true);

datastore = imageDatastore(fullfile('.'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

rng(1); % for reproducible results
% MUST be consistent across all stages to properly exclude test set
[crossValSet, testingSetBase] = splitEachLabel(datastore, train_percent);
% we won't use the testingSetBase here, but we need to keep it out of the
% optimization step

crossValSet = shuffle(crossValSet);

% count the number of images in each class so we can perform weighted
% sampling, used to handle class imbalance problems
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

idx = crossvalind('Kfold', k, k);
i = which_fold; % real cross val too expensive, will be used in later stages
validate_idx = (idx == i);
train_idx = ~validate_idx;

validate_StoreBase = imageDatastore(partStores{validate_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
train_StoreBase = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% DispatchInBackground interferred with my setup and kept causing low level
% crashes
% you can try turning it on to speed things up a bit
validate_Store = augmentedImageDatastore(augmentedResolution, validate_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
train_Store = augmentedImageDatastore(augmentedResolution, train_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);

% makeObjFcn is defined below
ObjFcn = makeObjFcn(...
    train_Store,...
    validate_Store,...
    validate_StoreBase,...
    mini_batch,...
    max_epochs,...
    verbose,...
    validation_freq,...
    validation_patience,...
    inputResolution,...
    alphabetical_labels);

BayesObject = bayesopt(ObjFcn,...
    optimVars,...
    'MaxObj',max_optimization_runs,...
    'IsObjectiveDeterministic',false,...
    'UseParallel',false,...
    'NumSeedPoints',int8(max_optimization_runs/k));

% close training plotting at the end since there's gonna be tons of them
close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'))

% print out which hyperparam set performed the best
% that model will be stored in a file, so store the file name
bestIdx = BayesObject.IndexOfMinimumTrace(end);
disp(BayesObject.ObjectiveTrace(bestIdx));
bestNetFName = BayesObject.UserDataTrace{bestIdx};

% END of execution
% below code is called by above code (or each other)

% define the objective function we wish to optimize
function ObjFcn = makeObjFcn(train_Store,...
    validate_Store,...
    validate_StoreBase,...
    mini_batch,...
    max_epochs,...
    verbose,...
    validation_freq,...
    validation_patience,...
    inputResolution,...
    alphabetical_labels)

    ObjFcn = @valErrorFun; % execute valErrorFun and return it
    function [objective,constraints,fname] = valErrorFun(optVars,~)
        
        % uncomment the below code if max_epochs and lrdp are being optimized
        % ensures Learn Rate Drop Period is not greater than num epochs
        % make sure to use lrdp, not optVars.LRDP, if optimzing both
        %max_ep = optVars.maxEpoch;
        %lrdp = optVars.LRDP;
        %if lrdp > max_ep
        %    lrdp = max_ep;
        %end
        
        % hard code everything not being optimized so you don't have to
        % change parameter passing every time you change what is being
        % optimized
        % frequently changing params aren't hard coded by default
        options = trainingOptions('rmsprop', 'MiniBatchSize', mini_batch, ...
        'MaxEpochs', max_epochs, 'InitialLearnRate', 0.000999, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', optVars.LRDP, ...
        'LearnRateDropFactor', 0.2, ...
        'L2Regularization', 0.00001, ...
        'SquaredGradientDecayFactor', 0.5, ...
        'Verbose', verbose, 'VerboseFrequency', validation_freq, ...
        'ValidationFrequency', validation_freq, ...
        'ValidationData', validate_Store, ...
        'ValidationPatience',validation_patience, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch');

        initialNumFilters = optVars.initFilt;
        % number of convolution layers that show up in the same block
        network_depth = 2; 
        
        % define your network architecutre here
        % modify convBlock at the bottom if you want to change what the
        % convenience function does
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
            fullyConnectedLayer(length(alphabetical_labels));
            softmaxLayer();
            classificationLayer();
            ];
            
            convnet = trainNetwork(train_Store, layers, options);
            
            % this gets rid of Matlab's poor syntax highlighting issue
            validate_StoreBase;
            max_epochs;

            validatePredictions = classify(convnet, validate_Store);
            validateLabels = validate_StoreBase.Labels;
            validate_accuracy = sum(validatePredictions == validateLabels)/numel(validateLabels);

            objective = 1 - validate_accuracy;
            
            % checkpointing
            % dicomuid generates a unique id
            uid = split(dicomuid, '.');
            fname = num2str(objective) + "_" + uid{length(uid)} + ".mat";
            % save the model for later so we don't loose everything if the
            % computer crashes, for example
            save(fname,'objective','options','convnet','optVars','network_depth','initialNumFilters');
            
            constraints = []; % no constraints
            
            % variables: objective, fname, and constraints must be set to be
            % returned
    end
end

function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1); % repeat layers numConvLayers times
end