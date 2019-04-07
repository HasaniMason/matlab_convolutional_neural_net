% Stage 3:
% After you have a good set of hyperparameters, or if you want to skip
% stages 1 and 2, it is time to build the networks for real use
% make sure to copy the optimum hyperaparams and put them here

% SAVE your workspace after running this for future stages

% delete accuracy between runs if changing number of iterations
% convolutional output area = 1+(inputWidth - filterSize + 2*Padding) / Stride

% labels in alphabetical order, for mapping labels to indices
alphabetical_labels = {'class_1', 'class_2', 'class_3'};

close_graphics = true; % close training plots at the end

% if you finished stages 1 and/or 2, this should be true
prepare_for_deployment = true; % train on everything, ready for production

ensembles = 3; % num of Bootstrap AGgregations (BAGs)
k = 4; % number of folds
k_deploy = 4; % folds if deployment is true
train_percent = 0.80; % amount from each label to use in cross validation

mini_batch = 320;
max_epochs = 24;
initial_learn_rate = 0.000999;
learn_rate_drop_period = 20;
learn_rate_drop_factor = 0.25;
squared_gradient_decay_factor = 0.99;
l2reg = 0.00001;

validation_freq = 2048; % in iterations, also controls verbose frequency
validation_patience = 2; % early stopping if valid loss inc this many times
verbose = false;

bootstrapFactor = 8; % how big to make bootstrap compared to num images

augmentedResolution = [128 128];
inputResolution = augmentedResolution;
inputResolution(3) = 3; % color dimension

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

augmenter = imageDataAugmenter('RandRotation', [-4 4], ...
    'RandXTranslation', [-4 4], ...
    'RandYTranslation', [-4 4], ...
    'RandXShear', [-4 4], ...
    'RandYShear', [-4 4], ...
    'RandXReflection', true);
          
datastore = imageDatastore(fullfile('.'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

rng(1); % for reproducible results
if prepare_for_deployment == false
    [crossValSet, testingSetBase] = splitEachLabel(datastore, train_percent);
    testingSet = augmentedImageDatastore(augmentedResolution, testingSetBase, 'DataAugmentation', augmenter);
else
    train_percent = 1; % value not used but this is effectively what is happening
    k=k_deploy;
    crossValSet = datastore;
end

partStores{k} = [];
convnet{k * ensembles} = [];
train_accuracy{k * ensembles} = [];
validate_accuracy{k * ensembles} = [];
test_accuracy{k * ensembles} = [];

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

for j = 1:ensembles
    
    % do bootstrapping aggregation on crossValSet for ensembles
    % should reduce ensembled overfitting (not overfitting on each run)
    % also use the bootstrapping for class balancing

    crossValSetFiles = crossValSet.Files;
    bootstrapSize = round(length(crossValSetFiles) * bootstrapFactor);
    crossValSetBootstrap = datasample(crossValSetFiles, bootstrapSize, 'Weights', weightVec);
    bootstrapStore = imageDatastore(crossValSetBootstrap, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    for i = 1:k
       temp = partition(bootstrapStore, k, i);
       partStores{i} = temp.Files;
    end
    
    idx = crossvalind('Kfold', k, k);
    for i = 1:k
        % keeps cell arrays 1 dimensional
        nested_idx = (j-1) * k + i;
        
        validate_idx = (idx == i);
        train_idx = ~validate_idx;

        validate_StoreBase = imageDatastore(partStores{validate_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        train_StoreBase = imageDatastore(cat(1, partStores{train_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        validate_Store = augmentedImageDatastore(augmentedResolution, validate_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
        train_Store = augmentedImageDatastore(augmentedResolution, train_StoreBase, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
        
        % datastores change for each fold, so options must be set in the loop
        % every-epoch shuffle prevents discarding the same data due to
        % uneven divison of data into mini-batches
        options = trainingOptions('rmsprop', 'MiniBatchSize', mini_batch, ...
        'MaxEpochs', max_epochs, 'InitialLearnRate', initial_learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', learn_rate_drop_period, ...
        'LearnRateDropFactor', learn_rate_drop_factor, ...
        'L2Regularization', l2reg, ...
        'SquaredGradientDecayFactor', squared_gradient_decay_factor, ...
        'Verbose', verbose, 'VerboseFrequency', validation_freq, ...
        'ValidationFrequency', validation_freq, ...
        'ValidationData', validate_Store, ...
        'ValidationPatience',validation_patience, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch');

        convnet{nested_idx} = trainNetwork(train_Store, layers, options);

        trainPredictions = classify(convnet{nested_idx}, train_Store);
        trainLabels = train_StoreBase.Labels;
        train_accuracy{nested_idx} = sum(trainPredictions == trainLabels)/numel(trainLabels);
        disp(train_accuracy{nested_idx})

        validatePredictions = classify(convnet{nested_idx}, validate_Store);
        validateLabels = validate_StoreBase.Labels;
        validate_accuracy{nested_idx} = sum(validatePredictions == validateLabels)/numel(validateLabels);
        disp(validate_accuracy{nested_idx})

        if prepare_for_deployment == false
            testPredictions = classify(convnet{nested_idx}, testingSet);
            testLabels = testingSetBase.Labels;
            test_accuracy{nested_idx} = sum(testPredictions == testLabels)/numel(testLabels);
            disp(test_accuracy{nested_idx})
        end
    end
end
disp("Overall Mean:")
disp(mean(cell2mat(train_accuracy)))
disp(mean(cell2mat(validate_accuracy)))

if prepare_for_deployment == false
    disp(mean(cell2mat(test_accuracy)))
end

if close_graphics
    close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'));
end


function layers = convBlock(filterSize,numFilters,numConvLayers)
layers = [
    convolution2dLayer(filterSize,numFilters,'Stride',1,'Padding',1)
    batchNormalizationLayer
    reluLayer];
layers = repmat(layers,numConvLayers,1); % repmat = repeat matrix
end