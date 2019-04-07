% Stage 6:
% Recalibrate the neural network after you manually fix the undecided
% images in Stage 5. Or if you trust the neural network so much you can
% just run this on any newly labelled data from Stage 5.

% Put the images to be used for reclibration in its own working folder,
% with each sub-folder name corresponding to a class.

% use this after fixing undecidable instances from semisuper.m
% load Stage 3 workspace before running this
% prepare for deployment must be true
% put recalibration set in its own working directory before merging with
% the main working directory of classified images
% this is effectively transfer learning

% Once done and you are happy with the results you can save this workspace

rng(0); % for reproducible results

max_epochs = 2; % this is for bringing new images into an already trained network, should be small
mini_batch = 256;
recalibrate_learn_rate = initial_learn_rate * 0.4; % keep this small for recalibration, but not too small (prevent overfitting)
recalib_bootstrapFactor = 6; % if this is too big the effective max epochs will be higher than what is set
k = k_deploy;
recalibration_datastore = imageDatastore(fullfile('.'), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

train_accuracyR{k * ensembles} = [];
validate_accuracyR{k * ensembles} = [];
test_accuracyR{k * ensembles} = [];

partStoresR{k} = [];
crossValSetR = shuffle(recalibration_datastore);
labelCountsR = countEachLabel(crossValSetR);
labelCountsR = labelCountsR.Count;
weightsR = labelCountsR/sum(labelCountsR);
weightsR = weightsR.^(-1); % inverse so big is small and small is big
labelsR = crossValSetR.Labels;

weightVecR = zeros(1,length(labelsR));
for lab = 1:length(labelsR)
    for labidx = 1:length(alphabetical_labels)
        if labelsR(lab) == alphabetical_labels(labidx)
            weightVecR(lab) = weightsR(labidx);
        end
    end
end

for j = 1:ensembles
    crossValSetFilesR = crossValSetR.Files;
    bootstrapSizeR = round(length(crossValSetFilesR) * recalib_bootstrapFactor);
    crossValSetBootstrapR = datasample(crossValSetFilesR, bootstrapSizeR, 'Weights', weightVecR);
    bootstrapStoreR = imageDatastore(crossValSetBootstrapR, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

    for i = 1:k
       tempR = partition(bootstrapStoreR, k, i);
       partStoresR{i} = tempR.Files;
    end
    
    idx = crossvalind('Kfold', k, k);
    for i = 1:k
        % keeps cell arrays 1 dimensional
        nested_idx = (j-1) * k + i;
        
        validate_idx = (idx == i);
        train_idx = ~validate_idx;

        validate_StoreBaseR = imageDatastore(partStoresR{validate_idx}, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        train_StoreBaseR = imageDatastore(cat(1, partStoresR{train_idx}), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
        
        validate_StoreR = augmentedImageDatastore(augmentedResolution, validate_StoreBaseR, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
        train_StoreR = augmentedImageDatastore(augmentedResolution, train_StoreBaseR, 'DataAugmentation', augmenter, 'DispatchInBackground', false);
        
        % most of these options are loaded from the cnn.m workspace
        options = trainingOptions('rmsprop', 'MiniBatchSize', mini_batch, ...
        'MaxEpochs', max_epochs, 'InitialLearnRate', recalibrate_learn_rate, ...
        'LearnRateSchedule', 'piecewise', ...
        'LearnRateDropPeriod', learn_rate_drop_period, ...
        'LearnRateDropFactor', learn_rate_drop_factor, ...
        'L2Regularization', l2reg, ...
        'SquaredGradientDecayFactor', squared_gradient_decay_factor, ...
        'Verbose', verbose, 'VerboseFrequency', validation_freq, ...
        'ValidationFrequency', validation_freq, ...
        'ValidationData', validate_StoreR, ...
        'ValidationPatience',validation_patience, ...
        'Plots', 'training-progress', ...
        'Shuffle', 'every-epoch');
        
        % train the existing network further, i.e. transfer learning
        convnet{nested_idx} = trainNetwork(train_StoreR, convnet{nested_idx}.Layers, options); %#ok<SAGROW>

        trainPredictions = classify(convnet{nested_idx}, train_StoreR);
        trainLabels = train_StoreBaseR.Labels;
        train_accuracyR{nested_idx} = sum(trainPredictions == trainLabels)/numel(trainLabels);
        disp(train_accuracyR{nested_idx})

        validatePredictions = classify(convnet{nested_idx}, validate_StoreR);
        validateLabels = validate_StoreBaseR.Labels;
        validate_accuracyR{nested_idx} = sum(validatePredictions == validateLabels)/numel(validateLabels);
        disp(validate_accuracyR{nested_idx})
    end
end

rec_act = mean(cell2mat(train_accuracyR));
rec_acv = mean(cell2mat(validate_accuracyR));
orig_act = mean(cell2mat(train_accuracy));
orig_acv = mean(cell2mat(validate_accuracy));
ov_act = (rec_act + orig_act) / 2;
ov_acv = (rec_acv + orig_acv) / 2;

disp("Recalibrated Mean:")
disp(rec_act)
disp(rec_acv)

disp("Original Mean:")
disp(orig_act)
disp(orig_acv)

disp("Overall, Unweighted Mean:")
disp(ov_act)
disp(ov_acv)