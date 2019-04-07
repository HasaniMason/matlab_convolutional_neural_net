% Stage 4:
% Lets dig up the instances that were incorrectly classified
% This may be useful for debugging or for Stage 6: Recalibration

% see which supervised images were poorly predicted
% LOAD convnets and ALL Stage 3 vars before running this

% initialize
if prepare_for_deployment
    search_datastore = datastore;
else
    search_datastore = testingSetBase;
end

predictionEnsemble{length(search_datastore.Files)} = [];

for k = 1:length(search_datastore.Files)
    predictionEnsemble{k} = containers.Map;
    for j = 1:length(alphabetical_labels)
        predictionEnsemble{k}(alphabetical_labels{j}) = 0; 
    end
end

% predict
for j = 1:(ensembles*k_deploy)
    predictions = classify(convnet{j}, search_datastore);
    parfor k = 1:length(predictions)
        response = char(predictions(k));
        predictionEnsemble{k}(response) = predictionEnsemble{k}(response) + 1;
    end
end

% vote
count_good = 0;
which_bad{length(predictionEnsemble)} = [];
max_votes{length(predictionEnsemble)} = [];
parfor j = 1:length(predictionEnsemble)
    max_label = '';
    max_num = 0;
    for k = 1:length(predictionEnsemble{j})
        keys = predictionEnsemble{j}.keys;
        key = keys{k};
        if predictionEnsemble{j}(key) > max_num
            max_num = predictionEnsemble{j}(key);
            max_label = key;
        end
    max_votes{j} = max_label;
    end
    
    % compare to actual label
    if max_votes{j} == search_datastore.Labels(j) %#ok<PFBNS>
        count_good = count_good + 1;
        which_bad{j} = 0;
    else
        which_bad{j} = 1;
    end
end

% this stores the indices of the images in the datastore that were poorly
% predicted
which_bad = find(cell2mat(which_bad));

vote_accuracy = count_good / length(predictionEnsemble);

% run the below code to show the images that sucked
% for i = which_bad
%     manual_check(i, max_votes, search_datastore)
% end
% 
% function manual_check(index, max_v, ds)
%     figure;
%     imshow(readimage(ds,index));
%     disp(strcat('Predicted: ', max_v(index)))
%     disp(strcat('Actual:    ', char(ds.Labels(index))))
%     disp(strcat('File name: ', char(ds.Files(index))))
%
% end
% %close all;