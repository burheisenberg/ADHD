function [accuracy, sensitivity, precision, f1Score] = calculateMetrics(YTest, YPred)

    % Ensure categorical data type
    YTest = categorical(YTest);
    YPred = categorical(YPred);

    % Get unique categories
    categories = unique(YTest);

    % Calculate metrics for each category
    accuracy = 0;
    for i = 1:numel(categories)
        category = categories(i);
        TP = sum(YTest == category & YPred == category);
        TN = sum(~ismember(YTest, category) & ~ismember(YPred, category));
        FP = sum(~ismember(YTest, category) & ismember(YPred, category));
        FN = sum(ismember(YTest, category) & ~ismember(YPred, category));

        accuracy = accuracy + (TP + TN) / (TP + TN + FP + FN);
    end
    accuracy = accuracy / numel(categories);  % Average accuracy across categories

    % Calculate sensitivity, precision, and F1-score (assuming a single positive category)
    sensitivity = TP / (TP + FN);
    precision = TP / (TP + FP);
    f1Score = 2 * (precision * sensitivity) / (precision + sensitivity);

end
