%% Initial setup
data = readtable('train_cleaned.csv');
data.Score_Category = categorical(data.Score_Category, {'Poor', 'Standard', 'Good'}, 'Ordinal', true);

%% Important columns
cols = {'Credit_History_Age_Months', 'Num_Bank_Accounts', 'Num_Credit_Card', ...
                 'Num_of_Loan', 'Outstanding_Debt', 'Delay_from_due_date', ...
                 'Num_Credit_Inquiries', 'Score_Category'};

data = data(:, cols);

% Predictor variables
predictor_vars_1 = {'Credit_History_Age_Months', 'Num_Bank_Accounts', ...
                 'Num_of_Loan', 'Outstanding_Debt', 'Delay_from_due_date', ...
                 'Num_Credit_Inquiries'};

predictor_vars_2 = {'Credit_History_Age_Months', 'Num_Bank_Accounts', 'Num_Credit_Card', ...
                 'Num_of_Loan', 'Delay_from_due_date', ...
                 'Num_Credit_Inquiries'};

%% Poor vs Standard Model
data_poor_std = data(data.Score_Category ~= 'Good', :); % Only Poor and Standard
data_poor_std.Poor_vs_Standard = double(data_poor_std.Score_Category == 'Standard'); % Standard = 1
data_poor_std.Score_Category = []; % Remove the original category

% Train the scorecard
sc_poor_std = creditscorecard(data_poor_std, ...
    'IDVar', '', ...
    'ResponseVar', 'Poor_vs_Standard', ...
    'GoodLabel', 1, ...
    'PredictorVars', predictor_vars_1);
sc_poor_std = autobinning(sc_poor_std);
sc_poor_std = fitmodel(sc_poor_std);
sc_poor_std = formatpoints(sc_poor_std, 'PointsOddsAndPDO', [600 2 20], 'Round', 'AllPoints');
sc_poor_std_points = displaypoints(sc_poor_std);

% Score for Poor vs Standard
[scores_poor_std, ~] = score(sc_poor_std, data(:, 1:end-1));

%% Standard vs Good Model
data_std_good = data(data.Score_Category ~= 'Poor', :); % Only Standard and Good
data_std_good.Standard_vs_Good = double(data_std_good.Score_Category == 'Good'); % Good = 1
data_std_good.Score_Category = [];

sc_std_good = creditscorecard(data_std_good, ...
    'IDVar', '', ...
    'ResponseVar', 'Standard_vs_Good', ...
    'GoodLabel', 1, ...
    'PredictorVars', predictor_vars_2);
sc_std_good = autobinning(sc_std_good);
sc_std_good = fitmodel(sc_std_good);
sc_std_good = formatpoints(sc_std_good, 'PointsOddsAndPDO', [750 2 20], 'Round', 'AllPoints');
sc_std_good_points = displaypoints(sc_std_good);

% Score for Standard vs Good
[scores_std_good, ~] = score(sc_std_good, data(:, 1:end-1));

%% Add 2 columns of scores from the 2 models
data.scores_poor_std = scores_poor_std;
data.scores_std_good = scores_std_good;

%% Distribution 1
colors = containers.Map(...
    {'Poor', 'Standard'}, ...
    {[0.8500 0.3250 0.0980], [0 0.4470 0.7410]});  % Red, Blue

% Get scores by category
classes = {'Poor', 'Standard'};

% Create the plot
figure;
hold on;

% Loop through the categories and plot the corresponding histograms
for i = 1:length(classes)
    class = classes{i};
    idx = data.Score_Category == class; % Filter data by category
    histogram(data.scores_poor_std(idx), 60, ... % Create histogram for each category
        'FaceColor', colors(class), ... % Assign color
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.6); % Transparency to see overlaps
end

% Add the threshold line at 600
line([600 600], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);

% Set up the plot
xlabel('Credit Score');
ylabel('Frequency');
title('Credit Score Distribution by Category');
legend(classes, 'Location', 'best');
grid on;
hold off;

%% Distribution 2
% Colors by category
colors = containers.Map(...
    {'Standard', 'Good'}, ...
    {[0 0.4470 0.7410], [0.4660 0.6740 0.1880]});  % Blue, Green

% Get scores by category
classes = {'Standard', 'Good'};

% Create the plot
figure;
hold on;

% Loop through the categories and plot the corresponding histograms
for i = 1:length(classes)
    class = classes{i};
    idx = data.Score_Category == class; % Filter data by category
    histogram(data.scores_std_good(idx), 60, ... % Create histogram for each category
        'FaceColor', colors(class), ... % Assign color
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.6); % Transparency to see overlaps
end

% Set up the plot
xlabel('Credit Score');
ylabel('Frequency');
title('Credit Score Distribution by Category');
legend(classes, 'Location', 'best');
grid on;
hold off;

%% Final Score 1
% Initialize points with NaNs
points = nan(height(data), 1);

% 1. Assign scores_poor_std values
idx_valid_poor_std = scores_poor_std >= 0 & scores_poor_std <= 600;
points(idx_valid_poor_std) = scores_poor_std(idx_valid_poor_std);

% 2. Assign values from the second model to the remaining rows
idx_remaining = ~idx_valid_poor_std; % All rows that did NOT meet the condition above
points(idx_remaining) = scores_std_good(idx_remaining);

% Save to the table
data.points = points;

%% Distribution 3
% Colors by category
colors = containers.Map(...
    {'Poor', 'Standard', 'Good'}, ...
    {[0.8500 0.3250 0.0980], [0 0.4470 0.7410], [0.4660 0.6740 0.1880]});  % Red, Blue, Green

% Get scores by category
classes = {'Poor', 'Standard', 'Good'};

% Create the plot
figure;
hold on;

% Loop through the categories and plot the corresponding histograms
for i = 1:length(classes)
    class = classes{i};
    idx = data.Score_Category == class; % Filter data by category
    histogram(data.points(idx), 60, ... % Create histogram for each category
        'FaceColor', colors(class), ... % Assign color
        'EdgeColor', 'none', ...
        'FaceAlpha', 0.6); % Transparency to see overlaps
end

% Add the threshold line at 600
line([600 600], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);
line([725 725], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 2);

% Set up the plot
xlabel('Credit Score');
ylabel('Frequency');
title('Credit Score Distribution by Category');
legend(classes, 'Location', 'best');
grid on;
hold off;

%% Classification
predicted_category = strings(height(data), 1);

predicted_category(data.points < 600) = "Poor";
predicted_category(data.points >= 600 & data.points <= 725) = "Standard";
predicted_category(data.points > 725) = "Good";

data.predicted_category = predicted_category;

% Accuracy
true_labels = categorical(data.Score_Category);
pred_labels = categorical(data.predicted_category);

accuracy = sum(true_labels == pred_labels) / numel(true_labels);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

% Confusion Matrix
figure;
confusionchart(true_labels, pred_labels);
title('Confusion Matrix: True vs Predicted Categories');
