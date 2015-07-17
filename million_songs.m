
% Load data
song_data = csvread('notebook/data/cs190/millionsong.txt');
song_data = song_data(randperm(size(song_data, 1)), :);
all_x = song_data(:, 2:end);
all_y = song_data(:, 1);
all_size = size(all_x, 1);
clear song_data;

new_x = [];
for i = 1:all_size
    quadratic_features = all_x(i, :)' * all_x(i, :);
    new_x(i, :) = quadratic_features(:)';
end

all_x = new_x;
clear new_x;


all_x = [ones(all_size, 1), all_x];
feature_size = size(all_x, 2);

% Normalize data
%all_x = all_x - repmat(mean(all_x), all_size, 1);
%all_x = all_x - repmat(sqrt(var(all_x)), all_size, 1);
all_y = all_y - mean(all_y);


% Partition data
train_size = round(all_size * 0.8);
val_size = round(all_size * 0.1);
test_size = size(train_size, 1) - train_size - val_size;

train_x = all_x(1:train_size, :);
train_y = all_y(1:train_size, :);

val_x = all_x(train_size + 1:train_size + val_size, :);
val_y = all_y(train_size + 1:train_size + val_size, :);

test_x = all_x(train_size + val_size + 1:end, :);
test_y = all_y(train_size + val_size + 1:end, :);



% Train

best_w = 0;
best_loss = inf;

lambdas = sqrt(10) .^ [-15:15];

for lambda = lambdas
    w = pinv((train_x' * train_x) + lambda * eye(feature_size)) * train_x' * train_y;
    loss_val = norm(val_x * w - val_y);
    fprintf('lambda = %f, loss = %f\n', lambda, loss_val);
    if loss_val < best_loss
        best_w = w;
        best_loss = loss_val;
    end
end

test_loss = mean(abs(test_x * w - test_y));
test_bias = mean(test_x * w - test_y);
test_stddiv = sqrt(var(test_x * w - test_y));
fprintf('Test mean difference: %f\n', test_loss);
fprintf('Test bias: %f\n', test_bias);
fprintf('Test standard div: %f\n', test_stddiv);

