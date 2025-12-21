%% WaveDL ONNX Model Inference in MATLAB
% =============================================
% This script demonstrates how to use a WaveDL-exported ONNX model
% for inference in MATLAB. The model includes de-normalization by default,
% so outputs are in physical/original-scale units.
%
% Requirements:
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Converter for ONNX Model Format
%
% Ductho Le (ductho.le@outlook.com)
% =============================================

%% Configuration
ONNX_MODEL_PATH = 'model.onnx';           % Path to your exported ONNX model
TEST_DATA_PATH = 'Test_data_100.mat';    % Path to your test data

%% 1. Load the ONNX Model (Updated Function)
fprintf('Loading ONNX model: %s\n', ONNX_MODEL_PATH);

% Use the newer recommended function
net = importNetworkFromONNX(ONNX_MODEL_PATH);

fprintf('Model loaded successfully!\n');
fprintf('Network layers: %d\n', numel(net.Layers));

%% 2. Load Test Data
fprintf('\nLoading test data: %s\n', TEST_DATA_PATH);
data = load(TEST_DATA_PATH);

% Extract input and output
input_data = data.input_test;      % [N, 1, 500, 500]
ground_truth = data.output_test;   % [N, 3]

fprintf('Input shape: [%s]\n', num2str(size(input_data)));
fprintf('Ground truth shape: [%s]\n', num2str(size(ground_truth)));

%% 3. Single Sample Inference
fprintf('\n--- Single Sample Inference ---\n');

sample_idx = 1;

% Reshape: Python [N, C, H, W] -> MATLAB [H, W, C]
sample = squeeze(input_data(sample_idx, :, :, :));  % [1, 500, 500] -> [500, 500]
sample = reshape(sample, [size(sample, 1), size(sample, 2), 1]);  % [500, 500, 1]
sample = single(sample);

% Create dlarray (SSC = Spatial, Spatial, Channel)
dlSample = dlarray(sample, 'SSC');

% Run inference
tic;
dlPredictions = predict(net, dlSample);
elapsed = toc;

predictions = extractdata(dlPredictions);

fprintf('Prediction: [%s]\n', num2str(predictions', '%.4f '));
fprintf('Ground truth: [%s]\n', num2str(ground_truth(sample_idx, :), '%.4f '));
fprintf('Inference time: %.3f ms\n', elapsed * 1000);

%% 4. Batch Inference (All Samples)
fprintf('\n--- Batch Inference ---\n');

N = size(input_data, 1);
num_outputs = numel(predictions);
all_predictions = zeros(N, num_outputs);

tic;
for i = 1:N
    sample = squeeze(input_data(i, :, :, :));
    sample = reshape(sample, [size(sample, 1), size(sample, 2), 1]);
    sample = single(sample);
    
    dlSample = dlarray(sample, 'SSC');
    dlPred = predict(net, dlSample);
    all_predictions(i, :) = extractdata(dlPred)';
    
    if mod(i, 100) == 0
        fprintf('  Processed %d/%d samples\n', i, N);
    end
end
total_time = toc;

fprintf('Total: %.2f s (%.2f ms/sample)\n', total_time, total_time/N*1000);

%% 5. Compute Metrics
fprintf('\n--- Evaluation Metrics ---\n');

errors = all_predictions - ground_truth;
mae = mean(abs(errors), 'all');
rmse = sqrt(mean(errors.^2, 'all'));

% R² score
ss_res = sum((ground_truth - all_predictions).^2, 'all');
ss_tot = sum((ground_truth - mean(ground_truth, 'all')).^2, 'all');
r2 = 1 - (ss_res / ss_tot);

fprintf('R² Score: %.6f\n', r2);
fprintf('RMSE:     %.6f\n', rmse);
fprintf('MAE:      %.6f\n', mae);

% Per-parameter MAE
for p = 1:size(ground_truth, 2)
    fprintf('  MAE P%d: %.6f\n', p, mean(abs(errors(:, p))));
end

%% 6. Visualization
figure('Position', [100, 100, 1200, 400]);

for p = 1:size(ground_truth, 2)
    subplot(1, size(ground_truth, 2), p);
    scatter(ground_truth(:, p), all_predictions(:, p), 10, 'filled', 'MarkerFaceAlpha', 0.5);
    hold on;
    
    lims = [min([ground_truth(:, p); all_predictions(:, p)]), ...
            max([ground_truth(:, p); all_predictions(:, p)])];
    plot(lims, lims, 'r--', 'LineWidth', 1.5);
    
    xlabel(sprintf('True P%d', p));
    ylabel(sprintf('Predicted P%d', p));
    
    ss_res_p = sum((ground_truth(:, p) - all_predictions(:, p)).^2);
    ss_tot_p = sum((ground_truth(:, p) - mean(ground_truth(:, p))).^2);
    r2_p = 1 - (ss_res_p / ss_tot_p);
    
    title(sprintf('Parameter %d (R² = %.4f)', p, r2_p));
    grid on; axis equal;
end

sgtitle('WaveDL ONNX Model - Predictions vs Ground Truth');
saveas(gcf, 'onnx_results.png');

fprintf('\n=== Inference Complete ===\n');