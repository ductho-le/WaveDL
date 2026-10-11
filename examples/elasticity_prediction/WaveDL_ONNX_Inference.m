%% WaveDL ONNX Model Inference in MATLAB
% =========================================================================
% Demonstrates inference using a WaveDL-exported ONNX model in MATLAB.
% The model includes de-normalization, so outputs are in physical units.
%
% Requirements:
%   - MATLAB R2024a or later (for ONNX opset 17 support)
%   - Deep Learning Toolbox
%   - Deep Learning Toolbox Converter for ONNX Model Format
%
% Author: Ductho Le (ductho.le@outlook.com)
% =========================================================================

clear; clc;

%% ========================= CONFIGURATION ================================
EXAMPLE_DIR     = fileparts(mfilename('fullpath'));
ONNX_MODEL_PATH = fullfile(EXAMPLE_DIR, 'model.onnx');
TEST_DATA_PATH  = fullfile(EXAMPLE_DIR, 'Test_data_100.mat');
INPUT_SIZE      = [500, 500];  % Expected [H, W] for network initialization

%% ========================= LOAD MODEL ===================================
fprintf('📦 Loading ONNX model: %s\n', ONNX_MODEL_PATH);

% Keep importer warnings visible: unsupported operators need user attention.
net = importNetworkFromONNX(ONNX_MODEL_PATH);

exampleInput = dlarray(single(zeros([INPUT_SIZE, 1])), 'SSC');
if ~net.Initialized
    net = initialize(net, exampleInput);
end

fprintf('   ✔ Model loaded (%d layers)\n', numel(net.Layers));

%% ========================= LOAD DATA ====================================
fprintf('\n📂 Loading test data: %s\n', TEST_DATA_PATH);
data = load(TEST_DATA_PATH);

X_raw = data.input_test;
Y     = data.output_test;
validateattributes(Y, {'numeric'}, {'real', 'finite', 'nonempty', '2d'}, ...
    mfilename, 'output_test');
validateattributes(X_raw, {'numeric'}, {'real', 'finite', 'nonempty'}, ...
    mfilename, 'input_test');

fprintf('   Raw shape: [%s]\n', join(string(size(X_raw)), ' × '));

% Preprocess: handle MATLAB/Python dimension ordering differences
X = preprocessData(X_raw, size(Y, 1));
if size(X, 1) ~= size(Y, 1) || size(X, 2) ~= 1 || ...
        ~isequal([size(X, 3), size(X, 4)], INPUT_SIZE)
    error('WaveDL:InputShape', ...
        'Expected %d single-channel samples of size %d-by-%d.', ...
        size(Y, 1), INPUT_SIZE(1), INPUT_SIZE(2));
end

fprintf('   ✔ Loaded %d samples\n', size(X, 1));

%% ========================= INFERENCE ====================================
fprintf('\n🚀 Running inference...\n');

N = size(X, 1);
numOutputs = size(Y, 2);
predictions = zeros(N, numOutputs, 'single');

tic;
for i = 1:N
    sample = prepareSample(X, i);
    dlSample = dlarray(sample, 'SSC');
    dlPred = predict(net, dlSample);
    values = gather(extractdata(dlPred));
    if numel(values) ~= numOutputs || ~isreal(values) || any(~isfinite(values), 'all')
        error('WaveDL:InvalidPrediction', ...
            'Sample %d returned invalid predictions or an unexpected output count.', i);
    end
    predictions(i, :) = reshape(values, 1, []);
end
elapsedTime = toc;

fprintf('   ✔ Completed in %.2f s (%.1f ms/sample)\n', elapsedTime, elapsedTime/N*1000);

%% ========================= METRICS ======================================
fprintf('\n📊 Evaluation Metrics:\n');

errors = double(predictions) - double(Y);
metrics = struct(...
    'R2',   computeR2(Y, predictions), ...
    'RMSE', sqrt(mean(errors.^2, 'all')), ...
    'MAE',  mean(abs(errors), 'all'));

fprintf('   R² Score: %.6f\n', metrics.R2);
fprintf('   RMSE:     %.6f\n', metrics.RMSE);
fprintf('   MAE:      %.6f\n', metrics.MAE);

% Per-parameter breakdown
fprintf('\n   Per-Parameter MAE:\n');
for p = 1:numOutputs
    fprintf('     P%d: %.6f\n', p, mean(abs(errors(:, p))));
end

%% ========================= VISUALIZATION ================================
fprintf('\n📈 Generating plots...\n');

fig = figure('Position', [100, 100, 400*numOutputs, 350], 'Color', 'w');

for p = 1:numOutputs
    ax = subplot(1, numOutputs, p);

    scatter(Y(:, p), predictions(:, p), 25, 'filled', ...
        'MarkerFaceAlpha', 0.6, 'MarkerFaceColor', [0.2 0.4 0.8]);
    hold on;

    % Perfect prediction line
    lims = [min([Y(:, p); predictions(:, p)]), ...
        max([Y(:, p); predictions(:, p)])];
    plot(lims, lims, 'r--', 'LineWidth', 1.5);

    % Styling
    xlabel(sprintf('True P%d', p), 'FontWeight', 'bold');
    ylabel(sprintf('Predicted P%d', p), 'FontWeight', 'bold');
    title(sprintf('Parameter %d (R² = %.4f)', p, computeR2(Y(:,p), predictions(:,p))));
    grid on; axis equal; axis tight;
    ax.FontSize = 10;
    hold off;
end

sgtitle('WaveDL ONNX Model - Predictions vs Ground Truth', 'FontWeight', 'bold');

% Disable interactive toolbar before export to avoid warning
set(fig, 'ToolBar', 'none');
exportgraphics(fig, 'onnx_results.png', 'Resolution', 300);

fprintf('   ✔ Saved: onnx_results.png\n');
fprintf('\n✅ Inference Complete\n');

%% ========================= HELPER FUNCTIONS =============================

function X = preprocessData(X_raw, numSamples)
% Handle dimension transposition from Python/MATLAB format differences
dims = size(X_raw);

if dims(end) == numSamples && numel(dims) >= 2
    % Data is transposed: (H, W, N) or (H, W, C, N)
    if numel(dims) == 3
        X_raw = permute(X_raw, [3, 1, 2]);
    elseif numel(dims) == 4
        X_raw = permute(X_raw, [4, 3, 1, 2]);
    end
end

% Ensure 4D: (N, C, H, W)
if ndims(X_raw) == 3
    X_raw = reshape(X_raw, [size(X_raw, 1), 1, size(X_raw, 2), size(X_raw, 3)]);
end

X = X_raw;
end

function sample = prepareSample(X, idx)
% Extract and format single sample for dlarray
sample = squeeze(X(idx, :, :, :));

if ismatrix(sample)
    % 2D: transpose for row-major ordering, add channel
    sample = single(sample');
    sample = reshape(sample, [size(sample, 1), size(sample, 2), 1]);
else
    % 3D: permute (C, H, W) -> (H, W, C)
    sample = single(permute(sample, [2, 3, 1]));
end
end

function r2 = computeR2(y_true, y_pred)
% Match Python/sklearn's uniform mean of per-output R² values.
% Pooling columns around one global mean exaggerates R² when physical
% parameters have different offsets or units.
y_true = double(y_true);
y_pred = double(y_pred);
if ~isequal(size(y_true), size(y_pred))
    error('WaveDL:MetricShape', 'Prediction and target shapes must match.');
end
if size(y_true, 1) < 2
    r2 = NaN;  % R² is undefined for one sample.
    return;
end
ss_res = sum((y_true - y_pred).^2, 1);
ss_tot = sum((y_true - mean(y_true, 1)).^2, 1);
per_output = ones(1, size(y_true, 2));
nonconstant = ss_tot > 0;
per_output(nonconstant) = 1 - ss_res(nonconstant) ./ ss_tot(nonconstant);
per_output(~nonconstant & ss_res > 0) = 0;
r2 = mean(per_output);
end
