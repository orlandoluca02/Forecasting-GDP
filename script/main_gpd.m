clc; clear all; close all;

% PROJECT SCRIPT: GDFM vs Benchmarks (AR, VAR, PCA, RW)
% OOS evaluation: rolling-origin (many splits) with fixed rolling window
%
% Notes:
% - Change BASE_DIR / DATA_PATH to match your machine.
% - DATA is assumed to be numeric (readmatrix) with the first column removed.
% - Target variable is selected by target_idx (e.g., GDP column).
% - VAR benchmark uses DATA-DRIVEN variable selection:
%     VAR variables = [target_idx, top K drivers from correlation ranking]


% USER SETTINGS (change these paths on your machine)
full_path_script = mfilename('fullpath'); 
[script_dir, ~, ~] = fileparts(full_path_script);
%%

if isempty(script_dir) || contains(script_dir, 'private/var')
    BASE_DIR = '/Users/lucaorlando/Desktop/projects/gdp_forecasting';
else
    BASE_DIR = fileparts(script_dir);  
end
GDFM_DIR  = fullfile(BASE_DIR, 'tools', 'gdfm copia');           % consider renaming to 'gdfm'
ABC_DIR   = fullfile(BASE_DIR, 'tools', 'ABC_crit_fast');
DATA_PATH = fullfile(BASE_DIR, 'data', 'processed_data copia.xlsx');


% PATH SETUP + DATA LOADING

restoredefaultpath; rehash toolboxcache;

addpath(GDFM_DIR);
addpath(ABC_DIR);

disp('gdfm_onesided resolved to:'); which gdfm_onesided -all
disp('ABC_crit resolved to:');     which ABC_crit -all
disp('Data path:');                disp(DATA_PATH)

X_raw_matrix = readmatrix(DATA_PATH);
X_raw = X_raw_matrix(:, 2:end);
X_raw(end,:) = [];                 % Clean last row (if incomplete)
[T, n] = size(X_raw);

%% -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  

% SELECT NUMBER OF STATIC FACTORS VIA ABC (Alessi et al., 2010)

kmax  = 20;                  % maximum number of factors to consider
nbck  = floor(n/10);         % number of sub-blocks (default suggestion)
cmax  = 3;                   % max penalty constant
graph = 0;                   % 1 to show ABC plot

fprintf('\n--- ABC CRITERION (Alessi et al., 2010) ---\n');
[rhat1, rhat2] = ABC_crit(X_raw, kmax, nbck, cmax, graph);
fprintf('ABC suggested static factors (large window) : rhat1 = %d\n', rhat1);
fprintf('ABC suggested static factors (small window) : rhat2 = %d\n', rhat2);

% Choose one 
r_opt = rhat1;
fprintf('Using r_opt = %d static factors in the GDFM.\n', r_opt);

%% -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  

% PARAMETERS

target_idx = 1;       % TARGET COLUMN INDEX ( GDP )  <-- CHANGE IF YOU WANT TO CHOOSE ANOTHER TARGET VARIABLE
q_opt = 4;            % Dynamic factors
p = 1;                % VAR lag order (keep small in OOS)
steps = 8;            % Forecast horizon (h-step ahead)

% FIXED rolling window length (quarters in this dataset). 
window = 80;

% VAR data-driven selection: number of extra variables (besides target)
K_var = 2;            % VAR dimension = 1 + K_var  

% Basic feasibility checks
assert(window < T, 'window must be smaller than T.');
assert(steps >= 1 && steps < T, 'Invalid steps.');
assert(target_idx >= 1 && target_idx <= n, 'Invalid target_idx.');

% IN-SAMPLE: RUN GDFM + DRIVER ANALYSIS

m = floor(sqrt(T));
h = m;

[chi, ~, X_std, ~, ~, ~, ~] = gdfm_onesided(X_raw, q_opt, r_opt, m, h, steps, 1);

fprintf('\n--- IN-SAMPLE ANALYSIS (TARGET IDX = %d) ---\n', target_idx);
fprintf('Common variance explained by the GDFM (var(chi_target)): %.2f%%\n', var(chi(:, target_idx))*100);

% Drivers based on correlation between the target common component and the standardized panel
rho_drivers = corr(chi(:, target_idx), X_std, 'Rows', 'complete');
[~, idx_rho] = sort(abs(rho_drivers), 'descend');

fprintf('\n--- TOP DRIVERS (by |corr| with target common component) ---\n');
for k = 1:min(10, n)
    fprintf('%2d) Variable %3d | corr = %+0.4f\n', k, idx_rho(k), rho_drivers(idx_rho(k)));
end


% VAR variables = [target_idx, top K_var drivers]

K_var = 4;  % number of extra variables besides the target

candidate = idx_rho(idx_rho ~= target_idx);
candidate = candidate(:);                 % <-- FIX: force column vector

takeK = min(K_var, numel(candidate));

if takeK == 0
    var_indices = target_idx;
else
    var_indices = [target_idx; candidate(1:takeK)];
end

var_indices = unique(var_indices, 'stable');
fprintf('\nVAR variable indices (target first):\n');
disp(var_indices');

%% -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
%  OUT-OF-SAMPLE: MANY ORIGINS + FIXED ROLLING WINDOW

% Origins are t_end = window, window+1, ..., T-steps
t0     = window;
t_last = T - steps;
n_test = t_last - t0 + 1;
assert(n_test >= 1, 'Not enough data for this window/steps.');

actuals   = zeros(n_test, 1);
pred_GDFM = zeros(n_test, 1);
pred_AR1  = zeros(n_test, 1);
pred_RW   = zeros(n_test, 1);
pred_PCA  = zeros(n_test, 1);
pred_VAR  = zeros(n_test, 1);

fprintf('\nStarting OOS MANY-ORIGINS evaluation (window=%d, steps=%d, n_test=%d)...\n', ...
        window, steps, n_test);

j = 0;
for t_end = t0:t_last
    j = j + 1;

    % Training window indices
    t_start = t_end - window + 1;

    % Forecast target index (in full sample)
    t_forecast = t_end + steps;

    X_train = X_raw(t_start:t_end, :);
    m_curr  = floor(sqrt(size(X_train, 1)));

    % Actual value in original scale
    actuals(j) = X_raw(t_forecast, target_idx);

    % --- A. GDFM Forecast (forecast of the common component used as signal) ---
    [~, ~, ~, ~, ~, f_chi] = gdfm_onesided(X_train, q_opt, r_opt, m_curr, m_curr, steps, 0);
    pred_GDFM(j) = f_chi(target_idx);

    % --- B. AR(1) Forecast (h-step projection) ---
    y_ar = X_train(2:end, target_idx);
    x_ar = X_train(1:end-1, target_idx);
    b_ar = (x_ar' * x_ar) \ (x_ar' * y_ar);
    pred_AR1(j) = X_train(end, target_idx) * (b_ar^steps);

    % --- C. Random Walk (RW) ---
    pred_RW(j) = X_train(end, target_idx);

    % --- D. Static PCA (with back-transform to original scale) ---
    mu = mean(X_train, 'omitnan');
    sd = std(X_train,  'omitnan');
    X_s = (X_train - mu) ./ sd;

    [~, score] = pca(X_s);
    f_static = score(:, 1:r_opt);

    b_static = f_static(1:end-steps, :) \ X_s(steps+1:end, target_idx);
    pred_PCA_std = f_static(end, :) * b_static;
    pred_PCA(j)  = pred_PCA_std * sd(target_idx) + mu(target_idx);

    % --- E. VAR(p) Benchmark (iterated for h steps) ---
    % IMPORTANT: var_indices(1) must be target_idx so pred_VAR is for target
    data_v = X_train(:, var_indices);
    T_eff = size(data_v, 1) - p;

    Y_v = data_v(p+1:end, :);
    Z_v = ones(T_eff, 1);
    for l = 1:p
        Z_v = [Z_v, data_v(p+1-l : end-l, :)];
    end
    B_v = (Z_v' * Z_v) \ (Z_v' * Y_v);

    curr_data = data_v;
    for s = 1:steps
        last_obs_v = 1;
        for l = 0:p-1
            last_obs_v = [last_obs_v, curr_data(end-l, :)];
        end
        next_step = last_obs_v * B_v;
        curr_data = [curr_data; next_step];
    end
    pred_VAR(j) = curr_data(end, 1);

    % Optional progress print (every ~10%)
    if mod(j, max(1, floor(n_test/10))) == 0
        fprintf('Progress: %d/%d OOS forecasts computed...\n', j, n_test);
    end
end

%% -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
% PERFORMANCE METRICS 

rmse_gdfm = sqrt(mean((actuals - pred_GDFM).^2));
rmse_ar1  = sqrt(mean((actuals - pred_AR1).^2));
rmse_var  = sqrt(mean((actuals - pred_VAR).^2));
rmse_pca  = sqrt(mean((actuals - pred_PCA).^2));
rmse_rw   = sqrt(mean((actuals - pred_RW).^2));

cum_err_GDFM = cumsum((actuals - pred_GDFM).^2);
cum_err_AR1  = cumsum((actuals - pred_AR1).^2);
cum_err_VAR  = cumsum((actuals - pred_VAR).^2);
cum_err_PCA  = cumsum((actuals - pred_PCA).^2);
cum_err_RW   = cumsum((actuals - pred_RW).^2);

fprintf('\n====================================================\n');
fprintf('   FINAL FORECAST PERFORMANCE (HORIZON: %d)\n', steps);
fprintf('====================================================\n');
fprintf('MODEL        |   RMSE       |  RELATIVE (vs AR1)\n');
fprintf('----------------------------------------------------\n');
fprintf('GDFM (q=%d)    |   %.6f   |   %.2f\n', q_opt, rmse_gdfm, rmse_gdfm/rmse_ar1);
fprintf('AR(1)         |   %.6f   |   1.00\n', rmse_ar1);
fprintf('VAR(%d)        |   %.6f   |   %.2f\n', p, rmse_var, rmse_var/rmse_ar1);
fprintf('Static PCA    |   %.6f   |   %.2f\n', rmse_pca, rmse_pca/rmse_ar1);
fprintf('Random Walk   |   %.6f   |   %.2f\n', rmse_rw, rmse_rw/rmse_ar1);
fprintf('----------------------------------------------------\n');

%% -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  - 
% PLOT: Target vs GDFM common component (STANDARDIZED) 

y_obs_std = X_std(:, target_idx);   % observed target in standardized units (z-score)
y_common  = chi(:, target_idx);     % GDFM common component (same standardized units)

% "Common share" (R^2-like) in standardized space
common_share = var(y_common) / var(y_obs_std);

figure('Name', 'Target vs GDFM Common Component (Standardized)');
plot(y_obs_std, 'LineWidth', 1, 'Color', [0 0 0]); hold on;
plot(y_common,  'LineWidth', 2.5, 'Color', [0.3 0.6 0.9]);   
grid on;

title(sprintf('Target vs GDFM Common (standardized), idx=%d | Common share = %.1f%%', ...
      target_idx, 100*common_share));
xlabel('Quarters');
ylabel('Standardized units (z-score)');
legend('Observed target (z-score)', 'GDFM common (z-score)', 'Location', 'best');


% PLOT : Target (z-score) vs AR(1) one-step-ahead fitted values (z-score)
% Requires: X_std already available

y_obs_std = X_std(:, target_idx);              % observed target (z-score)

% AR(1) fitted (one-step ahead, in-sample): yhat_t = b * y_{t-1}
y_lag = y_obs_std(1:end-1);
y_now = y_obs_std(2:end);
b_ar  = (y_lag' * y_lag) \ (y_lag' * y_now);

y_ar_fit = [NaN; b_ar * y_lag];                % align to same length as y_obs_std

% R^2-like share (how much of var is captured by fitted values)
common_share_ar = var(y_ar_fit(2:end), 'omitnan') / var(y_obs_std(2:end), 'omitnan');

figure('Name', 'Target vs AR(1) fitted (Standardized)');
plot(y_obs_std, 'LineWidth', 1, 'Color', [0 0 0]); hold on;
plot(y_ar_fit,  'LineWidth', 2.5, 'Color', [0.85 0.35 0.35]);   % soft red
grid on;

title(sprintf('Target vs AR(1) fitted (standardized), idx=%d | Share = %.1f%%', ...
      target_idx, 100*common_share_ar));
xlabel('Quarters');
ylabel('Standardized units (z-score)');
legend('Observed target (z-score)', 'AR(1) fitted (z-score)', 'Location', 'best');

% PLOT : Target (z-score) vs Static PCA fitted values (z-score)
% Uses r_opt PCs; fits y_t on PCs in-sample

% Recompute PCA scores on the standardized panel (same as your benchmark)
% (If you prefer PCA on a specific window, change X_s accordingly)
X_s = X_std;                                    % already standardized by gdfm_onesided
[~, score] = pca(X_s);

f_static = score(:, 1:r_opt);                   % first r_opt PCs
b_pca    = f_static \ y_obs_std;                % in-sample fit: y ≈ F*b
y_pca_fit = f_static * b_pca;                   % fitted values (z-score)

common_share_pca = var(y_pca_fit, 'omitnan') / var(y_obs_std, 'omitnan');

figure('Name', 'Target vs Static PCA fitted (Standardized)');
plot(y_obs_std,  'LineWidth', 1,   'Color', [0 0 0]); hold on;
plot(y_pca_fit,  'LineWidth', 2.5, 'Color', [0.65 0.35 0.85]);   % soft purple
grid on;

title(sprintf('Target vs Static PCA fitted (standardized), idx=%d | Share = %.1f%%', ...
      target_idx, 100*common_share_pca));
xlabel('Quarters');
ylabel('Standardized units (z-score)');
legend('Observed target (z-score)', 'Static PCA fitted (z-score)', 'Location', 'best');
