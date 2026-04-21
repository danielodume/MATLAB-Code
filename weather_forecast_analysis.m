clc; clear; close all;

%% =========================
% 1. Load Data
% =========================
filename = 'open-meteo-40.88N73.36W4m.csv';
if ~isfile(filename)
    error('Input file not found: %s', filename);
end

opts = detectImportOptions(filename);
opts.DataLines = [3 Inf];   % skip metadata rows
T = readtable(filename, opts);

if isempty(T) || width(T) == 0
    error('CSV file has no usable tabular data.');
end

vars = T.Properties.VariableNames;

%% =========================
% 2. Auto-Detect Columns
% =========================

% ---- Time column ----
time_idx = contains(vars, 'time', 'IgnoreCase', true);
time_cols = find(time_idx);
if isempty(time_cols)
    error('No time column found.');
end

% Use first matching time column
time_col = time_cols(1);
time_raw = T{:, time_col};

% Robust datetime parsing with fallback for common Open-Meteo format
try
    time = datetime(string(time_raw), 'TimeZone', 'UTC');
catch
    try
        time = datetime(string(time_raw), 'InputFormat', 'yyyy-MM-dd''T''HH:mm', 'TimeZone', 'UTC');
    catch ME
        error('Failed to parse time column "%s": %s', vars{time_col}, ME.message);
    end
end

if all(ismissing(time))
    error('Time column parsed to all missing datetimes.');
end

% ---- Temperature column ----
temp_idx = contains(vars, 'temperature', 'IgnoreCase', true);
temp_cols = find(temp_idx);
if isempty(temp_cols)
    error('No temperature column found.');
end

temp_candidates = vars(temp_cols);

% Prefer temperature_2m if available; otherwise first temperature column
is_2m = contains(temp_candidates, '2m', 'IgnoreCase', true);
if any(is_2m)
    best_col = temp_cols(find(is_2m, 1, 'first'));
else
    best_col = temp_cols(1);
end

temp = T{:, best_col};
if ~isnumeric(temp)
    temp = str2double(string(temp));
end
temp = double(temp(:));

% ---- Precipitation column (optional) ----
rain_idx = contains(vars, 'precip', 'IgnoreCase', true);
rain_cols = find(rain_idx);
hasRain = ~isempty(rain_cols);

if hasRain
    rain_col = rain_cols(1);
    rain = T{:, rain_col};
    if ~isnumeric(rain)
        rain = str2double(string(rain));
    end
    rain = double(rain(:));
end

%% =========================
% 3. Create Timetable (FIXED)
% =========================
TT = timetable(time(:), temp, 'VariableNames', {'temp'});

% Remove invalid rows before retime (prevents downstream bugs)
TT = rmmissing(TT);
TT = sortrows(TT);

if height(TT) < 10
    error('Not enough valid observations after cleaning to build model.');
end

% Daily average temperature
TT_daily = retime(TT, 'daily', 'mean');
TT_daily = rmmissing(TT_daily);

if height(TT_daily) < 10
    error('Daily aggregated series is too short after cleaning.');
end

% Extract safely
y = TT_daily.temp;
y = double(y(:));
time_daily = TT_daily.Properties.RowTimes;

% Final safety clean while preserving time alignment
valid_daily = ~isnan(y) & ~isinf(y);
y = y(valid_daily);
time_daily = time_daily(valid_daily);

if isempty(y)
    error('Data became empty after cleaning.');
end

%% =========================
% 4. Train-Test Split
% =========================
n = numel(y);
if n < 20
    error('Need at least 20 daily points for train/test modeling. Found %d.', n);
end

train_size = floor(0.8 * n);
train_size = max(10, train_size);
train_size = min(train_size, n - 1); % ensure at least one test point

y_train = y(1:train_size);
y_test  = y(train_size+1:end);

time_train = time_daily(1:train_size);
time_test  = time_daily(train_size+1:end);

%% =========================
% 5. ARIMA Model (STABLE)
% =========================
model = arima(2, 1, 2);
fitModel = estimate(model, y_train, 'Display', 'off');

%% =========================
% 6. Forecast
% =========================
[y_forecast, ~] = forecast(fitModel, numel(y_test), 'Y0', y_train);
y_forecast = y_forecast(:);

% Length safety alignment (defensive)
common_len = min(numel(y_test), numel(y_forecast));
y_test_eval = y_test(1:common_len);
y_forecast_eval = y_forecast(1:common_len);
time_test_eval = time_test(1:common_len);

%% =========================
% 7. Error Metrics
% =========================
MAE  = mean(abs(y_test_eval - y_forecast_eval));
RMSE = sqrt(mean((y_test_eval - y_forecast_eval).^2));

fprintf('MAE: %.2f\n', MAE);
fprintf('RMSE: %.2f\n', RMSE);

%% =========================
% 8. Plot Forecast vs Actual
% =========================
figure;
plot(time_train, y_train, 'b'); hold on;
plot(time_test_eval, y_test_eval, 'k');
plot(time_test_eval, y_forecast_eval, 'r');
legend('Train', 'Actual', 'Forecast', 'Location', 'best');
title('Forecast vs Actual Temperature');
xlabel('Time'); ylabel('Temperature');
grid on;

%% =========================
% 9. Anomaly Detection (FIXED)
% =========================
residuals = y_test_eval - y_forecast_eval;
resid_std = std(residuals, 'omitnan');

if resid_std == 0 || isnan(resid_std)
    z = zeros(size(residuals));
else
    z = residuals ./ resid_std;   % proper normalization
end

threshold = 2.5;
anomaly_idx = abs(z) > threshold;

anomaly_times = time_test_eval(anomaly_idx);
anomaly_values = y_test_eval(anomaly_idx);

fprintf('Number of anomalies: %d\n', sum(anomaly_idx));

figure;
plot(time_test_eval, y_test_eval, 'b'); hold on;
scatter(anomaly_times, anomaly_values, 40, 'r', 'filled');
legend('Actual', 'Anomalies', 'Location', 'best');
title('Anomaly Detection');
xlabel('Time'); ylabel('Temperature');
grid on;

%% =========================
% 10. Future Forecast (30 days)
% =========================
futureSteps = 30;
[y_future, ~] = forecast(fitModel, futureSteps, 'Y0', y);
future_time = time_daily(end) + days(1:futureSteps);

figure;
plot(time_daily, y, 'k'); hold on;
plot(future_time, y_future, 'm');
legend('Historical', 'Future Forecast', 'Location', 'best');
title('Future Temperature Prediction');
xlabel('Time'); ylabel('Temperature');
grid on;

%% =========================
% 11. Extreme Days
% =========================

% Hottest & Coldest
[max_temp, idx_max] = max(y);
[min_temp, idx_min] = min(y);

fprintf('\nHottest Day: %s (%.2f)\n', string(time_daily(idx_max)), max_temp);
fprintf('Coldest Day: %s (%.2f)\n', string(time_daily(idx_min)), min_temp);

% Rainiest Day
if hasRain
    TT_rain = timetable(time(:), rain, 'VariableNames', {'rain'});
    TT_rain = rmmissing(TT_rain);
    TT_rain = sortrows(TT_rain);

    if height(TT_rain) > 0
        TT_rain_daily = retime(TT_rain, 'daily', 'sum');
        TT_rain_daily = rmmissing(TT_rain_daily);

        if height(TT_rain_daily) > 0
            [max_rain, idx_rain] = max(TT_rain_daily.rain);
            fprintf('Rainiest Day: %s (%.2f)\n', ...
                string(TT_rain_daily.Properties.RowTimes(idx_rain)), max_rain);
        else
            fprintf('Precipitation data became empty after daily aggregation.\n');
        end
    else
        fprintf('Precipitation column exists but has no valid data.\n');
    end
else
    fprintf('No precipitation column detected.\n');
end
