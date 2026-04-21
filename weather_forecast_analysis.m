clc; clear; close all;

%% =========================
% 1. Load Data
% =========================
filename = 'open-meteo-40.88N73.36W4m.csv';

% Allow selecting a CSV if the default file is not found.
if ~isfile(filename)
    [selectedFile, selectedPath] = uigetfile({'*.csv', 'CSV Files (*.csv)'}, ...
        'Select weather CSV file');
    if isequal(selectedFile, 0)
        error(['CSV file not found and no file selected. ', ...
            'Place your CSV near this script or select it in the dialog.']);
    end
    filename = fullfile(selectedPath, selectedFile);
end

opts = detectImportOptions(filename);
opts.DataLines = [3 Inf];   % Skip metadata rows
T = readtable(filename, opts);

vars = T.Properties.VariableNames;

%% =========================
% 2. Auto-Detect Columns
% =========================

% ---- Time column ----
time_idx = contains(vars, 'time', 'IgnoreCase', true);
if ~any(time_idx)
    error('No time column found.');
end

time_raw = T{:, find(time_idx, 1, 'first')};
if isdatetime(time_raw)
    time = time_raw;
    if isempty(time.TimeZone)
        time.TimeZone = 'UTC';
    else
        time = datetime(time, 'TimeZone', 'UTC');
    end
else
    time_str = strtrim(string(time_raw));
    parseFormats = [ ...
        "yyyy-MM-dd'T'HH:mm", ...
        "yyyy-MM-dd'T'HH:mm:ss", ...
        "yyyy-MM-dd HH:mm", ...
        "yyyy-MM-dd HH:mm:ss", ...
        "yyyy-MM-dd'T'HH:mmXXX", ...
        "yyyy-MM-dd'T'HH:mm:ssXXX", ...
        "yyyy-MM-dd'T'HH:mm'Z'", ...
        "yyyy-MM-dd'T'HH:mm:ss'Z'" ...
    ];

    time = NaT(size(time_str));
    parsed = false;

    for iFmt = 1:numel(parseFormats)
        try
            tTry = datetime(time_str, 'InputFormat', parseFormats(iFmt), 'TimeZone', 'UTC');
            if all(~isnat(tTry))
                time = tTry;
                parsed = true;
                break;
            end
        catch
            % Try next format.
        end
    end

    if ~parsed
        % Last resort: auto parser.
        time = datetime(time_str, 'TimeZone', 'UTC');
        if any(isnat(time))
            error('Unable to parse the time column. Check CSV time format.');
        end
    end
end

% ---- Temperature column ----
temp_idx = contains(vars, 'temperature', 'IgnoreCase', true);
if ~any(temp_idx)
    error('No temperature column found.');
end

temp_candidates = vars(temp_idx);
temp_candidate_idx = find(temp_idx);

% Prefer temperature_2m if available
[~, best] = max(contains(temp_candidates, '2m', 'IgnoreCase', true));
temp = T{:, temp_candidate_idx(best)};

% ---- Precipitation column (optional) ----
rain_idx = contains(vars, 'precip', 'IgnoreCase', true);
hasRain = any(rain_idx);

if hasRain
    rain = T{:, find(rain_idx, 1, 'first')};
end

%% =========================
% 3. Create Timetable
% =========================
TT = timetable(time, temp);

% Remove missing BEFORE retime
TT = rmmissing(TT);

% Daily average temperature
TT_daily = retime(TT, 'daily', 'mean');

% Extract safely
y = double(TT_daily{:, 1});
time_daily = TT_daily.Properties.RowTimes;

valid = ~isnan(y) & ~isinf(y);
y = y(valid);
time_daily = time_daily(valid);

if isempty(y)
    error('Data became empty after cleaning.');
end

%% =========================
% 4. Train-Test Split
% =========================
n = length(y);
train_size = floor(0.8 * n);

if train_size < 5 || train_size >= n
    error('Not enough data after cleaning for a stable train/test split.');
end

y_train = y(1:train_size);
y_test  = y(train_size + 1:end);

time_train = time_daily(1:train_size);
time_test  = time_daily(train_size + 1:end);

%% =========================
% 5. ARIMA Model
% =========================
model = arima(2, 1, 2);
fitModel = estimate(model, y_train, 'Display', 'off');

%% =========================
% 6. Forecast
% =========================
[y_forecast, ~] = forecast(fitModel, length(y_test), 'Y0', y_train);
y_forecast = y_forecast(:);

%% =========================
% 7. Error Metrics
% =========================
MAE  = mean(abs(y_test - y_forecast));
RMSE = sqrt(mean((y_test - y_forecast) .^ 2));

fprintf('MAE: %.2f\n', MAE);
fprintf('RMSE: %.2f\n', RMSE);

%% =========================
% 8. Plot Forecast vs Actual
% =========================
figure;
plot(time_train, y_train, 'b'); hold on;
plot(time_test, y_test, 'k');
plot(time_test, y_forecast, 'r');
legend('Train', 'Actual', 'Forecast');
title('Forecast vs Actual Temperature');
xlabel('Time'); ylabel('Temperature');
grid on;

%% =========================
% 9. Anomaly Detection
% =========================
residuals = y_test - y_forecast;
res_std = std(residuals);
if res_std == 0
    z = zeros(size(residuals));
else
    z = residuals ./ res_std;
end

threshold = 2.5;
anomaly_idx = abs(z) > threshold;

anomaly_times = time_test(anomaly_idx);
anomaly_values = y_test(anomaly_idx);

fprintf('Number of anomalies: %d\n', sum(anomaly_idx));

figure;
plot(time_test, y_test); hold on;
scatter(anomaly_times, anomaly_values, 40, 'filled');
legend('Actual', 'Anomalies');
title('Anomaly Detection');
grid on;

%% =========================
% 10. Future Forecast (30 days)
% =========================
futureSteps = 30;
[y_future, ~] = forecast(fitModel, futureSteps, 'Y0', y);
future_time = time_daily(end) + days(1:futureSteps);

figure;
plot(time_daily, y); hold on;
plot(future_time, y_future);
legend('Historical', 'Future Forecast');
title('Future Temperature Prediction');
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
    TT_rain = timetable(time, rain);
    TT_rain = rmmissing(TT_rain);
    TT_rain_daily = retime(TT_rain, 'daily', 'sum');

    [max_rain, idx_rain] = max(TT_rain_daily{:, 1});
    fprintf('Rainiest Day: %s (%.2f)\n', ...
        string(TT_rain_daily.Properties.RowTimes(idx_rain)), max_rain);
else
    fprintf('No precipitation column detected.\n');
end
