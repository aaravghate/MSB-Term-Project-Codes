%here i have modelled the biphasic behaviour using a non hill function
%similar to a logistic plot.
N_runs = 100;           % Monte carlo cycles;
N_bacteria = 100;      % Number of bacteria

% Parameters global;
L = 1; Nx = 200; x = linspace(0, L, Nx); %x axis parameters
T_total = 5; dt = 0.1; t_steps = 0:dt:T_total; %time parameters
v_speed = 0.05; lambda_0 = 1.0; %bacterial velocity and power stochastic factor
D_C = 0.01; D_S = 0.02; gamma = 0.1; k = 0.2; %diffusion rates and eating rate

% To push the code to get biphasic superiority, these are the parameters i
% will be toggling. 
chi_C = 5;             
alpha_base = 20.0;     
chi_S_base = 50;       

% To simplify things i have tried to include my data into a text file'
log_file = fopen('Swarm_Simulation_Log.txt', 'a');

% writing hyperparameters
fprintf(log_file, '\n======================================================\n');
fprintf(log_file, 'NEW SIMULATION STARTED: %s\n', datestr(now));
fprintf(log_file, '======================================================\n');
fprintf(log_file, '--- Inputs / Hyperparameters ---\n');
fprintf(log_file, 'N_runs = %d | N_bacteria = %d\n', N_runs, N_bacteria);
fprintf(log_file, 'T_total = %.1f | dt = %.2f\n', T_total, dt);
fprintf(log_file, 'k (consumption) = %.3f | chi_C (food sensitivity) = %.1f\n', k, chi_C);
fprintf(log_file, 'alpha (secretion) = %.1f | chi_S (social sensitivity) = %.1f\n', alpha_base, chi_S_base);
fprintf(log_file, '------------------------------------------------------\n');

food_consumed_log = zeros(3, N_runs);
num_bins = 40; 
edges = linspace(0, L, num_bins + 1); 
bin_centers = edges(1:end-1) + diff(edges)/2; 
distribution_log = zeros(3, N_runs, num_bins);
case_names = {'1: No Secretion', '2: Pure Attractant', '3: Biphasic'};

%master loop for 3 cases
for scenario = 1:3
    
    fprintf('\n--- Starting Scenario %s ---\n', case_names{scenario});
    
    % the three strategy cases
    if scenario == 1
        alpha = 0; chi_S = 0; S_opt = Inf;
    elseif scenario == 2
        alpha = alpha_base; chi_S = chi_S_base; S_opt = Inf;
    elseif scenario == 3
        alpha = alpha_base; chi_S = chi_S_base; S_opt = 1;
    end
    
    % Monte Carlo Runs
    for run = 1:N_runs
        
        % Food at x=1, Bacteria starts moving from at x=0.1
        C_current = max(0, 5 * (x - 0.7).^2); 
        S_current = zeros(1, Nx);
        initial_total_food = trapz(x, C_current); 
        
        x_b = 0.1 * ones(1, N_bacteria); 
        direction = sign(randn(1, N_bacteria)); direction(direction == 0) = 1; 
        
        C_memory = zeros(1, N_bacteria);
        S_memory = zeros(1, N_bacteria);
        for j = 1:N_bacteria
            C_memory(j) = interp1(x, C_current, x_b(j));
            S_memory(j) = interp1(x, S_current, x_b(j));
        end
        
        % The Time Loop
        for i = 2:length(t_steps)
            dC_dt = zeros(1, N_bacteria);
            dS_dt = zeros(1, N_bacteria);
            
            for j = 1:N_bacteria
                current_C = interp1(x, C_current, x_b(j));
                current_S = interp1(x, S_current, x_b(j));
                
                dC_dt(j) = (current_C - C_memory(j)) / dt;
                dS_dt(j) = (current_S - S_memory(j)) / dt;
                
                C_memory(j) = current_C;
                S_memory(j) = current_S;
                
                chi_eff = chi_S * (1 - (current_S / S_opt));
                
                % run and tumble conditions
                sensory_signal = (chi_C * dC_dt(j)) + (chi_eff * dS_dt(j));
                if sensory_signal > 0
                    lambda = lambda_0 * exp(-sensory_signal);
                else
                    lambda = lambda_0;
                end
                
                if rand() < (lambda * dt); direction(j) = -direction(j); end
                
                x_b(j) = x_b(j) + (direction(j) * v_speed * dt);
                if x_b(j) > L; x_b(j) = L; direction(j) = -1;
                elseif x_b(j) < 0; x_b(j) = 0; direction(j) = 1; end
            end
            
            % updating the KS pdes
            t_span = [0, dt/2, dt]; m = 0;
            sol = pdepe(m, @(x_space, t_time, u, dudx) pdefun(x_space, t_time, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt), ...
                           @(x_eval) icfun(x_eval, x, C_current, S_current), ...
                           @bcfun, x, t_span);
                       
            temp_C = sol(end, :, 1); C_current = temp_C(:)'; 
            temp_S = sol(end, :, 2); S_current = temp_S(:)'; 
        end
        
        final_total_food = trapz(x, C_current);
        food_consumed_log(scenario, run) = initial_total_food - final_total_food;
        distribution_log(scenario, run, :) = histcounts(x_b, edges);
    end
end

%data logging stuff
fprintf('\n=== FINAL NUTRIENT CONSUMPTION STATISTICS ===\n');
fprintf(log_file, '--- Final Outputs (Nutrients Consumed) ---\n');
for scenario = 1:3
    avg_food = mean(food_consumed_log(scenario, :));
    max_food = max(food_consumed_log(scenario, :));
    min_food = min(food_consumed_log(scenario, :));
    
    % Print to command window for all 3 cases
    fprintf('%s\n', case_names{scenario});
    fprintf('  Average: %.4f units\n', avg_food);
    fprintf('  Maximum: %.4f units\n', max_food);
    fprintf('  Minimum: %.4f units\n\n', min_food);
    
    % Print to Log file
    fprintf(log_file, '%s\n', case_names{scenario});
    fprintf(log_file, '  Average: %.4f units\n', avg_food);
    fprintf(log_file, '  Maximum: %.4f units\n', max_food);
    fprintf(log_file, '  Minimum: %.4f units\n\n', min_food);
end

fclose(log_file);

%plots
figure('Position', [100, 100, 1200, 400], 'Name', 'Average Swarm Distribution');
colors = {[0.6 0.6 0.6], [0.8 0.3 0.3], [0.2 0.6 0.8]}; 
for scenario = 1:3
    avg_distribution = squeeze(mean(distribution_log(scenario, :, :), 2));
    subplot(1, 3, scenario);
    bar(bin_centers, avg_distribution, 'FaceColor', colors{scenario}, 'EdgeColor', 'none');
    xline(1.0, 'k--', 'LineWidth', 1.5, 'Label', 'Food Peak');
    title(case_names{scenario});
    xlabel('Position in Environment');
    if scenario == 1; ylabel('Average Number of Bacteria'); end
    xlim([0 L]);
    ylim([0, N_bacteria * 0.6]); 
    grid on;
end

%helper functions 
function [c, f, s] = pdefun(x, t, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt)
    c = [1; 1]; 
    f = [D_C * dudx(1); D_S * dudx(2)];
    sigma = 0.05; 
    delta_sum = sum((1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b).^2 / (2*sigma^2)));
    s_C = -k * max(0, u(1)) * delta_sum;
    secretion_sum = sum( (alpha .* max(0, dC_dt)) .* ((1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b).^2 / (2*sigma^2))) );
    s_S = secretion_sum - (gamma * u(2));
    s = [s_C; s_S];
end
function u0 = icfun(x_eval, x_grid, C_curr, S_curr)
    u0 = [interp1(x_grid, C_curr, x_eval); 
          interp1(x_grid, S_curr, x_eval)];
end
function [pl, ql, pr, qr] = bcfun(~, ~, ~, ~, ~)
    pl = [0; 0]; ql = [1; 1]; 
    pr = [0; 0]; qr = [1; 1]; 
end