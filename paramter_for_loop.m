%here we are seeing how a single paramter varies the food consumption in
%case of the 3 strategies

N_runs = 5;           % cycles
N_bacteria = 100;     % bacteria number


% Here we are testing 5 different values for Food Sensitivity. I change
% this paramter in all the 8 cases
N_runs_val = [1,10,50,200,500,1000]; 

% Parameters global
L = 1; Nx = 200; x = linspace(0, L, Nx); 
T_total = 5; dt = 0.1; t_steps = 0:dt:T_total; 
v_speed = 0.05; lambda_0 = 1.0; 
D_C = 0.01; D_S = 0.02; gamma = 0.1; n_hill=2,k=1;

% Fixed Base Parameters for the other factors
alpha_base = 20.0;     
chi_S_base = 50;       
chi_C=20
%storing the data that will be plotted
avg_consumed_results = zeros(3, length(N_runs_val));
case_names = {'1: No Secretion', '2: Pure Attractant', '3: Biphasic'};

% how much time is left pop up
total_iters = length(N_runs_val) * 3;
current_iter = 0;
h_wait = waitbar(0, 'Starting Parameter Sweep...', 'Name', 'Sweep Progress');
tic;

%master loop
for idx = 1:length(N_runs_val)
    
    % choose the value of the parameter
    N_runs = N_runs_val(idx);
    
    
    for scenario = 1:3
        
        %define conditions for eachc of the 3 strategies
        if scenario == 1
            alpha = 0; chi_S = 0; S_opt = Inf;
        elseif scenario == 2
            alpha = alpha_base; chi_S = chi_S_base; S_opt = Inf;
        elseif scenario == 3
            alpha = alpha_base; chi_S = chi_S_base; S_opt = 1.0;
        end
        
        run_consumed = zeros(1, N_runs);
        
        % monte carlo run loop
        for run = 1:N_runs
            
            % Initial conditions
            C_current = max(0, 1 - 2 * (x - 0.7).^2); 
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
            
            % loop for time
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
                    
                    % linear case biphasic plot
                    hill_ratio = (current_S / S_opt)^n_hill;
                    chi_eff = chi_S * ( (1 - hill_ratio) / (1 + hill_ratio) );
                    
                    % run and tumble stochastic model
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
                
                % updating KS pdes
                t_span = [0, dt/2, dt]; m = 0;
                sol = pdepe(m, @(x_space, t_time, u, dudx) pdefun(x_space, t_time, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt), ...
                               @(x_eval) icfun(x_eval, x, C_current, S_current), ...
                               @bcfun, x, t_span);
                           
                temp_C = sol(end, :, 1); C_current = temp_C(:)'; 
                temp_S = sol(end, :, 2); S_current = temp_S(:)'; 
            end
            
            final_total_food = trapz(x, C_current);
            run_consumed(run) = initial_total_food - final_total_food;
        end
        
        % save average value for eventual plotting
        avg_consumed_results(scenario, idx) = mean(run_consumed);
        
        % timer Update
        current_iter = current_iter + 1;
        elapsed_time = toc;
        est_minutes_left = (elapsed_time / current_iter) * (total_iters - current_iter) / 60;
        waitbar(current_iter / total_iters, h_wait, sprintf('Progress: %d%% | Est. Time Left: %.1f min', round(100*current_iter/total_iters), est_minutes_left));
        
    end
end

% end the loading bar
if ishandle(h_wait); close(h_wait); end

%making plots

figure('Position', [100, 100, 800, 500], 'Name', 'Parameter Sweep Results');

% independent case in grey
plot(N_runs_val, avg_consumed_results(1, :), '-o', 'LineWidth', 2, 'Color', [0.6 0.6 0.6], 'MarkerSize', 8); 
hold on; 

% Pattractive factor in red
plot(N_runs_val, avg_consumed_results(2, :), '-s', 'LineWidth', 2, 'Color', [0.8 0.3 0.3], 'MarkerSize', 8);

% biphasic factor in blue
plot(N_runs_val, avg_consumed_results(3, :), '-^', 'LineWidth', 2, 'Color', [0.2 0.6 0.8], 'MarkerSize', 8);

%labels
title('Effect of Number of Runs on Swarm Foraging', 'FontSize', 14);
xlabel('Number of Runs', 'FontSize', 12);
ylabel('Average Nutrients Consumed', 'FontSize', 12);
legend(case_names, 'Location', 'Best', 'FontSize', 11);
grid on;

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