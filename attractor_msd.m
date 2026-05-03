%this code is to calculate the average position of two bacteria moving in a
%nutrient gradient that are secreting an attractive factor. The mean and
%mean square distance between them is also calculated. 
N_runs = 100; % runs
final_positions = zeros(N_runs, 2); % storing final positions

% parameters
L = 1; Nx = 200; x = linspace(0, L, Nx); 
T_total = 10; dt = 0.1; t_steps = 0:dt:T_total; 
v_speed = 0.05; lambda_0 = 1.0; chi_C = 50; chi_S = 30; 
D_C = 0.01; k = 0.5; D_S = 0.02; alpha = 1.0; gamma = 0.1; 

fprintf('Starting %d simulations. This may take a while...\n', N_runs);

% starting mc loop
for run = 1:N_runs
    
    % initial conditions
    C_current = max(0, 4-4 * (x - 0.5).^2); 
    S_current = zeros(1, Nx); 
    x_b = [0.5, 0.5];              
    direction = [1,-1];        
    C_memory = [interp1(x, C_current, x_b(1)), interp1(x, C_current, x_b(2))];
    S_memory = [interp1(x, S_current, x_b(1)), interp1(x, S_current, x_b(2))];

    % time loop
    for i = 2:length(t_steps)
        dC_dt = zeros(1, 2); dS_dt = zeros(1, 2);
        
        for j = 1:2
            dC_dt(j) = (interp1(x, C_current, x_b(j)) - C_memory(j)) / dt;
            dS_dt(j) = (interp1(x, S_current, x_b(j)) - S_memory(j)) / dt;
            C_memory(j) = interp1(x, C_current, x_b(j));
            S_memory(j) = interp1(x, S_current, x_b(j));
            %run and tumble
            lambda = lambda_0 * exp(-chi_C * dC_dt(j) - chi_S * dS_dt(j));
            if rand() < (lambda * dt); direction(j) = -direction(j); end
            
            x_b(j) = x_b(j) + (direction(j) * v_speed * dt);
            if x_b(j) > L; x_b(j) = L; direction(j) = -1;
            elseif x_b(j) < 0; x_b(j) = 0; direction(j) = 1; end
        end

        % updating pdes
        t_span = [0, dt/2, dt]; m = 0;
        sol = pdepe(m, @(x_space, t_time, u, dudx) pdefun(x_space, t_time, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt), ...
                       @(x_eval) icfun(x_eval, x, C_current, S_current), ...
                       @bcfun, x, t_span);
                   
        temp_C = sol(end, :, 1); C_current = temp_C(:)'; 
        temp_S = sol(end, :, 2); S_current = temp_S(:)'; 
    end
    
    % storing final positions
    final_positions(run, :) = x_b;
    fprintf('Run %d completed. Final Positions: [%.3f, %.3f]\n', run, x_b(1), x_b(2));
end

% output paramters
average_position = mean(final_positions, 1);
distances = abs(final_positions(:, 1) - final_positions(:, 2));
mean_distance = mean(distances);
mean_square_distance = mean(distances.^2);
fprintf('complete');
fprintf('Average Final Position of Bacterium 1: %.3f\n', average_position(1));
fprintf('Average Final Position of Bacterium 2: %.3f\n', average_position(2));
fprintf('Average Final Position of Bacterium 2: %.3f\n', average_position(2));
fprintf('Mean Distance Between Bacteria: %.4f\n', mean_distance);
fprintf('Mean Square Distance Between Bacteria: %.4f\n', mean_square_distance);
% helper functions
function [c, f, s] = pdefun(x, t, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt)
    c = [1; 1]; f = [D_C * dudx(1); D_S * dudx(2)];
    sigma = 0.02;
    delta_1 = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b(1))^2 / (2*sigma^2));
    delta_2 = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b(2))^2 / (2*sigma^2));
    s_C = -k * u(1) * (delta_1 + delta_2);
    s_S = (alpha * max(0, dC_dt(1)) * delta_1 + alpha * max(0, dC_dt(2)) * delta_2) - (gamma * u(2));
    s = [s_C; s_S];
end
function u0 = icfun(x_eval, x_grid, C_curr, S_curr)
    u0 = [interp1(x_grid, C_curr, x_eval); interp1(x_grid, S_curr, x_eval)];
end
function [pl, ql, pr, qr] = bcfun(~, ~, ~, ~, ~)
    pl = [0; 0]; ql = [1; 1]; pr = [0; 0]; qr = [1; 1]; 
end