%this simulation calculates the movement of two bacteria due to a a
%nutrient gradient. they are non interacting. Final positions for both
%bacteria as well as their mean and mean square distances are calucalated.
N_runs = 100; %runs
final_positions = zeros(N_runs, 2); % final positions

% Parameters 
L = 1; Nx = 200; x = linspace(0, L, Nx); 
T_total = 10; dt = 0.1; t_steps = 0:dt:T_total; 
v_speed = 0.05; lambda_0 = 1.0; chi_C = 50; 
D_C = 0.01; k = 0.5; 

fprintf('Starting %d simulations (No Secretion). \n', N_runs);

%  mc loop
for run = 1:N_runs
    
    % initial conditions
    C_current = max(0, 4-4 * (x - 0.5).^2); 
    x_b = [0.5, 0.5];              
    direction = [1,-1];       
    C_memory = [interp1(x, C_current, x_b(1)), interp1(x, C_current, x_b(2))];

    %  time loop
    for i = 2:length(t_steps)
        dC_dt = zeros(1, 2);
        
        for j = 1:2
            % calculating the gradient and putting to memory
            dC_dt(j) = (interp1(x, C_current, x_b(j)) - C_memory(j)) / dt;
            C_memory(j) = interp1(x, C_current, x_b(j));
            
            % run and tumble (no attractant! only food)
            lambda = lambda_0 * exp(-chi_C * dC_dt(j));
            
            if rand() < (lambda * dt); direction(j) = -direction(j); end
            
           
            x_b(j) = x_b(j) + (direction(j) * v_speed * dt);
            if x_b(j) > L; x_b(j) = L; direction(j) = -1;
            elseif x_b(j) < 0; x_b(j) = 0; direction(j) = 1; end
        end

        % update pdes
        t_span = [0, dt/2, dt]; m = 0;
        
        sol = pdepe(m, @(x_space, t_time, u, dudx) pdefun(x_space, t_time, u, dudx, D_C, k, x_b), ...
                       @(x_eval) icfun(x_eval, x, C_current), ...
                       @bcfun, x, t_span);
                   
        temp_C = sol(end, :); 
        C_current = temp_C(:)'; 
    end
    
    % storing final values
    final_positions(run, :) = x_b;
    fprintf('Run %d completed. Final Positions: [%.3f, %.3f]\n', run, x_b(1), x_b(2));
end

% output paramerts
average_position = mean(final_positions, 1);
distances = abs(final_positions(:, 1) - final_positions(:, 2));
mean_distance = mean(distances);
mean_square_distance = mean(distances.^2);
fprintf('\n=== NO SECRETION SIMULATION COMPLETE ===\n');
fprintf('Average Final Position of Bacterium 1: %.3f\n', average_position(1));
fprintf('Average Final Position of Bacterium 2: %.3f\n', average_position(2));
fprintf('Mean Distance Between Bacteria: %.4f\n', mean_distance);
fprintf('Mean Square Distance Between Bacteria: %.4f\n', mean_square_distance);


% helper functions

function [c, f, s] = pdefun(x, t, u, dudx, D_C, k, x_b)
    c = 1; 
    f = D_C * dudx; 
    
    sigma = 0.02;
    delta_1 = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b(1))^2 / (2*sigma^2));
    delta_2 = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b(2))^2 / (2*sigma^2));
    
    s = -k * u * (delta_1 + delta_2);
end

function u0 = icfun(x_eval, x_grid, C_curr)
    u0 = interp1(x_grid, C_curr, x_eval); 
end

function [pl, ql, pr, qr] = bcfun(~, ~, ~, ~, ~)
    pl = 0; ql = 1; pr = 0; qr = 1; 
end