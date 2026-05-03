%code to plot phase diagram for various parameters for both the hill
%function and the linear function. The implicit solver for the finite
%difference method has been used to make computation faster (and its still
%incredibly slow). 

%sweep parameters: these can be changed based on what i want on the phase
%plot. H is relative height of the peaks, d is distance of peaks from the
%center (x=0.5) and w steepness of the parabola representing food. 
d_vals = linspace(0.05, 0.45, 50);    % Distance of peaks from center
h_ratio = linspace(0.2, 1.2, 50);
w = 40;                                
%Simulation parameters for the actual code
N_runs = 200;            % no of runs
N_bacteria = 200;       % Swarm size
L = 1; Nx = 200; x = linspace(0, L, Nx); 
T_total = 5; dt = 0.1; t_steps = 0:dt:T_total; 
v_speed = 0.05; lambda_0 = 1.0; 
D_C = 0.01; D_S = 0.02; gamma = 0.1; k = 0.5; 

%strategy dependent parameters
chi_C = 5;             
alpha_base = 30.0;     
chi_S_base = 60;       
n_hill = 2;             

%storing the final output data for the phase plot
% Depth 1 = Linear Model, Depth 2 = Hill Model so both can be plotted with
% the same code. 
winning_matrix = zeros(length(d_vals), length(h_ratio), 2); 
model_names = {'Linear Formulation', 'Hill Function Formulation'};

%finite difference method computation
fprintf('Precomputing Sparse FDM Matrices...\n');
dx = L / (Nx - 1);
e = ones(Nx, 1);
L_mat = spdiags([e -2*e e], [-1 0 1], Nx, Nx);
L_mat(1,2) = 2;         % Zero-flux left wall bound cond
L_mat(Nx,Nx-1) = 2;     % Zero-flux right wall bound cond

% Implicit Left-Hand Side Matrices (Ident - dt * diffusion * laplacian)
A_C = speye(Nx) - (D_C * dt / dx^2) * L_mat;
A_S = speye(Nx) - (D_S * dt / dx^2) * L_mat;

%master loops
total_iters = length(d_vals) * length(h_ratio) * 2;
current_iter = 0;
h_wait = waitbar(0, 'Starting Sweep...', 'Name', 'Overnight Progress');
tic;

% loop to switch between linear and hill models
for model_type = 1:2
    fprintf('\n=== STARTING SWEEP FOR %s ===\n', upper(model_names{model_type}));
    
    % sweep distance loop
    for d_idx = 1:length(d_vals)
        d = d_vals(d_idx);
        x1 = 0.5 - d;
        x2 = 0.5 + d;
        
        % ratio of heights loop
        for h_idx = 1:length(h_ratio)
            h2 = h_ratio(h_idx); 
            h1 = 1.0; 
            
            avg_consumed = zeros(1, 3);
            
            % loop to compute for all 3 survival strategies
            for scenario = 1:3
                if scenario == 1; alpha = 0; chi_S = 0; S_opt = Inf; end
                if scenario == 2; alpha = alpha_base; chi_S = chi_S_base; S_opt = Inf; end
                if scenario == 3; alpha = alpha_base; chi_S = chi_S_base; S_opt = 1.0; end
                
                run_consumed = zeros(1, N_runs);
                
                % monte carlo run and tumble sumiluation (1D)
                for run = 1:N_runs
                    
                    % Initialize Environment
                    C_current = max(0, h1 - w*(x - x1).^2) + max(0, h2 - w*(x - x2).^2);
                    S_current = zeros(1, Nx);
                    initial_total_food = trapz(x, C_current); 
                    
                    % Initialize bacteria
                    x_b = 0.5 * ones(1, N_bacteria); 
                    direction = sign(randn(1, N_bacteria)); direction(direction == 0) = 1; 
                    
                    C_memory = interp1(x, C_current, x_b, 'linear', 'extrap');
                    S_memory = interp1(x, S_current, x_b, 'linear', 'extrap');
                    
                    % loop over time
                    for i = 2:length(t_steps)
                        
                        % measuring the gradient of both the chemicals and
                        % storing
                        current_C = interp1(x, C_current, x_b, 'linear', 'extrap');
                        current_S = interp1(x, S_current, x_b, 'linear', 'extrap');
                        
                        dC_dt = (current_C - C_memory) / dt;
                        dS_dt = (current_S - S_memory) / dt;
                        
                        C_memory = current_C;
                        S_memory = current_S;
                        
                        % cases for linear and hill models
                        if S_opt == Inf
                            chi_eff = chi_S; 
                        else
                            if model_type == 1
                                % LINEAR 
                                chi_eff = chi_S * (1 - (current_S / S_opt));
                            else
                                % HILL 
                                hill_ratio = (current_S ./ S_opt).^n_hill;
                                chi_eff = chi_S .* ( (1 - hill_ratio) ./ (1 + hill_ratio) );
                            end
                        end
                        
                        % Run and tumble stochastic code
                        sensory_signal = (chi_C .* dC_dt) + (chi_eff .* dS_dt);
                        lambda = lambda_0 * ones(1, N_bacteria); 
                        better_idx = sensory_signal > 0;         
                        lambda(better_idx) = lambda_0 .* exp(-sensory_signal(better_idx)); 
                        
                        tumble_idx = rand(1, N_bacteria) < (lambda .* dt);
                        direction(tumble_idx) = -direction(tumble_idx);
                        
                        x_b = x_b + (direction .* v_speed .* dt);
                        x_b(x_b > L) = L; direction(x_b > L) = -1;
                        x_b(x_b < 0) = 0; direction(x_b < 0) = 1;
                        
                        % code to update the fdm matrices
                        sigma = 0.05;
                        [X_grid, X_b_grid] = ndgrid(x, x_b);
                        Gaussians = (1 / sqrt(2*pi*sigma^2)) * exp(-(X_grid - X_b_grid).^2 / (2*sigma^2));
                        
                        delta_sum = sum(Gaussians, 2)'; 
                        secretion_weights = alpha .* max(0, dC_dt);
                        secretion_sum = (Gaussians * secretion_weights')'; 
                        
                        React_C = -k .* max(0, C_current) .* delta_sum;
                        React_S = secretion_sum - (gamma .* S_current);
                        
                        C_current = (A_C \ (C_current + dt * React_C)')';
                        S_current = (A_S \ (S_current + dt * React_S)')';
                    end
                    
                    final_total_food = trapz(x, C_current);
                    run_consumed(run) = initial_total_food - final_total_food;
                end
                avg_consumed(scenario) = mean(run_consumed);
            end
            
            % Saving data for the winner for each pair of paramters
            [~, winner_idx] = max(avg_consumed);
            winning_matrix(d_idx, h_idx, model_type) = winner_idx;
            
            
            current_iter = current_iter + 1;
            elapsed = toc;
            est_mins = (elapsed / current_iter) * (total_iters - current_iter) / 60;
            
            if mod(current_iter, 50) == 0 %code to find out how much time is left (doesn't help impatient people like me :))
                waitbar(current_iter/total_iters, h_wait, ...
                    sprintf('Computing... %d%% | Est. Left: %.1f hrs', ...
                    round(100*current_iter/total_iters), est_mins/60));
            end
            
        end
    end
end
if ishandle(h_wait); close(h_wait); end
fprintf('\nSimulation Complete! Generating Figures...\n');
% savind raw data in case of a crash
save('Overnight_PhaseDiagram_Data.mat', 'winning_matrix', 'd_vals', 'h_ratio');
%visualising the plots
figure('Position', [50, 100, 1200, 500], 'Name', 'Phase Diagrams: Linear vs Hill');
custom_map = [0.6 0.6 0.6; 0.8 0.3 0.3; 0.2 0.6 0.8];

for p = 1:2
    subplot(1, 2, p);
    imagesc(h_ratio, d_vals, winning_matrix(:,:,p));
    set(gca, 'YDir', 'normal'); 
    colormap(custom_map);
    clim([0.5, 3.5]); 
    
    cb = colorbar;
    cb.Ticks = [1, 2, 3];
    cb.TickLabels = {'No Secretion', 'Pure Attractant', 'Biphasic'};
    cb.FontSize = 10;
    
    title(model_names{p}, 'FontSize', 14);
    xlabel('Height of Right Peak', 'FontSize', 12);
    if p == 1; ylabel('Distance from Center (d)', 'FontSize', 12); end
end