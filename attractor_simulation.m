%Parameters
L = 1;                  
Nx = 200;               
x = linspace(0, L, Nx); 

T_total = 10;           
dt = 0.1;               
t_steps = 0:dt:T_total; 

v_speed = 0.05;         
lambda_0 = 1.0;         
chi_C = 50;             
chi_S = 30;             

D_C = 0.01;             
k = 0.5;                
D_S = 0.02;             
alpha = 1.0;            
gamma = 0.1;            

%Initial conditions
C_current = exp(-100 * (x - 0.5).^2); 
S_current = zeros(1, Nx); 

x_b = 0.1;              
direction = 1;          
%calculating the gradient
C_memory = interp1(x, C_current, x_b);
S_memory = interp1(x, S_current, x_b);

C_history = zeros(length(t_steps), Nx);
S_history = zeros(length(t_steps), Nx);
xb_history = zeros(length(t_steps), 1);

C_history(1, :) = C_current;
S_history(1, :) = S_current;
xb_history(1) = x_b;


%Time loop
for i = 2:length(t_steps)
    
    % movement of the bacterium 
    dC_dt = (interp1(x, C_current, x_b) - C_memory) / dt;
    dS_dt = (interp1(x, S_current, x_b) - S_memory) / dt;
    
    C_memory = interp1(x, C_current, x_b);
    S_memory = interp1(x, S_current, x_b);
    %run and tumble stochastic
    lambda = lambda_0 * exp(-chi_C * dC_dt - chi_S * dS_dt); %we are currently only modelling for an attractant
    
    if rand() < (lambda * dt)
        direction = -direction;
    end
    
    x_b = x_b + (direction * v_speed * dt);
    
    if x_b > L
        x_b = L; direction = -1;
    elseif x_b < 0
        x_b = 0; direction = 1;
    end

    % updating pdes
    t_span = [0, dt/2, dt]; 
    m = 0;
    
    
    sol = pdepe(m, @(x_space, t_time, u, dudx) pdefun(x_space, t_time, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt), ...
                   @(x_eval) icfun(x_eval, x, C_current, S_current), ...
                   @bcfun, x, t_span);
               
    %recording the data and converting it into a a 1D vector
    temp_C = sol(end, :, 1);
    C_current = temp_C(:)'; 
    
    temp_S = sol(end, :, 2);
    S_current = temp_S(:)'; 
    
    C_history(i, :) = C_current;
    S_history(i, :) = S_current;
    xb_history(i) = x_b;
end


%plots
figure('Position', [100, 100, 1000, 400]);

%Nutrient
subplot(1,2,1);
surf(x, t_steps, C_history, 'EdgeColor', 'none');
hold on;
plot3(xb_history, t_steps, max(C_history(:))*ones(size(t_steps)), 'r-', 'LineWidth', 2);
title('Food Attractant (C) & Bacterium Path');
xlabel('Position'); ylabel('Time'); zlabel('Concentration');
colormap(gca, summer);
view(3);

% secretion
subplot(1,2,2);
surf(x, t_steps, S_history, 'EdgeColor', 'none');
title('Secreted Factor (S)');
xlabel('Position'); ylabel('Time'); zlabel('Concentration');
colormap(gca, winter);
view(3);


%helper functions
function [c, f, s] = pdefun(x, t, u, dudx, D_C, k, D_S, alpha, gamma, x_b, dC_dt)
    c = [1; 1]; 
    f = [D_C * dudx(1); 
         D_S * dudx(2)];
     
    sigma = 0.02;
    delta_approx = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - x_b)^2 / (2*sigma^2));
    
    s_C = -k * u(1) * delta_approx;
    
    secretion_rate = alpha * max(0, dC_dt);
    s_S = (secretion_rate * delta_approx) - (gamma * u(2));
    
    s = [s_C; s_S];
end

function u0 = icfun(x_eval, x_grid, C_curr, S_curr)
    % initial condition vectors
    u0 = [interp1(x_grid, C_curr, x_eval); 
          interp1(x_grid, S_curr, x_eval)];
end

function [pl, ql, pr, qr] = bcfun(xl, ul, xr, ur, t)
    pl = [0; 0]; ql = [1; 1]; 
    pr = [0; 0]; qr = [1; 1]; 
end
