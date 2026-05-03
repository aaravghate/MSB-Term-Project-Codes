%this is the code to model the KS equations for bacterial and nutrient
%density

function KellerSegel1D()

    %first we define some global parameters
    global D_u D_v chi k;
    D_u = 0.1;  % Diffusion rate of bacteria 
    D_v = 0.1;   % Diffusion rate of the attractant 
    chi = 0.8;  % Chemotactic sensitivity 
    k = 1.0;     % consumption rate

    %spatial and temporal grids
    x = linspace(0, 1, 100); 
    t = linspace(0, 3, 60);  

    
    % m=0 is an input for pdepe pde solver to signify that we have boundary
    % conditions
    m = 0; 
    sol = pdepe(m, @keller_eqs, @initial_cond, @boundary_cond, x, t);

    %get solutions from matrix
    u = sol(:,:,1); % bacteria density
    v = sol(:,:,2); % attractant concentration

   %plots
    figure('Position', [100, 100, 800, 400]);
    
    % bacterial density surface plot
    subplot(1,2,1);
    surf(x, t, u, 'EdgeColor', 'none');
    title('Bacterial Density (u)');
    xlabel('Space (x)'); ylabel('Time (t)'); zlabel('Density');
    colormap(gca, winter); 
    view([30 40]);

    % nutrient density surface plot
    subplot(1,2,2);
    surf(x, t, v, 'EdgeColor', 'none');
    title('Attractant Concentration (v)');
    xlabel('Space (x)'); ylabel('Time (t)'); zlabel('Concentration');
    colormap(gca, summer); 
    view([30 40]);
end
% helper functions

% PDEs
function [c, f, s] = keller_eqs(x, t, state, dStateDx)
    global D_u D_v chi k;
    
    % state(1) is bacteria (u), state(2) is food (v)
    % dStateDx is the spatial derivative (du/dx and dv/dx)
    
    c = [1; 1]; % coefficients of time derivatives du/dt, dv/dt
    
    
    % f(1) is bacterial diffusion minus chemotactic flow
    % f(2) is food diffusion
    f = [D_u * dStateDx(1) - chi * state(1) * dStateDx(2); 
         D_v * dStateDx(2)];
     
    
    s = [0; 
        -k * state(1) * state(2)];
end

% initial conditions
function state0 = initial_cond(x)
    u0 = exp(-100 * (x - 0.2)^2);
    v0 = 3 * exp(-30 * (x - 0.8)^2);
    state0 = [u0; v0];
end

% C. boundary conditions
function [pl, ql, pr, qr] = boundary_cond(xl, ul, xr, ur, t)
    % these are zero flux (neumann conditions)
    % for pdepe, p + q*f = 0. Setting p=0 and q=1 means f=0 (flux is zero).
    pl = [0; 0];
    ql = [1; 1];
    pr = [0; 0];
    qr = [1; 1];
end