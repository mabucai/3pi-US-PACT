function phase_sync_stack = dfc_phase(signals)
    % Get dimensions
    [nRegions, nTimePoints] = size(signals); 
    fs = 10; % Sampling frequency
    fprintf('Data Dim: %d regions x %d timepoints \n', nRegions, nTimePoints);
    
    %% 1. Instantaneous Phase Extraction (Hilbert Transform)
    fprintf('>>> Extracting instantaneous phases...\n');
    phase_data = zeros(size(signals));
    for i = 1:nRegions
        analytic_signal = hilbert(signals(i, :)); 
        phase_data(i, :) = angle(analytic_signal); % Range: [-π, π]
    end
    fprintf('>>> Phase extraction complete.\n');
    
    %% 2. Dynamic Phase Synchronization (Phase Matrix Stack)
    fprintf('>>> Computing dynamic synchronization matrices...\n');
    
    % Define step size for computation
    step = 1; 
    window_indices = 1:step:nTimePoints; 
    nWindows = length(window_indices);
    
    % Pre-allocate memory for 3D stack
    phase_sync_stack = zeros(nRegions, nRegions, nWindows);
    
    for t_idx = 1:nWindows
        current_sample = window_indices(t_idx);
        
        % Get current phase vector
        current_phases = phase_data(:, current_sample);
        
        % Compute synchronization matrix: cos(phi_i - phi_j)
        % Using identity: cos(a-b) = cos(a)cos(b) + sin(a)sin(b)
        cos_phi = cos(current_phases);
        sin_phi = sin(current_phases);
        
        % Efficient matrix multiplication
        sync_matrix = cos_phi * cos_phi' + sin_phi * sin_phi';
        
        phase_sync_stack(:, :, t_idx) = sync_matrix;
    end
    fprintf('>>> Computation complete. Total windows: %d.\n', nWindows);
end