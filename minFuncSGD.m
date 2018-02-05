function [opttheta] = minFuncSGD(funObj, theta, data, labels, ...
    options)

%%======================================================================
% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
    'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
numSamples = length(labels); 
mom = 0.5;
momIncrease = 20;

velocity = zeros(size(theta));

options.test_interval_iterations = 50;
options.learning_rate_schedule = 'adagrad';
options.heuristics_learning_rate_schedule = 'power';
options.numFigures = 1;
options.test_results_save_file = 'a' ;

test_interval_iterations = options.test_interval_iterations;
learning_rate_schedule = options.learning_rate_schedule;
heuristics_learning_rate_schedule = options.heuristics_learning_rate_schedule;
if strfind(learning_rate_schedule, 'adadec')
    adadec_averaging_window_size = options.adadec_averaging_window_size;
    gradient_histories = {};
    gradient_histories_oldest = 1;
    running_sum_gradient_histories = zeros(0);
end

%%======================================================================
%% SGD loop
iteration = 0;
test_results.iterations = zeros(0);
test_results.accuracies = zeros(0);

sum_gradients = zeros(0);
learning_rate = alpha;

for e = 1:epochs
    
    % randomly permute indices of data for quick minibatch sampling
    rp = randperm(numSamples);
    ticEpoch = tic;
    for s=1 : minibatch : (numSamples - minibatch + 1)
        tic;
        iteration = iteration + 1;
        
        % increase momentum after momIncrease iterations
        if iteration == momIncrease
            mom = options.momentum;
        end;
        
        % get next randomly selected minibatch
        mb_data = data(:, :, :,rp(s : min(s + minibatch - 1, end)));
        mb_labels = labels(rp(s : min(s + minibatch - 1, end)));
        
        % evaluate the objective function on the next minibatch
        [cost(iteration), grad] = funObj(theta, mb_data, mb_labels);
        
        switch heuristics_learning_rate_schedule
            case 'power'
                alpha = alpha / (1 + iteration / 1e4);
            case 'exponential'
                alpha = alpha * pow(10,  -iteration / 1e4);
        end
        if strfind(learning_rate_schedule, 'adagrad')
            if isempty(sum_gradients)
                sum_gradients = grad .^ 2;
            else
                sum_gradients = sum_gradients + grad .^ 2;
            end
            learning_rate = alpha ./ sqrt(1 + sum_gradients);
        end
%         if strfind(learning_rate_schedule, 'adadec')
%             oldest = mod(gradient_histories_oldest - 1, adadec_averaging_window_size) + 1;
%             if gradient_histories_oldest > adadec_averaging_window_size
%                 running_sum_gradient_histories = running_sum_gradient_histories - gradient_histories{oldest};
%             end
%             gradient_histories{oldest} = grad .^ 2;
%             gradient_histories_oldest = gradient_histories_oldest + 1;
%             if isempty(running_sum_gradient_histories)
%                 running_sum_gradient_histories = gradient_histories{oldest};
%                 sum_gradients = running_sum_gradient_histories;
%             else
%                 running_sum_gradient_histories = running_sum_gradient_histories + gradient_histories{oldest};
%                 sum_gradients = options.adadec_forgetting_factor * ...
%                     sum_gradients + running_sum_gradient_histories;
%             end
%             learning_rate = alpha ./ sqrt(1 + sum_gradients);
%         end
        velocity = mom * velocity + learning_rate .* grad;
        theta = theta - velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n', e, iteration, cost(iteration));
        toc;

    end;
    fprintf('Epoch %d\n', e);
    toc(ticEpoch);
    if strcmp(heuristics_learning_rate_schedule, 'half_per_epoch')
        alpha = alpha / 2.0;
    end
end;

opttheta = theta;
figure()
plot(cost)
save(options.test_results_save_file, 'test_results');

end

