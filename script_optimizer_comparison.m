%clear Everything
clear; close all; clc;


%load the cost data
load('data_lost_no_optimizer.mat');
load('data_lost_momentum.mat');
load('data_lost_nag.mat');

%plot them out
plot(t_cost_data_no_optimizer(:,1), t_cost_data_no_optimizer(:, 2), 'g-',...
        t_cost_data_momentum(:,1), t_cost_data_momentum(:,2),'r-',...
        t_cost_data_nag(:,1), t_cost_data_nag(:,2),'b-'...
        ...
        );
legend('No Optimizer', 'Momentum');
