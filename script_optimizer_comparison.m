%clear Everything
clear; close all; clc;


%load the cost data
load('data_lost_no_optimizer.mat');
load('data_lost_momentum.mat');
load('data_lost_nag.mat');
load('data_lost_adagrad.mat');
load('data_lost_adadelta.mat');

%plot them out
plot(t_cost_data_no_optimizer(:,1), t_cost_data_no_optimizer(:, 2), 'g-',...
        t_cost_data_momentum(:,1), t_cost_data_momentum(:,2),'r-',...
        t_cost_data_nag(:,1), t_cost_data_nag(:,2),'b-',...
        t_cost_data_adagrad(:,1), t_cost_data_adagrad(:,2),'c-',...
        t_cost_data_adadelta(:,1), t_cost_data_adadelta(:,2),'k-'...
        );
legend('No Optimizer', 'Momentum','NAG','Adagrad','Adadelta');
