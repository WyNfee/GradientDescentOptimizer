%This Script is targeting on using ReLu neuron and BGD to do a digit recognition
%using Nesterov Accelerated Gradient (NAG)

%It is a full connection neuron network with only 1 hidden layer with 100 hidden neurons and 1 output
%the final determine function is using softmax classifier

%the architecture:
%input: 400 neurons
%hidden layer: 100 neurons, full connection, activation: ReLu
%output layer: 10 neuron,  full connection, activation: softmax

%clear every thing
clear; close all; clc;

%set up system execuate environment
addpath(genpath('Utils'))


%load the data
load('data_MNIST.mat');

%change the variable name of the input data
%we assume the data has been normalized, check the data!
%in practise, we need to do batch normalization
g_input_data = X;

%set up architecture parameters
g_layer_one_neuron_amount = 400;
g_layer_two_neuron_amount = 100;
g_layer_three_neuron_amount = 10;

%build the input answer matrix
g_input_answer_amount = size(y,1);
g_input_answer = zeros(g_input_answer_amount, g_layer_three_neuron_amount);
for i = 1 : g_input_answer_amount
    g_input_answer(i,y(i)) = 1;
end

%init the weight for layer one
g_layer_one_input_size = g_layer_one_neuron_amount + 1;
g_layer_one_weight = function_XavierInitialization_For_ReLu(g_layer_one_input_size, g_layer_two_neuron_amount);

%init the weight for layer two
g_layer_two_input_size = g_layer_two_neuron_amount + 1;
g_layer_two_weight = function_XavierInitialization_For_ReLu(g_layer_two_input_size, g_layer_three_neuron_amount);

%pack the weigth together to compute the weight
g_packed_weight = [g_layer_one_weight(:); g_layer_two_weight(:)]; 

%provide the layer one and layer two size
g_layer_one_size = size(g_layer_one_weight);
g_layer_two_size = size(g_layer_two_weight);

%a hyper parameter of regularization param, close the regularization here
g_h_reularization_param = 0;

%assign the hyperparameter learning rate
g_h_learning_rate = 0.01;

%assgin the hyperparameter, stochasic data size
g_h_stochasitic_data_size = 50;

%a storage for weigth in sgd 
t_packedweightforSGD  = g_packed_weight;

%define the iteration time, 100 * 100 times
t_iteration_time = 10000;

%a cost data storage for further ploting
t_record_cost_data = zeros(t_iteration_time/100, 1);

%a gate whether we do gradient descent
g_do_gradient_descent = false;

%nag updater param
t_nag_param = 0.95;

%nag updater
t_nag_updater = zeros(size(t_packedweightforSGD));

if(g_do_gradient_descent == true)
    %do gradient descent
    for i = 1: t_iteration_time
        
        %this will make the data always within [1:g_input_answer_amount]
        t_rand_picked_data_index = floor(rand(1, g_h_stochasitic_data_size) * (g_input_answer_amount - 1) ) + 1;
        %generate stocastic data from dataset
        t_rand_picked_data = g_input_data(t_rand_picked_data_index(1:end), :);
        %provide the coresponding answer data as well
        t_rand_picked_answer = g_input_answer(t_rand_picked_data_index(1:end), :);

        %find the cost and gradient
        t_nag_updater = t_nag_param * t_nag_updater;
        t_nag_adjust_weight = t_packedweightforSGD - t_nag_updater;
        
        [t_cost_param, t_gradient_param] = function_NN_Learning_Algorithm(t_nag_adjust_weight,t_rand_picked_data, t_rand_picked_answer, g_layer_one_size, g_layer_two_size, g_h_reularization_param);
        
        %simply compute the gradient
        t_nag_updater = t_nag_updater + g_h_learning_rate * t_gradient_param;
        t_packedweightforSGD = t_packedweightforSGD - t_nag_updater;
        
        
        %output the cost to console every 100 iterate, so that we know
        %whether it is working, and the progress so far
        if( rem(i, 100) == 0)
            %record the cost for plot
            t_record_cost_data(i/100) = t_cost_param;
            fprintf('update cost, current cost %.6f,\n',t_cost_param);
        end
    end
    
    %Save the data so taht we can use it to generate the plot
    %because running the learning alogorithm is taking time
    %we save the data can saving the time running it again
    s = input('save the loss data, close the gate and run a gain using the loss data can create plot?, y to save:','s');
    
    if(s == 'y')
        save('data_nag.mat', 't_packedweightforSGD', 't_record_cost_data');
        fprintf('Data Saved\n');
    else
        fprintf('No Data Saved\n');
    end
    
else
    %we can do plot and check the performance without running the learning
    %algorithm again
    
    %plot the gradient descent
    load('data_nag.mat');
    
    %prepare the data for plot
    t_cost_data_size = length(t_record_cost_data);
    t_cost_data_nag = zeros(t_cost_data_size, 2);
    
    for i = 1 : t_cost_data_size
        
        t_cost_data_nag(i, 1) = i;
        t_cost_data_nag(i, 2) = t_record_cost_data(i);
        
    end
    
    %plot the data
    plot(t_cost_data_nag(:,1), t_cost_data_nag(:,2),'--');
    
    %Save the data to compare with other learning algorithm
    s = input('save the plot data?, y to save:','s');
    if(s == 'y')
        save('data_lost_nag.mat', 't_cost_data_nag');
        fprintf('Data Saved\n');
    else
         fprintf('No Data Saved\n');
    end
end



%unpack the parameters again, no matter what happens above, we can still
%get our descent gradient weight
t_layer_one_weight_size = g_layer_one_size(1) * g_layer_one_size(2);
t_layer_one_weight = reshape(t_packedweightforSGD ( 1 : t_layer_one_weight_size), g_layer_one_size);
t_layer_two_weight_size = t_layer_one_weight_size+1;
t_layer_two_weight = reshape(t_packedweightforSGD(t_layer_two_weight_size : end), g_layer_two_size);

%now do the prediction and plot
t_test = true;

while(t_test)
    %whether we want to preview some result
    s = input('Press enter to display a image, q to exit:','s');
    if(s == 'q')
        t_test = false;
        break;
    end
    
    %random pick a data from training set
    t_picked_image_index = floor(rand(1) * g_input_answer_amount);
    
    %prepare the data to predict
    t_picked_Image_data = g_input_data(t_picked_image_index,:);
    t_image_data_size = size(t_picked_Image_data);
    t_image_edge = sqrt(t_image_data_size(2));
    t_image_matrix = reshape(t_picked_Image_data, t_image_edge,t_image_edge);
    
    %do prediction
    t_picked_Image_data = [1, t_picked_Image_data];
    t_layer_one_data = function_ReLu(t_picked_Image_data * t_layer_one_weight');
    t_layer_one_data = [1, t_layer_one_data];
    t_prediction = function_Softmax(t_layer_one_data * t_layer_two_weight');
    
    [t_probability, t_index]  =max(t_prediction);
    
    %adjust the output, no 0 comlum, so we use 10 columns for 0, thus, we
    %need adjust it back
    if(t_index == 10)
        t_index = 0;
    end
    
    %print the result
    fprintf('predict %d, probability is %1.4f\n',t_index,t_probability)
    
    colormap(gray);
    imagesc(t_image_matrix);
    axis image off;
        
end

%Now compute the accuracy we have made (in training cases)
%This is just for this sample, in practise, DO NOT do something like that,
%the input data could be super huge

%prepare the data for all training case
t_helper_for_evaluate = ones(g_input_answer_amount, 1);
t_input_data_for_evaluate = [t_helper_for_evaluate ,g_input_data];
t_layer_one_data = function_ReLu(t_input_data_for_evaluate * t_layer_one_weight');
t_layer_one_data = [t_helper_for_evaluate,t_layer_one_data];
t_predictions_matrix = function_Softmax(t_layer_one_data * t_layer_two_weight');
t_predictions_matrix = t_predictions_matrix';
%we use the max probability in k-means output, in practise, sometimes using
%top 5 output, this cases is so small, using top 5 is silly
[t_probability, t_prediction] = max(t_predictions_matrix);


t_right_prediction_count = sum(t_prediction' == y);
t_accuracy = t_right_prediction_count / g_input_answer_amount;

fprintf('prediction accurracy %1.6f\n',t_accuracy)

