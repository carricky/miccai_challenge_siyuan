clear;
%% hyperparameters
THRESHOLD = 0.001;
% pct = 0.01;
K_FOLD = 10;
base_dir = "/Users/siyuangao/Working_Space/fmri/code_siyuan/2019summer/miccai_challenge";
seed = 665;
%% load the label
load(base_dir+'/data/all_label.mat');
num_sub = numel(all_label);
%% set the random seed
rng(665);

%% global results record
all_accuracy = [];

%% AAL parcellation
% load the connectome
load(base_dir+'/data/all_mats_aal.mat');
[predict_label_aal, accuracy_aal] = cpm_svm(all_mats_aal, all_label, THRESHOLD, K_FOLD);

%% HO parcellation
% load the connectome
load(base_dir+'/data/all_mats_ho.mat');
[predict_label_ho, accuracy_ho] = cpm_svm(all_mats_ho, all_label, THRESHOLD, K_FOLD);

%% CC parcellation
% load the connectome
load(base_dir+'/data/all_mats_cc.mat');
[predict_label_cc, accuracy_cc] = cpm_svm(all_mats_cc, all_label, THRESHOLD, K_FOLD);

%% feed all the connectome all in
num_node_aal = size(all_mats_aal, 1);
num_edge_aal = num_node_aal * (num_node_aal - 1) / 2;
num_node_ho = size(all_mats_ho, 1);
num_edge_ho = num_node_ho * (num_node_ho - 1) / 2;
num_node_cc = size(all_mats_cc, 1);
num_edge_cc = num_node_cc * (num_node_cc - 1) / 2;
all_edges = zeros(num_edge_aal+num_edge_ho+num_edge_cc, num_sub);
for i_sub = 1 : num_sub
    all_edges(1:num_edge_aal, i_sub) = squareform(tril(all_mats_aal(:, :, i_sub), -1));
    all_edges(num_edge_aal+1:num_edge_aal+num_edge_ho, i_sub) = squareform(tril(all_mats_ho(:, :, i_sub), -1));
    all_edges(num_edge_aal+num_edge_ho+1:end, i_sub) = squareform(tril(all_mats_cc(:, :, i_sub), -1));
end
[predict_label_cat, accuracy_cat] = cpm_svm(all_edges, all_label, THRESHOLD, K_FOLD);

%% Majority vote
predict_label_mv = mode([predict_label_aal, predict_label_ho, predict_label_cc], 2);
accuracy_mv = 1 - sum(abs(predict_label_mv-all_label))/num_sub;


    
    
    