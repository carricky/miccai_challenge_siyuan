function [predict_label, accuracy] = cpm_svm(all_mats, all_label, threshold, k_fold)
    
    % convert to edges if all_mats is connectome
    if size(all_mats, 1) == size(all_mats, 2)
        num_sub = size(all_mats, 3);
        num_node = size(all_mats, 1);
        num_edge = num_node * (num_node - 1) / 2;
        all_edges = zeros(num_edge, num_sub);
        for i_sub = 1 : num_sub
            all_edges(:, i_sub) = squareform(tril(all_mats(:, :, i_sub), -1));
        end
    else
        num_sub = size(all_mats, 2);
        all_edges = all_mats;
    end
    
    indices = crossvalind('Kfold', num_sub, k_fold);
    predict_label = zeros(num_sub, 1);
    
    for i_fold = 1 : k_fold
        % prepare training and testing data
        test_ind = (indices == i_fold);
        train_mats = all_edges;
        train_mats(:, test_ind) = [];
        train_label = all_label;
        train_label(test_ind, :) = [];
        
        % feature selection
        [r_mat,p_mat] = corr(train_mats', train_label);
        p_mat(r_mat < 0) = 1;
        % p_mat_sorted = sort(p_mat);
        % thresh = p_mat_sorted(round(nEdge *  pct));
        pos_edges = find(p_mat <= threshold);
        train_feature = train_mats(pos_edges, :)';
        
        % SVM model fitting
        mdl = fitcsvm(train_feature, train_label');
        
        % SVM model prediction
        predict_label(test_ind) = predict(mdl, all_edges(pos_edges, test_ind)');
    end
    
    accuracy = 1 - sum(abs(predict_label-all_label))/num_sub;
end

