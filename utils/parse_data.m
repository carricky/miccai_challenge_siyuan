base_dir = "/Users/wei/Desktop/2019_CNI_ValidationRelease-master";

sub_list = importdata(base_dir+'/SupportingInfo/phenotypic_validation.csv');
sub_list = sub_list.textdata(:, 1);
sub_list(1,:) = [];

num_sub = numel(sub_list);

%% AAL
num_parcel_aal = 116;
all_mats_aal = zeros(num_parcel_aal, num_parcel_aal, num_sub);
for i_sub = 1 : num_sub
    ts_temp = csvread(base_dir+'/Validation/'+sub_list{i_sub}+'/timeseries_aal.csv');
    all_mats_aal(:, :, i_sub) = atanh(corr(ts_temp'));
end

%% CC200
num_parcel_cc = 200;
all_mats_cc = zeros(num_parcel_cc, num_parcel_cc, num_sub);
for i_sub = 1 : num_sub
    ts_temp = csvread(base_dir+'/Validation/'+sub_list{i_sub}+'/timeseries_cc200.csv');
    all_mats_cc(:, :, i_sub) = atanh(corr(ts_temp'));
end

%% HO
num_parcel_ho = 112;
all_mats_ho = zeros(num_parcel_ho, num_parcel_ho, num_sub);
for i_sub = 1 : num_sub
    ts_temp = csvread(base_dir+'/Validation/'+sub_list{i_sub}+'/timeseries_ho.csv');
    all_mats_ho(:, :, i_sub) = atanh(corr(ts_temp'));
end

%% phenotypical
% 0-female 1-male
% 0-Control 1-AHDH 

phenotypicvaliation = importdata(base_dir+'/SupportingInfo/phenotypic_validation.csv');
phenotypicvaliation = phenotypicvaliation.textdata;
all_sex = zeros(num_sub, 1);
for i_sub = 1 : num_sub
    if char(phenotypicvaliation(i_sub+1, 2)) == 'F'
        all_sex(i_sub) = 0;
    else
        all_sex(i_sub) = 1;
    end
end

all_label = zeros(num_sub, 1);
for i_sub = 1 : num_sub
    if string(phenotypicvaliation(i_sub+1, 4)) == 'Control'
        all_label(i_sub) = 0;
    else
        all_label(i_sub) = 1;
    end
end

save('/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_aal.mat', "all_mats_aal");
save('/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_ho.mat', "all_mats_ho");
save('/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_mats_cc.mat', "all_mats_cc");
save('/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_label.mat', "all_label");
save('/Users/wei/Desktop/miccai_challenge_siyuan-master/validation/validation_sex.mat', "all_sex");