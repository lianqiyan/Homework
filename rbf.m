clear;clc;
raw_data = importdata('iris.csv');
feature = raw_data.data(:, 3:end)';
label = raw_data.data(:, 2)';
[feature, label] = shuffle(feature, label);
[tr_d, tr_la, te_d, te_la] = divide_data(feature, label, 0.8);
rbf_network = newrbe(tr_d, tr_la, 1000);
p = round(sim(rbf_network, te_d));
accruacy = sum(p==te_la)/length(te_la)

function [d_out, la_out] = shuffle(d_in, la_in)
    rand_index = randperm(size(d_in, 2));
    d_out = d_in(:, rand_index);
    la_out = la_in(:, rand_index);
end
function [train_data, train_label, test_data, test_label] = divide_data(data, label, percent)
    train_size = round(size(data, 2) * percent);
    train_data = data(:, 1:train_size);
    train_label = label(:, 1:train_size);
    test_data = data(:, train_size+1:end);
    test_label = label(:, train_size+1:end);
end