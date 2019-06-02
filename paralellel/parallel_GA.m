clear;clc;
% data = xlsread('FT06.xlsx');
% data = xlsread('FT10.xlsx');
% data = xlsread('LA01.xlsx');
data = xlsread('FT35.xlsx');  %1888
order = data(:, 1:2:end);
order = order + 1;
time = data(:, 2:2:end);

tic
cross_rate = 0.95;
mutate_rate = 0.05;
record = zeros(100,4);

for i = 1:100
    if i>3
        mutate_rate = 0.6;
    end
    parpool(4)
    spmd
     if i == 1
        [pfit, psol] = JSP_GA(time, order, cross_rate, mutate_rate, 500);
     else
        [pfit, psol] = JSP_GA(time, order, cross_rate, mutate_rate, 500, pop);
     end
    end
    pop = zeros(500, size(psol{1}, 2));
    disp(['The result of No.', num2str(i), ' communication']);
    record(i, 1) = min(pfit{1});
    record(i, 1)
    record(i, 2) = min(pfit{2});
    record(i, 2)
    record(i, 3) = min(pfit{3});
    record(i, 3)
    record(i, 4) = min(pfit{4});
    record(i, 4)
    for k = 1: 4
        pop((k-1)*125+1:k*125, :) = get_sol(pfit{k}, psol{k});
    end
    delete(gcp)
end

[end_time, ~] = cal_fit(pop, time, order);
disp('the final best result is ')
min(end_time)
toc



function  better = get_sol(fit, sol)
  [~, index] = sort(fit, 'ascend');
  better = sol(index(1:125), :);   
end

function [end_time, fit] = cal_fit(pop, time, order)
    end_time = zeros(size(pop, 1), 1);
    for i = 1: size(pop, 1)
        job_time = zeros(1, size(order, 1)); % record the end time of the job
        mach_time = zeros(1, size(order, 2)); % record the rest time of the machine
        for j = 1: size(pop, 2)
            job = pop(i, j);
            count = 0;
            % get the order of the job
            for k = 1:size(pop, 2)
                if pop(i, k) == job && k <= j
                    count = count + 1;
                    if k == j
                        break;
                    end
                end
            end
            work_time = time(job, count);
            machine = order(job, count);
            if job_time(job)>mach_time(machine)
                job_time(job) = job_time(job) + work_time;
                mach_time(machine) = job_time(job);
            else
                job_time(job) = mach_time(machine) + work_time;
                mach_time(machine) = job_time(job);
            end
        end
        end_time(i) = max(mach_time);
    end
    fit = 1./(1+end_time);
    max_fit = max(fit);
    min_fit = min(fit);
    fit = (fit - min_fit)./(max_fit - min_fit);
end