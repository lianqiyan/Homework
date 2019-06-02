function [end_time, sol] = JSP_GA(time, order, cross_rate, mutate_rate, max_iter, pop)
    end_time = zeros(max_iter, 1);
    n = size(time, 1);  % the number of jobs
    m = size(order, 2); % the number of machine
    sol = zeros(max_iter, n*m);
    pop_size = 500;
    if nargin < 6
        pop = init_pop(n, m, pop_size); % init pop
        disp('Initialize the population');
    end
    
    [~,fit] = cal_fit(pop, time, order);
    for i = 1: max_iter
        new_pop = zeros(size(pop));
        for j = 1:pop_size/2
            better = pop(fit>rand, :);
            while(size(better,1)<2)
                better = pop(fit>rand, :);
            end
            sel_index = randi([1, size(better, 1)], 1, 2);
            p1 = better(sel_index(1), :);
            p2 = better(sel_index(2), :);
            if rand<cross_rate
                [c1, c2] = crossover(p1, p2); % cross over
            else
                c1 = p1;
                c2 = p2;
            end
            c1 = mutate(c1, 2, mutate_rate); % mutate
            c2 = mutate(c2, 2, mutate_rate);
            new_pop(2*(j-1) + 1, :) = c1;
            new_pop(2*(j-1) + 2, :) = c2;
        end
        [temp_time,fit] = cal_fit(new_pop, time, order);
        end_time(i) = min(temp_time);
        
        temp = new_pop(temp_time==min(temp_time), :);
        sol(i, :) = temp(1,:);
        pop = new_pop;
    end

end


function pop = init_pop(n, m, pop_size)
    pop = zeros(pop_size, n*m);
    for k = 1: pop_size
        r_order = randperm(n*m);
        for i = 1:n
            pop(k, r_order(1+(i-1)*m :i*m)) = i;
        end
    end
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

function child_new=mutate(child, nsite, mutate_rate)
  nn=length(child);
  child_new=child;
  if rand < mutate_rate
      for i=1:nsite
        index1=floor(rand*nn)+1;
        index2 = index1;
        while(index1 == index2)
            index2=floor(rand*nn)+1;
        end
        temp = child_new(index1);
        child_new(index1) = child_new(index2);
        child_new(index2) = temp;
      end
  end    
end % end for mutation operation


function [child1, child2] = crossover(parent1, parent2)
    len = length(parent1);
    start = randi([1, len-1], 1);
    terminal = 0;
    while(terminal<=start)
        terminal = randi([2, len], 1);
    end
%     start = 4; terminal = 6;
    child1 = gen_child(start, terminal, parent1, parent2);
   child2 = gen_child(start, terminal, parent2, parent1);
end

function child = gen_child(start, terminal, parent1, parent2)
    len = length(parent1);
    succeed = parent1(start:terminal);
    res = [parent2(terminal + 1:end), parent2(1:terminal)];
    index = start:terminal;
    tuple = zeros(length(succeed), 2);
    for i = 1:length(succeed)
        count = 0;
        for j = 1:len
            if parent1(j) == succeed(i) && j <= index(i)
                count = count + 1;
                if j == index(i)
                    tuple(i, :) = [succeed(i), count];
                    break;
                end
            end
        end
    end
    % get delete index
    del_index = zeros(1, length(succeed));
    for i = 1:size(tuple, 1)
        count = 0; 
        for j = 1:len
            if res(j) == tuple(i, 1) && count < tuple(i, 2)
                count = count + 1;
                if count == tuple(i, 2)
                    del_index(i) = j;
                    break;
                end
            end
        end
    end
    % del element
    del_res = zeros(len - length(succeed), 1);
    index = 1;
    for i = 1:length(res)
        if ismember(i, del_index)
            continue
        end
        del_res(index) = res(i);
        index = index + 1;
    end
    child = zeros(1, len);
    child(start:terminal) = succeed;
    child(terminal+1:end) = del_res(1:len-terminal);
    child(1:start-1) = del_res(len-terminal+1:end);
end