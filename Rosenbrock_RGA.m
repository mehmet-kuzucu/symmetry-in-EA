function [pop, fit_array_P] = rga()

    rng shuffle;
    
    global op;
    op.ub = 50.0;
    op.lb = -50.0;

    global gas; 
    gas.n_individuals = 500;
    gas.generations = 100;
    gas.p_c = 1.0;
    gas.p_m = 0.9;
    gas.eta_c = 10;
    gas.eta_m = 1.0;
    gas.variance_generations = 10;
    gas.verbose = true;
    
    gas.fIdx.fit = 1;
    gas.fIdx.ref = 2;
    
    
    
    %--INITIALIZATION 
    variance_array= zeros(1,gas.n_individuals);
    queue=zeros(1,gas.variance_generations);   % queue used to calculate the variance of the last 'variance_generations' generations best individuals
    qIndex = 1;
    variance = 0;
    
    % in case a funny user decides to have an odd number of idividuals in the population...
    if mod(gas.n_individuals,2) ~= 0
        gas.n_individuals = gas.n_individuals + 1;
    end
    
    % a funnier user...
    if gas.n_individuals <= 0
        gas.n_individuals = 1;
    end
    
    %--RANDOM INITIALIZATION
    pop = initializeRandomPopulation();  

    %--EVALUATION
    [fit_array_P] = evaluate(pop);
    
    plotPop(pop,'bo',true);
   
    %--ITERATIONS
    for gen=1:1:gas.generations
        %--SELECTION
        matPool = selection(fit_array_P,2,true);
        plotPop(pop,'bo',true);
        plotMatPool(pop,matPool);
        
        %--VARIATION
        offspring = variation(pop, matPool);
        plotPop(offspring,'ko',false);
        
        %--EVALUATION
        fit_array_O = evaluate(offspring);
         
        %--SURVIVOR
        [pop, fit_array_P] = survivor(pop, offspring, fit_array_P, fit_array_O);

        % calculate variance over the last 'varianceGen' generations
        queue(qIndex)=fit_array_P(1,1);     % variance is on ik fitness only (ranking fitness depends on the current population, so it makes no sense to compare the rank of individuals from different generations)
        qIndex=qIndex+1;                    % the queue is implemented as a static array
        if qIndex>size(queue,2)             % when the index reaches the end of the array
            qIndex = 1;                     % goes back to 1
        end
        variance = var(nonzeros(queue));    % calculate variance
        variance_array(gen)= variance;
        
        %--VERBOSE (SHOW LOG)
        if gas.verbose
            fprintf('%d)\t', gen);
            
            fprintf('Fit: %.3f ', fit_array_P(1,gas.fIdx.fit));
            fprintf('\n');
        end
        
        plotPop(pop,'bo',false);
        
        % stop if the variance is 0.0000
        
        if (round(variance,3) == 0) && (gen>gas.variance_generations)
            break;
        end
         
    end  % place a breakpoint here as you run the algorithm to pause, and check how the individuals are evolving by plotting the best one with 'drawProblem2D(decodeIndividual(pop(:,:,1)))'
end

function [] = plotPop(pop,style,isNew)
    global op;
    if isNew == true
        figure;
    end
    
    plot(pop(:,1),pop(:,2),style);
    hold on;
    plot(1,1,'r+');
    xlim([op.lb op.ub]);
    ylim([op.lb op.ub]);
    hold off;
end

function [] = plotMatPool(pop,matPool)
    %figure;
    global gas; 
    hold on;
    for i=1:1:gas.n_individuals
        plot(pop(matPool(i),1),pop(matPool(i),2),'go');
    end
    plot(1,1,'r+');
end

function [pop] = initializeRandomPopulation()
    global op;
    global gas; 
    
    pop = zeros(gas.n_individuals,2);
    for i=1:1:gas.n_individuals
        pop(i,1) = (op.ub-op.lb)*rand + op.lb;
        pop(i,2) = (op.ub-op.lb)*rand + op.lb;
    end
end

function [fit_array] = evaluate(pop)
    global op;
    global gas; 
    
    fit_array = zeros(gas.n_individuals,3);
    for i=1:1:gas.n_individuals
        x1 = pop(i,1);
        x2 = pop(i,2);
        fit_array(i,gas.fIdx.fit) = 100.0 * (x2 - x1^2)^2 + (1 - x1)^2; % objective function
        fit_array(i,gas.fIdx.ref) = i;
    end
end

function [matPool] = selection(fit_array, k, isMin)
    global gas; % genetic algorithm settings
    
    matPool = zeros(gas.n_individuals,1);
    for i=1:gas.n_individuals
        bestFit = 0;
        winner = 0;
        for j=1:k
            index = ceil((gas.n_individuals)*rand);
            if j==1
                bestFit = fit_array(index,gas.fIdx.fit);
                winner = fit_array(index,gas.fIdx.ref);
            else
                if isMin == true
                    % for minimization problems
                    if bestFit > fit_array(index,gas.fIdx.fit)
                        bestFit = fit_array(index,gas.fIdx.fit);
                        winner = fit_array(index,gas.fIdx.ref);
                    end
                else
                    % for maximization problems
                    if bestFit < fit_array(index,gas.fIdx.fit)
                        bestFit = fit_array(index,gas.fIdx.fit);
                        winner = fit_array(index,gas.fIdx.ref);
                    end
                end
            end
        end
        matPool(i) = winner;
    end
end

function [offspring] = variation(pop, matPool)

    global op;  % optimization problem
    global gas; % genetic algorithm settings

    % declare a static array of chromosomes filled with zeros
    offspring = zeros(gas.n_individuals,2);
    
    matPool = matPool(randperm(length(matPool))); % shuffle the mating pool
    
    % this cannot be parallelized (or can it?)
    for i=1:2:gas.n_individuals

        % crossover
        index_p1 = matPool(i);
        index_p2 = matPool(i+1);

        p1 = pop(index_p1,:);
        p2 = pop(index_p2,:);

        [o1, o2] = crossover(p1, p2);   

        % mutation
        o1 = mutation(o1);
        o2 = mutation(o2);

        offspring(i,:) = o1;
        offspring(i+1,:) = o2;
        
    end
    
end

function [o1,o2] = crossover(p1,p2)
    global op;
    global gas;
    o1 = p1;
    o2 = p2;
    if rand() <= gas.p_c
        k = rand();
        beta = 0.0;
        if k <= 0.5
            beta = (2.0*k)^(1.0/(gas.eta_c+1));
        else
            beta = (1/(2.0*(1.0-k)))^(1.0/(gas.eta_c+1));
        end
        o1 = 0.5 * ((p1+p2)-beta*(p2-p1));
        o2 = 0.5 * ((p1+p2)+beta*(p2-p1));
        
        o1(1,1) = max(min(o1(1,1),op.ub),op.lb);
        o1(1,2) = max(min(o1(1,2),op.ub),op.lb);
        o2(1,1) = max(min(o2(1,1),op.ub),op.lb);
        o2(1,2) = max(min(o2(1,2),op.ub),op.lb);
        
    end
end

function [o_mutated] = mutation(o_original)
    global op;
    global gas;
    o_mutated = o_original;
    if rand() <= gas.p_m
        r = rand();
        delta = 0.0;
        if r < 0.5
            delta = (2.0*r)^(1.0/(gas.eta_m+1))-1.0;
        else
           delta = 1.0-(2.0*(1-r))^(1.0/(gas.eta_m+1));
        end
        o_mutated = o_original + (op.ub-op.lb)*delta;
        
        o_mutated(1,1) = max(min(o_mutated(1,1),op.ub),op.lb);
        o_mutated(1,2) = max(min(o_mutated(1,2),op.ub),op.lb);
        
    end
end

function [pop, fit_array_P] = survivor(pop, offspring, fit_array_P, fit_array_O)
    global op;  % optimization problem
    global gas; % genetic algorithm settings
    
    newPop = [pop;offspring];
    new_fit_array = [fit_array_P;fit_array_O];
    for i=gas.n_individuals+1:1:gas.n_individuals*2
        new_fit_array(i,gas.fIdx.ref) = new_fit_array(i,gas.fIdx.ref) + gas.n_individuals;
    end
    
    new_fit_array = sortrows(new_fit_array,gas.fIdx.fit);
    
    new_fit_array(gas.n_individuals+1:end,:) = [];
    
    for i=1:1:gas.n_individuals
        if new_fit_array(i,gas.fIdx.ref) <= gas.n_individuals
            pop(i,:) = pop(new_fit_array(i,gas.fIdx.ref),:);
        else
            pop(i,:) = offspring(new_fit_array(i,gas.fIdx.ref)-gas.n_individuals,:);
        end
    end
    
    for i=1:1:gas.n_individuals
        new_fit_array(i,gas.fIdx.ref) = i;
    end
    
    fit_array_P = new_fit_array;
    
end