function [best] = sea()

    rng shuffle;
    
    global op;
    op.ub = 50.0;
    op.lb = -50.0;

    global gas; 
    gas.n_individuals = 50;
    gas.generations = 100;
    gas.p_m = 0.9;
    gas.chaos_exp = 30.0;
    gas.variance_generations = 10;
    gas.verbose = true;
    
    gas.fIdx.fit = 1;
    gas.fIdx.ref = 2;
    
    best = [0,0,0];

    % in case a funny user decides to have an odd number of idividuals in the population...
    if mod(gas.n_individuals,2) ~= 0
        gas.n_individuals = gas.n_individuals + 1;
    end
    
    % a funnier user...
    if gas.n_individuals <= 0
        gas.n_individuals = 1;
    end
    
    %--INITIALIZATION 
    variance_array = zeros(1, gas.n_individuals);
    queue = zeros(1, gas.variance_generations);   % queue used to calculate the variance of the last 'variance_generations' generations best individuals
    qIndex = 1;
    variance = 0;
    
    %--RANDOM INITIALIZATION
    pop = initializeRandomPopulation();  

    %--EVALUATION
    [fit_array_P] = evaluate(pop);
    
    plotPop(pop,'bo',true);
   
    %--ITERATIONS
    for gen = 1:1:gas.generations

        if (mod(gen,2) == 1)
            % do symmetry
            offspring = generateOffspringWithSymmetry(pop);

            %--EVALUATION
            fit_array_O = evaluate(offspring);

            %--SURVIVOR
            [pop, fit_array_P] = survivorElitism(pop, offspring, fit_array_P, fit_array_O);
            %[pop, fit_array_P] = survivorNonElitism(offspring, fit_array_O);
        else
            % do mutation
            offspring = generateOffspringWithChaos(pop);

            %--EVALUATION
            fit_array_O = evaluate(offspring);

            %--SURVIVOR
            %[pop, fit_array_P] = survivorElitism(pop, offspring, fit_array_P, fit_array_O);
            [pop, fit_array_P] = survivorNonElitism(offspring, fit_array_O);
        end

        
        

        % calculate variance over the last 'varianceGen' generations
        queue(qIndex) = fit_array_P(1,1);     % variance is on ik fitness only (ranking fitness depends on the current population, so it makes no sense to compare the rank of individuals from different generations)
        qIndex=qIndex + 1;                    % the queue is implemented as a static array
        if qIndex>size(queue,2)             % when the index reaches the end of the array
            qIndex = 1;                     % goes back to 1
        end
        variance = var(nonzeros(queue));    % calculate variance
        variance_array(gen)= variance;

        if(gen == 1 || fit_array_P(1,gas.fIdx.fit)<best(1,3))
            best(1,1:2) = pop(fit_array_P(1,gas.fIdx.ref),:);
            best(1,3) = fit_array_P(1,gas.fIdx.fit);
        end
        
        %--VERBOSE (SHOW LOG)
        if gas.verbose
            fprintf('%d)\t', gen);
            
            %fprintf('Fit: %.3f ', fit_array_P(1,gas.fIdx.fit));
            disp(best);
            %fprintf('\n');
        end
        
        plotPop(pop,'bo',false);
        
        % stop if the variance is 0.0000
        
        if (round(variance,3) == 0) && (gen > gas.variance_generations)
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
    for i = 1:1:gas.n_individuals
        plot(pop(matPool(i),1),pop(matPool(i),2),'go');
    end
    plot(1,1,'r+');
end

function [pop] = initializeRandomPopulation()
    global op;
    global gas; 
    
    % population array is matrix of size (how many individuals) x (how many decision variables) for rosenbrock it is 2
    pop = zeros(gas.n_individuals, 2);
    for i = 1:1:gas.n_individuals
        pop(i,1) = (op.ub-op.lb)*rand + op.lb;
        pop(i,2) = (op.ub-op.lb)*rand + op.lb;
    end
end

function [fit_array] = evaluate(pop)
    global op;
    global gas; 
    s = size(pop,1);
    
    fit_array = zeros(s,3);
    for i = 1:1:s
        x1 = pop(i,1);
        x2 = pop(i,2);
        fit_array(i,gas.fIdx.fit) = 100.0 * (x2 - x1^2)^2 + (1 - x1)^2; % objective function
        fit_array(i,gas.fIdx.ref) = i;
    end
end

function [offPop] = generateOffspringWithSymmetry(pop)
    global op; 
    global gas; 
    offPop = [];
    for i=1:1:gas.n_individuals
        off = symmetry(pop(i,:));
        offPop = [offPop;off];
    end
    offPop = offPop{:,:};
    for i=1:1:size(offPop,1)
        offPop(i,1) = max(min(offPop(i,1),op.ub),op.lb);
        offPop(i,2) = max(min(offPop(i,2),op.ub),op.lb);
    end
end

function [off] = symmetry(ind)
    dv = size(ind,2);
    % combs = [];
    % for i=1:1:dv
    %     combs = [combs;[-1,1]];
    % end
    if (dv == 1)
        combs = combinations([-1,1]);
    elseif (dv == 2)
        combs = combinations([-1,1],[-1,1]);
    elseif (dv == 3)
        combs = combinations([-1,1],[-1,1],[-1,1]);
    end
    
    combs(size(combs,1),:) = [];
    off = ind .* combs;
end

function [offPop] = generateOffspringWithChaos(pop)
    global gas; 
    offPop = [];
    for i=1:1:gas.n_individuals
        off = chaos(pop(i,:));
        offPop = [offPop;off];
    end
end

function [o_mutated] = chaos(o_original)
    global op;
    global gas;

    dv = size(o_original,2);
    v = randn(1,dv);
    v = v./sqrt(v*v');

    r = ((op.ub-op.lb)/gas.chaos_exp) * rand();
    

    o_mutated = o_original + (r*v);
    
    o_mutated(1,1) = max(min(o_mutated(1,1),op.ub),op.lb);
    o_mutated(1,2) = max(min(o_mutated(1,2),op.ub),op.lb);
end

function [pop, fit_array_P] = survivorNonElitism(offspring, fit_array_O)
    global op;  % optimization problem
    global gas; % genetic algorithm settings

    fit_array_O = sortrows(fit_array_O,gas.fIdx.fit);
    

    for i=1:1:gas.n_individuals
        pop(i,:) = offspring(fit_array_O(i,gas.fIdx.ref),:);
    end
    fit_array_P = fit_array_O(1:gas.n_individuals,:);
    
end

function [pop, fit_array_P] = survivorElitism(pop, offspring, fit_array_P, fit_array_O)
    global op;  % optimization problem
    global gas; % genetic algorithm settings
    
    newPop = [pop;offspring];
    new_fit_array = [fit_array_P;fit_array_O];
    for i=gas.n_individuals+1:1:size(newPop,1)
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