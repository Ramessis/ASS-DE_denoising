%Different Evolution with autonomous strategy selection

%==input parameter==
%image:the original image
%imn:the denosing image
%problem: the serial number of testing function recorded in "Public\benchmark_func.m"
%N: the population size
%runmax: the number of the algorithm runs

%==output parameter==
%RunResult: the  optimal value produced by each algorithm runs
%RunOptimization: the optimal value produced by reach algorithm runs
%RunValue: the fitness of optimal value produced by each 10000 FES
%RunParameter:the optimal value produced by each 10000 FES
%RunTime: the time spent by each algorithm runs
%RunFES: the FES required to satisfy the conditions

function [RunResult,RunValue,RunTime,RunFES,RunOptimization,RunParameter]=ASS_DE(img,imn,problem,N,runmax)
    D=Dim(problem);
    lu=Boundary(D);
    tempTEV=Error(D);
    TEV = tempTEV(problem);
    G=50;
    FESMAX=G*N;
    RunOptimization=zeros(runmax,D);

    for run=1:runmax
        TimeFlag=0;
        TempFES=FESMAX;
        t1=clock;

        %--------------------CR--------------------
        CR=zeros(N,1);
        good_uCR=0.5*ones(1,4);
        goodCR=zeros(N,4);

        %--------------------F---------------------
        F=zeros(N,1);
        good_uF=0.5*ones(1,4);
        goodF=zeros(N,4);

        % parameter of q-learning
        gama=0.95;
        alpha=0.9;
        c=0.1;
        a1=0;a2=0;a3=0;a4=0;
        A1=0;A2=0;A3=0;A4=0;
        a1_num=0;a2_num=0;a3_num=0;a4_num=0;
        C=100;
        x=Initpop(N,D,lu);% initialize population

        x_old=[];
        v=zeros(N,D);

        % parameter of p-best srategy
        p=0.05;

        % historical archive
        history_bestpop = zeros(3001,10);

        % calculate fitness
        fitness=benchmark_func(imn,img,x);
        [x_best,x_best_number]=min(fitness);
        history_bestpop = x(x_best_number,:);
        stagnation=0;
        FES=N;
        k=1;
        
        % Initialize the parameter Q-table for each individual
        Q=zeros(2,4,N);
        state=ceil(rand(N,1)*1);
        state1=0;
        state2=0;
        epsilon=0.5;
        Pro = 0;
        T=x;
        a=ceil(rand*4);
        Fit = zeros(N,1);
        G=1;

        %% main loop
        while FES<=FESMAX
            Fitness = fitness;
            min_fitness=x_best;
            epsilon=epsilon-(1/(FESMAX/100*2));
            Diversity = std(std(x));
            
            for i=1:N
                % calculate the action selection probability
                pos=0;
                pso=0;
                
                % Generate CR according to a normal distribution with mean CRm, and std 0.1
                % Generate F according to a cauchy distribution with location parameter Fm, and scale parameter 0.1
                [F(i,a), CR(i,a)] = randFCR(good_uCR(1,a), 0.1, good_uF(1,a), 0.1);
                A(i,G)=a;
                  
                x_all=[x; x_old];

                % generate ri individuals used in mutation
                indexSet=1:size(x_all,1);
                indexSet(i)=[];

                temp=floor(rand*(N-1))+1;
                r(1)=indexSet(temp);
                indexSet(temp)=[];

                temp=floor(rand*(N-2))+1;
                r(2)=indexSet(temp);
                indexSet(temp)=[];

                temp=floor(rand*(N-3))+1;
                r(3)=indexSet(temp);
                indexSet(temp)=[];

                temp=floor(rand*(N-4))+1;
                r(4)=indexSet(temp);
                indexSet(temp)=[];

                temp=floor(rand*(size(x_all,1)-5))+1;
                r(5)=indexSet(temp);
                
                len_ku = size(history_bestpop,1);
                if len_ku < 2
                    R = [1,1];
                else
                    R=randperm(round(len_ku));
                end

                % Find the p-best solutions
                pNP = max(round(p * N), 2); % choose at least two best solutions
                randindex=ceil(rand*pNP);
                [valBest, indBest] = sort(fitness, 'ascend');
                pbest = x(indBest(randindex), :); % randomly choose one of the top 100p% solutions
                
                % select action space according to diversity
                if Diversity > 1
                    if a==1
                        v(i,:)=x(indBest(randindex),:)+F(i,a)*(x(r(2),:)-x(r(3),:));%Ibest/1
                        %a1_num=a1_num+1;
                    elseif a==2
                        v(i,:)=x(i,:)+F(i,a)*(pbest-x(i,:))+F(i,a)*(x(r(1),:)-x_all(r(4),:));%DE/current-to-pbest/1
                        %a2_num=a2_num+1;
                    elseif a==3 
                        v(i,:)=x(i,:)+F(i,a)*(x(r(1),:)-x(i,:))+F(i,a)*(x(r(2),:)-x_all(r(3),:));%DE/current-to-rand/1
                        %a3_num=a3_num+1;
                    elseif a==4
                        v(i,:)=x(i,:)+F(i,a)*(x(indBest(randindex),:)-x(i,:))+F(i,a)*(x(r(1),:)-x(r(2),:));%DE/best-to-rand/1
                        %a4_num=a4_num+1;
                    end
                else
                    if a==1
                        v(i,:)=x(indBest(1),:)+F(i,a)*(x(r(2),:)-x(r(3),:));%best/1
                        %a1_num=a1_num+1;
                    elseif a==2
                        v(i,:)=x(i,:)+F(i,a)*(x(r(1),:)-x(i,:))+F(i,a)*(x(r(2),:)-x_all(r(3),:));%DE/current-to-rand/1
                        %a2_num=a2_num+1;
                    elseif a==3
                        v(i,:)=x(i,:)+F(i,a)*(x(indBest(1),:)-x(i,:))+F(i,a)*(history_bestpop(R(1),:)-history_bestpop(R(2),:));%DE/current-to-best-hbest/1
                        %a3_num=a3_num+1;
                    elseif a==4
                        v(i,:)=x(i,:)+F(i,a)*(x(indBest(1),:)-x(i,:))+F(i,a)*(x(r(1),:)-x(r(2),:));%DE/current-to-best/1
                        %a4_num=a4_num+1;
                    end
                end

                % crossover
                jrand=floor(rand*D)+1;
                for j=1:D
                    if  v(i,j)>lu(2,j)
                        v(i,j)=max(lu(1,j),2*lu(2,j)-v(i,j));
                    end
                    if  v(i,j)<lu(1,j)
                        v(i,j)=min(lu(2,j),2*lu(1,j)-v(i,j));
                    end
                    if (rand<CR(i))||(j==jrand)
                        newx(i,j)=v(i,j);
                    else
                        newx(i,j)=x(i,j);
                    end
                end
                
                children_fitness(i)=benchmark_func(imn,img,newx(i,:));

                if a == 1
                    a1 = a1 + 1;
                elseif a == 2
                    a2 = a2 + 1;
                elseif a == 3
                    a3 = a3 + 1;
                else
                    a4 = a4 + 1;
                end
 
                % selection
                coeff(i) = myPearson(newx(i,:),x(i,:));

                if children_fitness(i)<fitness(i)
                    new_state=1;
                    state1=state1+1;
                    
                    Promote_rate(i)=(fitness(i)-children_fitness(i))/fitness(i);

                    % calculate success rate
                    if a == 1
                        A1 = A1 + 1;
                    elseif a == 2
                        A2 = A2 + 1;
                    elseif a == 3
                        A3 = A3 + 1;
                    else
                        A4 = A4 + 1;
                    end
                    a1_num=A1/a1;
                    a2_num=A2/a2;
                    a3_num=A3/a3;
                    a4_num=A4/a4;

                    % calculate individual reward
                    if FES <= 100
                        reward=Promote_rate(i)*C;
                    elseif FES > 100 && a == 1
                        reward=Promote_rate(i)*C + a1_num;
                    elseif FES > 100 && a == 2
                        reward=Promote_rate(i)*C + a2_num;
                    elseif FES > 100 && a == 3
                        reward=Promote_rate(i)*C + a3_num;
                    elseif FES > 100 && a == 4
                        reward=Promote_rate(i)*C + a4_num;
                    end
                    
                    % individual update
                    x(i,:)=newx(i,:);

                    % fitness update
                    fitness(i)=children_fitness(i);

                    % recording success parameters
                    goodF(i,a) = F(i,a);
                    goodCR(i,a) = CR(i,a);

                else
                    new_state=2;
                    state2=state2+1;

                    reward=0;
                end 

                % calculate q-value and update q-table
                Q(state(i,:),a,i)=Q(state(i,:),a,i)+alpha*(reward+gama*max(Q(new_state,:,i))-Q(state(i,:),a,i));
               
                if mod(FES,100)==0
                    Fit = [Fit, fitness];
                end
                 
                % select action       
                if coeff(i) > 0.2 && max(Q(state(i),:,i))~=0 && epsilon < rand
                    [~,a] = max(Q(state(i),:,i));
                else
                    a=ceil(rand*4);
                end

                state(i,:)=new_state;
                
                FES=FES+1;
                if mod(FES,50) == 0
                    [kk,ll]=min(fitness);
                    RunValue(run,k)=kk;
                    Para(k,:)=x(ll,:);
                    k=k+1;
                    fprintf('Algorithm:%s problemIndex:%d Run:%d FES:%d Best:%g\n','ASS_DE',problem,run,FES,kk);
                end
                if TimeFlag==0
                    if min(fitness)<=TEV
                        TempFES=FES;
                        TimeFlag=1;
                    end
                end
            end
            G=G+1;

            [min_fit_g,min_fit_index]=min(fitness);
            history_bestpop(G,:) = x(min_fit_index,:);
            
            % parameter archive update
            % F
            for K=1:4
                goodF_pos = find(goodF(:,K) > 0);
                dif = abs(Fitness - children_fitness');
                dif_val = dif(goodF_pos);
                num_success_params = numel(goodF_pos);
                if num_success_params > 0
                    sum_dif = sum(dif_val);
                    dif_val = dif_val / sum_dif;
                    good_uF(1,K)=(1-c)*good_uF(1,K)+c*(dif_val' * (goodF(goodF_pos,K) .^ 2)) / (dif_val' * goodF(goodF_pos,K));
                end
            end
            
            % CR
            for K=1:4
                goodCR_pos = find(goodCR(:,K) > 0);
                Dif_val = dif(goodCR_pos);
                Num_success_params = numel(goodCR_pos);
                if Num_success_params > 0
                    sum_dif = sum(Dif_val);
                    Dif_val = Dif_val / sum_dif;
                    good_uCR(1,K)=(1-c)*good_uCR(1,K)+c*(Dif_val' * (goodCR(goodCR_pos,K) .^ 2)) / (Dif_val' * goodCR(goodCR_pos,K));
                end
            end
           
            for o=1:N
                if Fitness(o) > fitness(o)
                    pos=[pos,o];
                else
                    pso=[pso,o];
                end
            end

            if ~isempty(pso)
                pso(1)=[];
            end

            x_old=x(pso,:);
            
            goodF=zeros(N,4);
            goodCR=zeros(N,4);
        end
        
        [kk,ll]=min(fitness);
        gbest=x(ll,:);
        RunResult(run)=kk;
        t2=clock;
        RunTime(run)=etime(t2,t1);
        RunResult(run)=kk;
        RunFES(run)=TempFES;
        RunOptimization(run,1:D)=gbest;
        RunParameter{run}=Para;
    end
end


function [F,CR] = randFCR( CRmgood_uCR, CRsigma, good_uF,  Fsigma)

% this function generate CR according to a normal distribution with mean "CRm" and sigma "CRsigma"
%           If CR > 1, set CR = 1. If CR < 0, set CR = 0.
% this function generate F  according to a cauchy distribution with location parameter "Fm" and scale parameter "Fsigma"
%           If F > 1, set F = 1. If F <= 0, regenrate F.
% generate CR
    CR = CRmgood_uCR + CRsigma * randn();
    CR = min(1, max(0, CR));                % truncated to [0 1]

    % generate F
    F = randCauchy(good_uF, Fsigma);
    F = min(1, F);                          % truncation

    % we don't want F = 0. So, if F<=0, we regenerate F (instead of trucating it to 0)

    while F<=0
        F = randCauchy(good_uF, Fsigma);
        F = min(1, F);                      % truncation
    end
end


% Cauchy distribution: cauchypdf = @(x, mu, delta) 1/pi*delta./((x-mu).^2+delta^2)
function result = randCauchy(mu, delta)
    % http://en.wikipedia.org/wiki/Cauchy_distribution
    result = mu + delta * tan(pi * (rand - 0.5));
end