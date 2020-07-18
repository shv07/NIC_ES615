%reference - https://www.youtube.com/watch?time_continue=1835&v=xPkRL_Gt6PI&feature=emb_logo

clc;
clear;
close all;

%% Problem Defination
    CostFunction = @(x) Cit(x);  % Function to optimize (defined at the end of the file)

    nVar = 2;                   % Number of Unknown (Decision) Variables, as defined for the given problem

    VarSize = [1 nVar];         % Matrix Size of Decision Variables

    VarMin = -10;	% Lower limit of each of the Decision Variables
    VarMax = 10;    % Upper limit of each of the Decision Variables
    


%% Parameters of PSO

    MaxIt = 200;   % Maximum Number of Iterations for the PSO algorithm

    nPop = 5;     % Population Size or Swarm Size

    w = 1;           % Intertia Coefficient according to the standard PSO
    wdamp = 0.99;   % Damping Ratio of Inertia Coefficient 
    c1 = 2;         % Personal Acceleration Coefficient according to the standard PSO
    c2 = 2;         % Social Acceleration Coefficient according to the standard PSO
    
    MaxVelocity = 0.2*(VarMax-VarMin);
    MinVelocity = -MaxVelocity;
    
%% Initialization for PSO 

    empty_particle.Position = [];
    empty_particle.Velocity = [];
    empty_particle.Cost = [];
    empty_particle.Best.Position = [];  %P_best
    empty_particle.Best.Cost = [];      %fitness of P_best
    
    particle = repmat(empty_particle, nPop, 1);
    
    GlobalBest.Cost = inf;   %initial global best fitness set to +infinity since a minimisation problem

    for i=1:nPop

        % Generate Random Solution
        particle(i).Position = unifrnd(VarMin, VarMax, VarSize);

        % Initialize Velocity
        particle(i).Velocity = zeros(VarSize);

        % Evaluation
        particle(i).Cost = CostFunction(particle(i).Position);

        % Update the Personal Best
        particle(i).Best.Position = particle(i).Position; %pbest
        particle(i).Best.Cost = particle(i).Cost;         %fitness of pbest

        % Update Global Best
        if particle(i).Best.Cost < GlobalBest.Cost        %gbest
            GlobalBest = particle(i).Best;
        end
    end
    % Array to Hold Best Cost Value on Each Iteration
    BestCosts = zeros(MaxIt, 1);
    
    

    %f = zeros(MaxIt, 1);
    
    X = linspace(VarMin, VarMax);
    Y = linspace(VarMin, VarMax);
    [X,Y] = meshgrid(X,Y);
    Z = Cit1(X,Y);
    figure;
    surf(X,Y,Z);
    title('Plot of the Cross-over-tray Function');
    
    for k=1:10
        f(k) = getframe(gcf);
    end
    figure;
    contourf(X, Y, Z)
    view(0,270)
%% Main Loop of PSO
    for it=1:MaxIt
        for i=1:nPop
            % Update Velocity using the PSO equations
            particle(i).Velocity = w*particle(i).Velocity ...
                + c1*rand(VarSize).*(particle(i).Best.Position - particle(i).Position) ...
                + c2*rand(VarSize).*(GlobalBest.Position - particle(i).Position);

            % Apply Velocity Limits
            particle(i).Velocity = max(particle(i).Velocity, MinVelocity);
            particle(i).Velocity = min(particle(i).Velocity, MaxVelocity);

            % Update Position using the PSO equations
            particle(i).Position = particle(i).Position + particle(i).Velocity;

            % Apply Lower and Upper Bound Limits
            particle(i).Position = max(particle(i).Position, VarMin);
            particle(i).Position = min(particle(i).Position, VarMax);

            % Evaluation
            particle(i).Cost = CostFunction(particle(i).Position);

            % Update Personal Best
            if particle(i).Cost < particle(i).Best.Cost

                particle(i).Best.Position = particle(i).Position;
                particle(i).Best.Cost = particle(i).Cost;

                % Update Global Best
                if particle(i).Best.Cost < GlobalBest.Cost
                    GlobalBest = particle(i).Best;
                end
            end
        end
        
        % Store the Best Cost Value
        BestCosts(it) = GlobalBest.Cost;


        % Damping Inertia Coefficient
        w = w * wdamp;
        hold on
        plot3(GlobalBest.Position(1),GlobalBest.Position(2),GlobalBest.Cost,'.r','markersize',8)  
        title({['Movement of Gbest with respect to number of iterations'], ['Position of Gbest: ' num2str(GlobalBest.Position(1)) ' , ' num2str(GlobalBest.Position(2))], ['Iteration No:' num2str(it)], ['Best functional value =' num2str(GlobalBest.Cost)]});
        pause(0.00025)
        f(10+it) = getframe(gcf);
        hold off;
    end
    
%% Results and plots
figure;
plot(BestCosts, 'LineWidth',2);

xlabel('Iteration');
ylabel('Fitness of Global Best');
title(['Plot of fitness of global best v/s iteration for population size ', num2str(nPop)]);
grid on;


f(10+MaxIt+1) = getframe(gcf);
writerObj = VideoWriter('C:\Users\SHIVJI\Desktop\Assignment2_PSO1.avi','Uncompressed AVI' );
writerObj.FrameRate = 5;
open(writerObj);
writeVideo(writerObj,f);
close(writerObj);


%% PSO Function Definition
    
    function f = Cit(x)%cross_in_tray function
        f = -0.0001*(( abs( sin(x(1)).*sin(x(2)).*exp(abs( 100 - (sqrt((x(1)).^2 + (x(2)).^2)/3.143) ) ) )+1).^0.1);
    end
    
    function f1 = Cit1(x,y) %cross in tray function alternate defination to be used for plots
            f1 = -0.0001*(( abs( sin(x).*sin(y).*exp(abs( 100 - (sqrt((x).^2 + (y).^2)/3.143) ) ) )+1).^0.1);
    end
    
        