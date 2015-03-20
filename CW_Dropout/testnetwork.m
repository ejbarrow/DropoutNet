 clear
 load ('MINSTdata.mat');
 MyNetwork = nnetwork;
 %,Inputsize, inputcount, HiddenLayers, NeuronsPerLayer, bias,learnrate , OutputCount, outputlabels
 MyNetwork.init(784,42000,3,[200,200,500],1,0.1,10,datalabels);
 
load ('CW_Weights.mat');

MyNetwork.hiddenLayers(1).weights = weights{1};
MyNetwork.hiddenLayers(2).weights = weights{2};
MyNetwork.hiddenLayers(3).weights = weights{3};
% % MyNetwork.outputLayer.weights = weights{4};
%  
count =1;
minimum =1;
timoutup=0;
timeoutdown=0;
found=false;

zigzag=0;
zig=0;
zag=0;

testSetSize =1000;
numoutputs = 10;
 
while found ==  false,
    fprintf('Training run %s  \n',num2str(count))
    MyNetwork.dropoutTrain(trainset2,targets);
    %MyNetwork.train(trainset,trainsettargets);
    x(count)=count;
    
    validationerror=0;
    for j=1:testSetSize
        result = MyNetwork.predict(testset(j,:)); 
        error=0;
        for k =1:numoutputs
           outputerror = result(k) * (1 - result(k))*(testsettargets(j,k) - result(k)); 
           error = error+outputerror^2;
        end
        validationerror = validationerror + (error/numoutputs);
    end    
    totalvalidationerror(1,count) = validationerror/testSetSize;
    %totalvalidationerror(1,count) = validationerror;
    totaltrainerror(1,count) = MyNetwork.iterationError;
    fprintf('Epoch Validation error %s \n \n ',num2str(validationerror/testSetSize))
    plot(x,totalvalidationerror,'--*',x,totaltrainerror,'-');
    title('Graph of Validation and Test Error over time (iterations)')
    legend('Test  Set Error','Train Set Error');
    xlabel('Iterations'); % x-axis label
    ylabel('Mean Squared Error'); % y-axis label
    pause(.00000000000001);
    
    if (count>1) && (totalvalidationerror(count)<totalvalidationerror(minimum))
        minimum = count;
        temp = MyNetwork;
    end
    
    %how long to go up before Quitting
    if (count>1) && (totalvalidationerror(count)>totalvalidationerror(count-1))
        timeoutup= timeoutup+1;    
    else
        timeoutup=0;
    end
    
    %how long to go down before Quitting
    if (count>1) && (totalvalidationerror(count)<totalvalidationerror(count-1)) && (totalvalidationerror(count)>totalvalidationerror(minimum))
        timeoutdown= timeoutdown+1;    
    else
        timeoutdown=0;
    end
    
    if (count>1) && (totalvalidationerror(count)<totalvalidationerror(count-1))
        if (zag==1)
           zigzag = zigzag+1; 
        else
           zigzag=0;
        end
        zig=1;
        zag=0;
    end
    
    if (count>1) && (totalvalidationerror(count)>totalvalidationerror(count-1))
        if (zig==1)
           zigzag = zigzag+1; 
        else
            zigzag=0;
        end
        zig=0;
        zag=1;
    end
    
    %set found if  timouts reached
    if (timeoutup>3000) || (timeoutdown>3500) ||(zigzag >500)
        found=true;
    end
    
    
    count= count + 1;
    
end

message = sprintf('Minimum Found On Iteration %s \n The Test Error At This Iteration Was %s \n The Validation Error At This Iteration Was %s \n',num2str(minimum),num2str(totaltrainerror(minimum)),num2str(totalvalidationerror(minimum)));
MyNetwork = temp;
h = msgbox(message,'Training Completed');
%plot(x,totalvalidationerror,'--*',x,totaltrainerror,'-');