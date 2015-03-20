classdef outputlayer < handle
   properties
      inputs
      outputs
      
      weights 
      
      bias
      learningrate
      outputerror
      
      neurons = outputNeuron;
      
      numNeurons
      numInputs
      
      Target
      Targetnames
      
      onesarray
      
      errorArray %to pass to previous array
       neuronerror
       
      count
      neuronerrorarray
      meanerror
      
      dropAmount
      droppedNeurons
      weightCopy
       
   end
   
   methods
       
        function obj = init(obj,Inputsize, Neuronsize, bias,learnrate,targetnames)
          obj.bias(1:Neuronsize) = bias;
          obj.numInputs = Inputsize;
          obj.numNeurons=Neuronsize;
          obj.learningrate = learnrate;
          obj.outputerror=0;
          obj.Targetnames = targetnames;
          obj.onesarray=ones(1,Neuronsize);
          obj.count =1;
          
             for i= 1:Neuronsize,
                 for j= 1:Inputsize,
                      rand =randi([-100, 100]); % -1 and 1
                      rand=rand/100;%redundant
                      rand=rand/Inputsize;
                      %obj.neurons(i) = hiddenNeuron;
                      % obj.neurons(i).init(Inputsize,bias,learnrate);
                      %obj.neurons(i).setinput(ins);
                      obj.weights(i,j) = rand;
                 end

             end                        
        end

        
        function obj = SetInputs(obj,input)
          obj.inputs = input;
       
%              for i= 1:obj.numNeurons,
%                   obj.neurons(i).setinput(input);
%              end                        
        end
        
        function obj = getOutputs(obj)       
%              for i= 1:obj.numNeurons,
%                   obj.neurons(i).calculateoutput();
%                   obj.outputs(i) = obj.neurons(i).output;
%              end     

                obj.outputs = sigmoid(obj.inputs*transpose(obj.weights)+obj.bias);
        end
        
        function obj = backProp(obj,targets, samplesize)       
             obj.Target = targets;


                inputonesarray=ones(1,obj.numInputs);

                obj.neuronerror = obj.outputs .* (1-obj.outputs) .* (obj.Target - obj.outputs);
                obj.errorArray = obj.neuronerror * obj.weights;
                nerror = transpose(obj.neuronerror) * inputonesarray;
                inputmat =transpose( transpose(obj.inputs)*obj.onesarray);

                    obj.weights = obj.weights + (obj.learningrate * nerror .* inputmat);
                    obj.applyDrops();
                     obj.bias = obj.bias + (obj.learningrate * obj.neuronerror);

                

             obj.outputerror = sumsqr(obj.neuronerror)/ obj.numNeurons;

        end
       
        function obj = chooseDrops(obj)   
            min=1;
            max=floor(obj.numInputs/10);
            
            amount = randi([min, max]);
            obj.dropAmount = amount;             
            obj.weightCopy=obj.weights;
            
            obj.droppedNeurons = randperm(obj.numInputs,amount);
            
        end
        
        function obj = applyDrops(obj)       
            for i=1:obj.dropAmount,
                obj.weights(:,obj.droppedNeurons(i))=0;
            end            
        end
        
        function obj = revertDrops(obj)       
            for i=1:obj.dropAmount,
                obj.weights(:,obj.droppedNeurons(i))=obj.weightCopy(:,obj.droppedNeurons(i));
            end     
                    
        end
       
   end
   
   
end
       