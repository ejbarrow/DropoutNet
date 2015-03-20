classdef nnetwork < handle
   properties

       hiddenLayers = hiddenlayer;
       outputLayer
       
       iterationError
       
       inputSize
       NumInputs
       
       NumHiddenLayers
       NumNeuronsPerLayer
       NumOutputs
       
       OutputLabels
       
       Bias
       LearningRate
       
       tempOutput
       tempInputError
       
   end
   
   methods
           function obj = nnetwork

           end
       
       
        %init
         function obj = init(obj,Inputsize, inputcount, HiddenLayers, NeuronsPerLayer, bias,learnrate , OutputCount, outputlabels)
             obj.NumHiddenLayers = HiddenLayers;
             obj.inputSize = Inputsize;
             obj.NumInputs = inputcount;
             obj.NumNeuronsPerLayer = NeuronsPerLayer;
             obj.NumOutputs = OutputCount;
             obj.OutputLabels = outputlabels;
             obj.iterationError =0;
             
             obj.Bias = bias;
             obj.LearningRate = learnrate;     
            
             obj.hiddenLayers(1:HiddenLayers) = hiddenlayer;
             
             if HiddenLayers ==1
                  obj.hiddenLayers(1)=hiddenlayer;
                  obj.hiddenLayers(1).init(Inputsize,NeuronsPerLayer(1),bias,learnrate);
                 
             else
                 obj.hiddenLayers(1)=hiddenlayer;
                 obj.hiddenLayers(1).init(Inputsize,NeuronsPerLayer(1),bias,learnrate);
                 for i= 2:HiddenLayers,
                     obj.hiddenLayers(i)=hiddenlayer;
                     obj.hiddenLayers(i).init(NeuronsPerLayer(i-1),NeuronsPerLayer(i),bias,learnrate);             
                 end 
             end
             
             obj.outputLayer = outputlayer;
             obj.outputLayer.init(NeuronsPerLayer(HiddenLayers),OutputCount,bias,learnrate,outputlabels);

         end
               
        %train
         function obj = train(obj, TrainingData, Targets)
             TrainingData = TrainingData/255;
             obj.iterationError =0;
             for j=1:obj.NumInputs,
                 %fprintf('Training Data %s of %s \n',num2str(j),num2str(obj.NumInputs))
                 if j==(obj.NumInputs/2)
                    fprintf('50%%')
                 end
                 if j==(obj.NumInputs/4)
                    fprintf('25%%')
                 end
                 if j==(obj.NumInputs/2+obj.NumInputs/4)
                    fprintf('75%%')
                 end
                 obj.tempOutput= TrainingData(j,:);
                 for i= 1:obj.NumHiddenLayers, 
                      
                         obj.hiddenLayers(i).SetInputs(obj.tempOutput);
                         obj.hiddenLayers(i).getOutputs();
                         obj.tempOutput = obj.hiddenLayers(i).outputs;
                 end 
                 
                 obj.outputLayer.SetInputs(obj.tempOutput);
                 obj.outputLayer.getOutputs();
                 obj.outputLayer.backProp(Targets(j,:),obj.NumInputs);
                 obj.tempInputError =  obj.outputLayer.errorArray;
                 
                 for i= obj.NumHiddenLayers:-1:1,
                         %fprintf('debug4')
                         obj.hiddenLayers(i).backProp(obj.tempInputError,obj.NumInputs);
                         obj.tempInputError =  obj.hiddenLayers(i).errorArray;
                 end 
                 
                 obj.iterationError = obj.iterationError + obj.outputLayer.outputerror^2;
                 
             end
             %obj.iterationError = obj.iterationError / obj.NumInputs ;
             obj.iterationError = obj.iterationError ;
             fprintf('Epoch Train Error is %s \n',num2str(obj.iterationError))
         end
         
         %train
         function obj = dropoutTrain(obj, TrainingData, Targets)
             TrainingData = TrainingData/255;
             obj.iterationError =0;
             
                 for i= 1:obj.NumHiddenLayers, 
                         obj.hiddenLayers(i).chooseDrops();
                 end 
                 obj.outputLayer.chooseDrops();
                 obj.outputLayer.applyDrops();
                 for i= 1:obj.NumHiddenLayers, 
                     if i==obj.NumHiddenLayers
                         obj.hiddenLayers(i).applyDrops(obj.outputLayer.droppedNeurons,obj.outputLayer.dropAmount);
                     else
                         obj.hiddenLayers(i).applyDrops(obj.hiddenLayers(i+1).droppedNeurons,obj.hiddenLayers(i+1).dropAmount);
   
                     end
                 end 
            
             for j=1:obj.NumInputs,
             %for j=1:1,
                 %fprintf('Training Data %s of %s \n',num2str(j),num2str(obj.NumInputs))
                 if j==(obj.NumInputs/2)
                    fprintf('50%%')
                 end
                 if j==(obj.NumInputs/4)
                    fprintf('25%%')
                 end
                 if j==(obj.NumInputs/2+obj.NumInputs/4)
                    fprintf('75%%')
                 end
                 obj.tempOutput= TrainingData(j,:);
                 for i= 1:obj.NumHiddenLayers, 
                      
                         obj.hiddenLayers(i).SetInputs(obj.tempOutput);
                         obj.hiddenLayers(i).getOutputs();
                         obj.tempOutput = obj.hiddenLayers(i).outputs;
                 end 
                 
                 obj.outputLayer.SetInputs(obj.tempOutput);
                 obj.outputLayer.getOutputs();
                 obj.outputLayer.backProp(Targets(j,:),obj.NumInputs);
                 obj.tempInputError =  obj.outputLayer.errorArray;
                 
                 for i= obj.NumHiddenLayers:-1:1,
                         %fprintf('debug4')
                         obj.hiddenLayers(i).backProp(obj.tempInputError,obj.NumInputs);
                         obj.tempInputError =  obj.hiddenLayers(i).errorArray;
                 end 
                 
                 obj.iterationError = obj.iterationError + obj.outputLayer.outputerror^2;
                 
%                  for i= 1:obj.NumHiddenLayers, 
%                          obj.hiddenLayers(i).applyDrops();
%                  end 
                 
             end
             obj.iterationError = obj.iterationError / obj.NumInputs ;
             %obj.iterationError = obj.iterationError ;
             fprintf('\n Epoch Train Error is %s \n',num2str(obj.iterationError))
             
                 for i= 1:obj.NumHiddenLayers, 
                         obj.hiddenLayers(i).revertDrops();
                 end 
                 obj.outputLayer.revertDrops();
            
         end
        
        
        %predict
        function obj = predict(obj, data)
            data=data/255;     
            obj.tempOutput= data;
                 
            for i= 1:obj.NumHiddenLayers,
                
                     obj.hiddenLayers(i).SetInputs(obj.tempOutput);
                     obj.hiddenLayers(i).getOutputs();
                     obj.tempOutput = obj.hiddenLayers(i).outputs;
            end 
                 
            obj.outputLayer.SetInputs(obj.tempOutput);
            obj.outputLayer.getOutputs();
            
            outputs= obj.outputLayer.outputs;
            
%             for i = 1:obj.NumOutputs,
%                 
%                 disp(obj.OutputLabels(i,:));
%                 disp(outputs(i));
%             end
            obj = outputs;     
  
        end    
       
       
       
   end
   
   
end
       