classdef hiddenNeuron < handle
   properties
      inputs
      weights
      bias
      output
      size
      learningrate
      neuronerror
      inputError
  
   end
   
   methods
       
       function obj = hiddenNeuron

       end
       
      function obj = init(obj,size,bias,learnrate)
          obj.bias = bias;
          obj.size = size;
          obj.output = 0;
          obj.learningrate = learnrate;
          
         for i= 1:size,
             rand =randi([-100, 100]); % -1 and 1
             rand=rand/100;%redundant
             rand=rand/size;
             obj.weights(i)=rand;
         end
      end
      
      
      function obj=  setinput(obj,input)
         obj.inputs = input;
      end
      
      function obj = calculateoutput(obj)
          obj.output=0;
%           for i= 1:obj.size,
%               obj.output = obj.output + (obj.inputs(i)*obj.weights(i));
%           end
          obj.output = obj.inputs*transpose(obj.weights);
          obj.output = sigmoid(obj.output+obj.bias);
          %obj.output = obj.output+obj.bias;
      end
      
      function obj = backProp(obj, Outputerror)
         
          %obj.neuronerror= obj.output * (1-obj.output) * Outputerror;
         
          for i= 1:obj.size,
              %obj.inputError(i) = obj.neuronerror * obj.weights(i); % prepare error to pass to previous layer --- can matrix multiply 
             %obj.weights(i)= obj.weights(i)+ (obj.learningrate * obj.neuronerror * obj.inputs(i)); %update weights
          end
          obj.bias = obj.bias + (obj.learningrate * obj.neuronerror);
      end
      
      
   end 
end