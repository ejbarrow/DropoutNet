classdef outputNeuron < handle
   properties
      inputs
      weights
      bias
      output
      size
      learningrate
      outputerror
      inputerror
  
   end
   
   methods
       
      function obj = init(obj,size,bias,learnrate)
          obj.bias = bias;
          obj.size = size;
          obj.output = 0;
          obj.learningrate = learnrate;
          obj.outputerror=0;
          
         for i= 1:size,
             rand =randi([-100, 100]);
             rand=rand/100;
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
      
      function obj = backProp(obj, Target)
         
       obj.outputerror = obj.output * (1 - obj.output)*(Target - obj.output); %get output error
       
       for i= 1:obj.size, %update output weights
           obj.inputerror(i) = obj.outputerror * obj.weights(i); % prepare error to pass to previous layer 
           obj.weights(i)= obj.weights(i) +  (obj.learningrate * obj.outputerror * obj.inputs(i));
       end
       obj.bias = obj.bias + (obj.learningrate * obj.outputerror);
          
      end
      
      
   end 
end