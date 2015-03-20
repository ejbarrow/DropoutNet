traincorrect =0;

for i=1:42000,
   results = temp.predict(trainset2(i,:)); 
   highest =0;
   index =0;
   index2 =0;
   for j=1:10,
       if results(j)>highest
          highest = results(j);
          index2=index;
          index = j;
       end      
   end
   
   if targets(i,index)==1
      traincorrect = traincorrect +1; 
   else
       if index2 ~=0
           if targets(i,index2)==1
              traincorrect=traincorrect +1; 
           end
       end
       %fprintf('wrong \n \n');
      
   end
   
    
end

testcorrect =0;
testset3=trainset/255;

for i=1:10000,
   results = temp.predict(testset3(i,:)); 
   highest =0;
   index =0;
   index2=0;
   for j=1:10,
       if results(j)>highest
          highest = results(j);
          index2=index;
          index = j;
       end      
   end
   
   if trainsettargets(i,index)==1
      testcorrect = testcorrect +1; 
   else
       if index2 ~=0
           if trainsettargets(i,index2)==1
               testcorrect = testcorrect+1;
           end
       end
       %fprintf('wrong \n \n');
      
   end
   
    
end