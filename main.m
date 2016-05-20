%Assignment-5


data=dlmread('heart.txt');
[trainind,testind]=dividerand(270,200,70);
traindata=data(trainind,:);
testdata=data(testind,:);
test=testdata(:,1:13);
testclasses=testdata(:,14);
train=traindata(:,1:13);
trainclasses=traindata(:,14);

%using gaussian kernel
variance=0.8;
for i=1:200
    for j=1:200
        diff=train(i,:)-train(j,:);     %distance between the features
       squared_diff=dot(diff,diff);
     
      kernel(i,j)=exp((squared_diff*(-1))/(2*variance*variance));
    end
end
%using sigmoid gradient descent kernel
c1=0.009;
c2=-1000;
for i=1:200
    for j=1:200
        mul=dot(train(i,:),train(j,:));
        kernel(i,j)=tanh((c1*mul)+c2);
        
    end
end

%using polynomial kernel
for i=1:200
    for j=1:200
        mul=dot(train(i,:),train(j,:));
        mul=1+mul;
        kernel(i,j)=power(mul,2);
        
    end
end


eps=0.0005;
%for p=1:200
 %   for q=1:200
  %      kernel(p,q)=1;
   % end
%end
[weight,bias, truePositive, falsePositive]=smo(test,testclasses,train,trainclasses, kernel, eps);
