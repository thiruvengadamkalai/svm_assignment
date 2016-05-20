function [weight,bias, truePositive, falsePositive]=trysmo(test,testclasses,train,trainclasses, kernel, eps)
num_pos=0;
num_neg=0;

train_rows=size(train(:,1));
test_rows = size(test,1);
num_attr = size(train(1,:));
for i=1:train_rows
    if train(i)==1
        num_pos=num_pos+1;
    else
        num_neg=num_neg+1;
    end
end

 alpha = randi([0, 1000], [trainrowCount, 1]);
    diff = dot(alpha, trainclasses);

for i=1:num_pos
      alpha(i) = alpha(i) - (diff /num_pos);
end
b=zeros(train_rows,1);
for iters=1:100
    w=zeros(1,num_attr)
    for i=1:num_rows
      w=w+alpha(i)*trainclasses(i)*train(i,:);
    end
    for i=1:num_rows
        temp=dot(w(i,:),train(i,:));
        temp=trainclasses(i)*(temp+b(i));
        kkt(i)=alpha(i)*(temp-1);
    end
    [maxkktval, maxkktind] = max(kkt);
    X2=train(maxkktind,:);
    e = zeros(train_rows,1);
    for i=1:train_rows
        for j=1:train_rows
         temp=alpha(j)*trainclasses(j)*(kernel(j,maxkktind)-kernel(j,i));
        e(i)=e(i)+temp+trainclasses(i)-trainclasses(maxkktind);    
        end
    end
    abse=abs(e);
    [maxeval,maxeind]=max(abse);
    k=kernel(maxkktind,maxkktind)+kernel(maxeind,maxeind)-2*kernel(maxkktind,maxkeind);
    oldalpha2=alpha(maxeind);
    newalpha2=oldalpha2+(trainclasses(maxeind) * e(maxeind)) / k;
    alpha(maxeind)=newalpha2;
    if (alpha(maxeind)<0)
     alpha(maxeind)=0;
    end
    oldalpha1=alpa(maxkktind);
    newalpha1=oldalpha1+trainclasses(maxeind)*trainclasses(maxkktind)*(oldalpha2-newaplha2);      
    alpha(maxkktind)=newalpha1;
    if(alpha(maxkktind)<0)
     alpha(maxkktind)=0;
    end
%calculating the bias
     for i = 1:train_rows
            if alpha(i) > 0
                b(i) = trainclasses(i) - dot(w, train(i, :));
            end
     end
 
     if abs(dot(alpha, trainclasses)) > 0.00000001
            disp('dot(alpha, classes) != 0:');
           % error('The dot(alpha, classes) != 0');
     end
%total bias
    biascount=0;
    for i=1:train_rows
      if alpha(i)~=0
        biascount=biascount+1;
      end
    end
    totalbias=sum(b)/biascount;
%the updated weight vector
    w=zeros(1,num_attr);
    for i=1:train_rows
        w=w+alpha(i)*trainclasses(i)*train(i,:);
    end
    for i=1:train_rows
         if(testclasses(i)==2)
          testclasses(i)=-1;
         end
    end

    classes=sign(dot(w,test(i,:))+totalbias);
%it gives what the corresponding test instance is being classified as
    accuracy=confusionmat(classes,testclasses);
    if (accuracy(1,2)+accuracy(2,1))/(accuracy(1,1)+accuracy(1,2)+accuracy(2,1)+accuracy(2,2))<eps
       break
    end
end
weight=w;
bias=b;
truePositive=accuracy(1,1);
falsePositive=accuracy(2,1);
end