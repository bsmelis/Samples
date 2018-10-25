function [overall,accuracy,error,classifiedrate] = thirdgcnn(xtrain,ytrain,xtest,V,nclass,classes)
%xtrain and xtest are data, ytrain onehotencoded (0.1,0.9) labels
%classes class information to calculate fmeasure
%V calculated smoothing parameter
%nclasses #ofclasses


[m,n] = size(xtrain);  %m #of features, n #of element the last column is class
[mt,nt] = size(xtest); %mt %of features, n #of element the last column is class  

%fprintf('############################GCNN######################\n');
%for each test sample, calculate similarity and choose the most similar
%sample in training set
    for p = 1:nt  
        dist = sum(((xtest(1:m-1,p)*ones(1,n))- xtrain(1:m-1,:)).^2);   
        pl = exp(-1*(dist./(2.*(V.^2))));
        nom = pl*(exp(ytrain - (0.9*ones(nclass,n))).*ytrain)';
        denom = sum(pl);
        out = nom ./ denom;               
       [result(p),ind(p)] = max(out); 
       clear nom pl dist denom out
    end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Create confusion matrix for fmeasure %%
    confmat = zeros(nclass,nclass);
    trues = zeros(nclass,1);
    falses = zeros(nclass,1);
    elemans = zeros(nclass,1);

    for l = 1 : nt
        if(xtest(mt,l) == classes(ind(l)))
            confmat(ind(l),ind(l)) =  confmat(ind(l),ind(l)) + 1;
            trues(ind(l)) = trues(ind(l)) + 1;
            elemans(ind(l)) = elemans(ind(l)) + 1;
        else
            indt = find(classes == xtest(mt,l));
          confmat(indt,ind(l)) =  confmat(indt,ind(l)) + 1;
          falses(indt) = falses(indt) + 1;
          elemans(indt) = elemans(indt) + 1;
          clear indt
        end
    end
    %%%% Call fmaeasure function: if there are more than two classes,
    %%%% overall performance is calculated as average of fmeasures
       overall = fmeasure(nclass,elemans,trues,falses);
        error = 0;
        accuracy = 0;
        classifiedrate = 0;
     % to calculate classification performance predefined function
     % classperf is used
      if(any(xtest(mt,:) ~= xtest(mt,1)))
           cp = classperf(xtest(mt,:),classes(ind));
           error =cp.ErrorRate
           accuracy =cp.CorrectRate
           classifiedrate=cp.ClassifiedRate    
           % if some results approache to Nan value, accuracy will be 0
           if (isnan(accuracy))
               error = 1;
               accuracy = 0;
               classifiedrate = 0;
           end
      end
end



