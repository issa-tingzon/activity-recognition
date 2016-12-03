cd '/home/innacalimbahin/Desktop/activity-recognition-master/UCI-ADL-Binary-Dataset/Segmented_Dataset/OrdonezB/with_idle_states/Raw'
currentDir = pwd;
subdircount = 0;
averageAccuracy = [];
averageFmeasure = [];

files = dir;   % assume starting from current directory
filenames = {files.name};
subdirs = filenames([files.isdir]);
for s = 1:length(subdirs)
  subdir = subdirs{s}
  
  % process subdirectory
  tf = isstrprop(subdir, 'digit');
  if(tf)
    subdircount = subdircount+1;
    f = fullfile(currentDir,subdir);
    cd(f);
    pwd
    
    % Start HMM
    trainFile = dlmread('hmm_train.txt');
    testFile = dlmread('hmm_test.txt');

    STATENAMES = [1 2 3 4 5 6 7 8 9 10 11];
    SYMBOLS = [];

    states = transpose(trainFile(:,2));
    trainSeq = transpose(trainFile(:,1));
    testActualStates = transpose(testFile(:,2));
    testSeq = transpose(testFile(:,1));

    for i=1:length(trainSeq)
      if ~ismember(trainSeq(1,i),SYMBOLS)
        SYMBOLS = [SYMBOLS trainSeq(1,i)];
      end
    end

    for i=1:length(testSeq)
      if ~ismember(testSeq(1,i),SYMBOLS)
        SYMBOLS = [SYMBOLS testSeq(1,i)];
      end
    end

    sizeStates = length(STATENAMES);
    sizeSymbols = length(SYMBOLS);

    % Input requirement for hmmtrain: Need to map actual values in training set to 1-length(SYMBOLS)
    newTrainSeq = [];

    PSEUDOE = ones(sizeStates,sizeSymbols);
    PSEUDOTR = ones(sizeStates,sizeStates);

    vals = [1:sizeSymbols];

    for i=1:length(trainSeq)
      for j=1:length(SYMBOLS)
        if (trainSeq(1,i) == SYMBOLS(1,j))
          newTrainSeq = [newTrainSeq vals(1,j)];
          break
        end
      end
    end

    % Convert also test sequence using the same mapping
    newTestSeq = [];
    for i=1:length(testSeq)
      for j=1:length(SYMBOLS)
        if (testSeq(1,i) == SYMBOLS(1,j))
          newTestSeq = [newTestSeq vals(1,j)];
          break
        end
      end
    end

    % Estimate transition and emission matrices
    [TRANS,EMIS] = hmmestimate(newTrainSeq,states,'Statenames',STATENAMES,'Symbols',vals,'Pseudoemissions',PSEUDOE,'Pseudotransitions',PSEUDOTR)
    [ESTTR,ESTEMIT] = hmmtrain(newTrainSeq,TRANS,EMIS,'Symbols',vals,'Pseudoemissions',PSEUDOE,'Pseudotransitions',PSEUDOTR)

    % Test
    likelystates = hmmviterbi(newTestSeq, ESTTR, ESTEMIT,'Statenames',STATENAMES,'Symbols',vals);

    % Metrics
    accuracy = (sum(testActualStates==likelystates))/(length(newTestSeq));
    wrongCount = 0;
    correctCount = 0;

    % Confusion matrix
    m=zeros(length(STATENAMES),length(STATENAMES));
    for j=1:length(newTestSeq)
      actual=testActualStates(1,j);
      inferred=likelystates(1,j);
      if (actual == inferred)
        correctCount = correctCount+1;
      else
        wrongCount = wrongCount + 1;
      end
      m(actual,inferred) = m(actual,inferred)+1;
    end

    % Compute TI,TP,TT
    TI=sum(m);
    TP=transpose(diag(m));
    TT=sum(transpose(m));

    % Compute for precision
    precision = 0.0;
    for j=1:length(STATENAMES)
      if (TP(1,j) == 0)
      else
        precision = precision + TP(1,j)/TI(1,j);
      end
    end

    % Compute for recall
    recall = 0.0;
    for j=1:length(STATENAMES)
      if (TP(1,j) == 0)
      else
        recall = recall + TP(1,j)/TT(1,j);
      end
    end

    c = TP./TI;
    d = TP./TT;
    precision = precision/length(STATENAMES);
    recall = recall/length(STATENAMES);

    % Compute for F-measure
    fmeasure = (2*precision*recall)/(precision+recall);

    % save results to file
    thisDirectory = pwd;
    fid = fopen('hmm_result.txt', 'w');
    fprintf(fid, '%s \n\n\n',thisDirectory);
    fprintf(fid, '-----------------------------------------------------------------------------------------\n');
    fprintf(fid, 'Total test items: %d\n\n', length(newTestSeq));
    fprintf(fid, 'Confusion matrix:\n');
    fprintf(fid, [repmat('%d\t', 1,length(STATENAMES)) '\n\n'], m);
    fprintf(fid, 'Accuracy = %f\n',accuracy);
    fprintf(fid, 'Precision = %f\n',precision);
    fprintf(fid, 'Recall = %f\n',recall);
    fprintf(fid, 'F-measure = %f\n', fmeasure);
    fprintf(fid, '-----------------------------------------------------------------------------------------\n');
    fclose(fid);
    fprintf('Done testing for %s \n', thisDirectory);
    averageAccuracy = [averageAccuracy accuracy];
    averageFmeasure = [averageFmeasure fmeasure];
  end
end

subdircount
cd ..;
aveAccuracy = sum(averageAccuracy)/subdircount
aveFmeasure = sum(averageFmeasure)/subdircount


fid = fopen('result.txt', 'w');
fprintf(fid, '%s \n\n\n', currentDir);
fprintf(fid, '-----------------------------------------------------------------------------------------\n');
fprintf(fid, 'Average Accuracy = %f\n',aveAccuracy);
fprintf(fid, 'Average F-measure = %f\n', aveFmeasure);
fprintf(fid, '-----------------------------------------------------------------------------------------\n');
fclose(fid);
