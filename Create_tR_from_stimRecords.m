location = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\DetailsProcessedPhysOnly\\bas078_2016-05-24_15-48-03_1\\052416";
mouseID = 'bas078';
subloc = fileparts(location);
d=dir(fullfile(location,'stimRecords*.mat'));
seq = []
for i = 1:length(d)
  [s,e,te,m,t] = regexp(d(i).name,'stimRecords_(\d+)-');
  seq(end+1) = str2num(t{1}{1});
endfor

if length(unique(seq))~=length(seq)
  error('many trials have same trial number');
endif

[seq,order] = sort(seq);
d = d(order);

% now they are in order

% load and collect necessary data

trialRecords = [];
currnum = 1;
for i = 1:length(d)
  temp = load(fullfile(location,d(i).name));
  trialRecords(currnum).trialNumber = temp.trialNum;
  trialRecords(currnum).refreshRate = temp.refreshRate;
  trialRecords(currnum).stepName = temp.stepName;
  trialRecords(currnum).stimManagerClass = temp.stimManagerClass;
  trialRecords(currnum).trialManagerClass = temp.stimulusDetails.trialManagerClass;
  if strcmp(trialRecords(currnum).stimManagerClass,'afcGratings')
    trialRecords(currnum).afcGratingType = temp.stimulusDetails.afcGratingType;
    trialRecords(currnum).stimulus = temp.stimulusDetails.chosenStim;
  else
    trialRecords(currnum).afcGratingType = 'NotAAFCGrating';
    trialRecords(currnum).stimulus = NaN;
  endif
  
  trialRecords(currnum).LEDON = temp.stimulusDetails.LEDON;
  trialRecords(currnum).LEDIntensity = temp.stimulusDetails.LEDIntensity;
  
  currnum = currnum+1;
endfor
startTimeStr = temp.trialStartTime;
[s,e,te,m,t] = regexp(startTimeStr,'(\d+)T');
dateString = startTimeStr(s:e-1);
name = sprintf("trialRecords_%s_%s_%d-%d.mat",mouseID,datestring,trialRecords(1).trialNumber,trialRecords(end).trialNumber)
save(fullfile(subloc,name),'trialRecords')