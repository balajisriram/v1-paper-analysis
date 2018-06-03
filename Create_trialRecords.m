location = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\SessionsWithLED\\bas081b_2017-07-28_20-31-03";
d=dir(fullfile(location,'OLD.trialRecords*.mat'));
temp = load(fullfile(location,d.name));
trialRecords = [];
currnum = 1;
for i = 1:length(temp.trialRecords)
  trialRecords(currnum).trialNumber = temp.trialRecords(i).trialNumber;
  trialRecords(currnum).refreshRate = temp.trialRecords(i).resolution.hz;
  trialRecords(currnum).stepName = temp.sessionLUT{temp.trialRecords(i).stepName};
  trialRecords(currnum).stimManagerClass = temp.sessionLUT{temp.trialRecords(i).stimManagerClass};
  trialRecords(currnum).trialManagerClass = temp.sessionLUT{temp.trialRecords(i).trialManagerClass};
  if strcmp(trialRecords(currnum).stimManagerClass,'afcGratings') 
    trialRecords(currnum).afcGratingType = temp.trialRecords(i).stimDetails.afcGratingType;
    trialRecords(currnum).stimulus = temp.trialRecords(i).stimDetails.chosenStim;
  elseif strcmp(trialRecords(currnum).stimManagerClass,'autopilotGratings')
    trialRecords(currnum).afcGratingType = 'notSpecified';
    stimulus.pixPerCyc = temp.trialRecords(i).stimDetails.pixPerCyc
    stimulus.driftfrequency = temp.trialRecords(i).stimDetails.driftfrequency
    stimulus.orientation = temp.trialRecords(i).stimDetails.orientation
    stimulus.phase = temp.trialRecords(i).stimDetails.phase
    stimulus.contrast = temp.trialRecords(i).stimDetails.contrast
    stimulus.maxDuration = temp.trialRecords(i).stimDetails.maxDuration
    stimulus.radius = temp.trialRecords(i).stimDetails.radius
    stimulus.annulus = temp.trialRecords(i).stimDetails.annulus
    stimulus.waveform = temp.trialRecords(i).stimDetails.waveform
    stimulus.radiusType = temp.trialRecords(i).stimDetails.radiusType
    stimulus.location = temp.trialRecords(i).stimDetails.location
    
    trialRecords(currnum).stimulus = stimulus;
    trialRecords(currnum).LEDON = temp.trialRecords(i).stimDetails.LEDDetails.LEDON;
    trialRecords(currnum).LEDIntensity = temp.trialRecords(i).stimDetails.LEDDetails.LEDON;
  
  else
    trialRecords(currnum).afcGratingType = 'NotAAFCGrating';
    trialRecords(currnum).stimulus = NaN;
    trialRecords(currnum).LEDON = 0;
    trialRecords(currnum).LEDIntensity = 0;
  endif
  
  currnum = currnum+1;
endfor
datestring = datestr(temp.trialRecords(1).date,'yyyymmdd');
name = sprintf("trialRecords_bas081b_%s_%d-%d.mat",datestring,trialRecords(1).trialNumber,trialRecords(end).trialNumber)
save(fullfile(location,name),'trialRecords')
