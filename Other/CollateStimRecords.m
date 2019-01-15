foldername = 'D:\bas070\08032015';

d = dir(foldername);
d = d(~ismember({d.name},{'.','..'}));

trial_numbers = [];
for i = 1:length(d)
  [a,b] = regexp(d(i).name, '_(\d+)*','match','tokens');
  if length(a)>1
    error('weird');
  endif
  trial_numbers(end+1) = str2num(b{1}{1});
    
endfor

trial_numbers