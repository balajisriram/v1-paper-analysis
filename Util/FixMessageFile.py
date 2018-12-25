import os
import pdb
import pprint
ppr = pprint.PrettyPrinter(indent=2)

def get_index(line):
    if 'TrialStart' in line:
        x = line.find('TrialStart')
        return line[0:x-1]
    elif 'TrialEnd' in line:
        x = line.find('TrialEnd')
        return line[0:x-1]

def fix_message_file(msg_file,start_trial_num):
    base,file = os.path.split(msg_file)
    
    # create new message file
    new_file_name = 'messages.NEW.events'
    new_file = os.path.join(base,new_file_name)
    
    with open(msg_file,'r',errors='ignore') as f:
        lines = f.readlines()
    
    #pdb.set_trace()
    expected_trial_num = start_trial_num
    prev_index = []
    prev_line = []
    curr_line_is_trial_start = False
    prev_line_is_trial_start = False
    curr_line_is_trial_end = False
    prev_line_is_trial_end = False
    
    is_first_line = True
    
    new_lines = []
    
    for line in lines[4:]:
        index = get_index(line)
        if line == prev_line: continue
        if 'TrialStart' in line:
            curr_line_is_trial_start = True
            curr_line_is_trial_end = False
        elif 'TrialEnd' in line: 
            curr_line_is_trial_start = False
            curr_line_is_trial_end = True
        
        if curr_line_is_trial_start and prev_line_is_trial_start:
            print('Two Trial Starts consecutively')
            # verify that the indices are the same
            if prev_index == index:
                continue
            else: 
                print('two trial start lines but different indices')
                pdb.set_trace()
            
        elif (curr_line_is_trial_start and prev_line_is_trial_end) or (curr_line_is_trial_start and is_first_line):
            print('Prev was TrialStart, this was TrialEnd')
            # print out the 'TrialStart' line after checking for expected_trial_num
            if 'TrialStart::{0}'.format(expected_trial_num) in line:
                new_line = '{0} TrialStart::{1}'.format(index,expected_trial_num)
                is_first_line = False
        elif curr_line_is_trial_end and prev_line_is_trial_start:
            print('Prev was TrialEnd, this was TrialStart')
            # print out the 'TrialEnd' line
            new_line = '{0} TrialEnd'.format(index)
        else:
            print('Huh? Ignoring')
            continue
        
        
        expected_trial_num = expected_trial_num+1
        prev_index = index
        prev_line = line
        prev_line_is_trial_start=curr_line_is_trial_start
        prev_line_is_trial_end=curr_line_is_trial_end
        new_lines.append(new_line)
        
    with open(new_file,'w') as f:
      for line in new_lines:
          f.write(line)
    print('done')
    
if __name__=='__main__':
    fix_message_file(r'C:\Users\bsriram\Desktop\Data_V1Paper\Inspected\bas070_2015-08-07_15-03-14\messages.events',7316)
    #get_index('2394872178463298 TrialStart')