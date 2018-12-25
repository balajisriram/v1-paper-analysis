import os
folders = ["DetailsProcessedBehaved","DetailsProcessedPhysOnly","Detected","Inspected","Inspected VPM","Sorted"]
main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\"

for folder in folders:
    print()
    print()
    print(folder)
    for session_folder in os.listdir(os.path.join(main_folder,folder)):
        trialRecord_file = [f for f in os.listdir(os.path.join(main_folder,folder,session_folder)) if f.startswith("trialRecords")]
        print(session_folder,":",trialRecord_file)