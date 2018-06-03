import os

main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\DetailsProcessedBehaved\\"
print("Behavior Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)
    
    
main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\DetailsProcessedPhysOnly\\"
print("Physiology Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)

    
main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\Inspected\\"
print("Inspected Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)
    
main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\Inspected VPM\\"
print("VPM Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)
    
    
main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\Detected\\"
print("Detected Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)


main_folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\BadSessions\\"
print("Bad Sessions....")
for folder in os.listdir(main_folder):
    prb_file = [f for f in os.listdir(os.path.join(main_folder,folder)) if '.prb' in f]
    print(folder,':',prb_file)
