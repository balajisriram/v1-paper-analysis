import OpenEphys
import os

folder = "C:\\Users\\bsriram\\Desktop\\Data_V1Paper\\DetailsProcessedPhysOnly\\bas070_2015-08-21_11-50-57";
events_file = os.path.join(folder,"all_channels.EVENTS")
print(events_file)
print()
print()
data = OpenEphys.load(events_file)

import pdb
pdb.set_trace()
