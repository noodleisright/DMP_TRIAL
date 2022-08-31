import pandas as pd
import numpy as np
path = '/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Learned_Parameter'
certain_file=path+ "/"+"Final_promp_trj.npy"
data=np.load(certain_file)
save_directory='/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Learned_Parameter'
pd.DataFrame(data).to_csv(save_directory+ "/"+"Test_trj_final.csv",header=None,index=None )