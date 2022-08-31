import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import pandas as pd
from scipy.interpolate import interp1d
import glob
import timeit



if __name__ == '__main__':
    tic = timeit.default_timer()
    mpl.rcParams['legend.fontsize']=10
    fig=plt.figure()

    ax=plt.axes(projection='3d')

    ws_x_scale = 25;
    ws_y_scale = 30;
    ws_z_scale = 25;

    max_length=1000#전체trajectory를형성하는point수
    #특정파일에있는모든csv파일을우선np화하기!!
    #path= r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\Expert_data_6dof"  #/\no상관
    # 우분투에선 이렇게 -> 터미널에서 pwd로 확인후 ctrl shift c ->v 해야해
    import_path='/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/CSV_Expert_data'
    export_path='/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/3DOF_ENV/Demo_data/Expert_data_np_3dof'

    all_files=glob.glob(import_path+"/*.csv")
    new_axis=np.linspace(0,1,max_length)
    file_count=0
    iteration_count=0
    #모든열을같은길이로만들기
    for filename in all_files:
        data=pd.read_csv(filename)
        old_axis=np.linspace(0,1,np.column_stack([data.iloc[:,0],data.iloc[:,1]]).shape[0])#start,stop,num//array.shape[0]행의개수 --pandas 함수로 전체 행데이터 가져오기
        interp_robot_x=interp1d(old_axis,np.transpose(data.iloc[:,0]))#np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_y=interp1d(old_axis,np.transpose(data.iloc[:,1]))#np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_z=interp1d(old_axis,np.transpose(data.iloc[:,2]))#np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        data_phase_array_x=interp_robot_x(new_axis)/ws_x_scale
        data_phase_array_y=interp_robot_y(new_axis)/ws_y_scale
        data_phase_array_z=interp_robot_z(new_axis)/ws_z_scale
        ax.plot(data_phase_array_x,data_phase_array_y,data_phase_array_z,label=str(file_count))
        #np.save(r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\Expert_data_np_3dof"+"/"+str(file_count)+ "번째_데이터"+".npy",np.column_stack([data_phase_array_x,data_phase_array_y,data_phase_array_z]))
        # 우분투에서는 마찬가지로 저렇게!
        np.save(export_path+"/"+str(file_count)+ "번째_데이터"+".npy",np.column_stack([data_phase_array_x,data_phase_array_y,data_phase_array_z]))
        file_count+=1



    ax.plot(-2/ws_x_scale ,-3.81/ws_y_scale ,-9.18/ws_z_scale ,label="start",color='yellow',marker='o',markersize=9)
    ax.plot(-1.483/ws_x_scale ,0/ws_y_scale ,-1.16/ws_z_scale ,label="end",color='red',marker='s',markersize=5)
    ax.plot(1.063/ws_x_scale ,0/ws_y_scale ,1.144/ws_z_scale ,label="end",color='red',marker='s',markersize=5)
    ax.plot(4.26/ws_x_scale ,4.76/ws_y_scale ,3.56/ws_z_scale ,label="end",color='blue',marker='o',markersize=9)

    plt.savefig(export_path+"/"+'test.png')
    plt.show(block=True)


    toc=timeit.default_timer()
    print('Finish! _ step1')
    print( 'it takse'+str(toc-tic)+'seconds' )
