import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math

import pandas as pd
from scipy.interpolate import interp1d
import glob
import timeit


def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if mag2 > tolerance:
        mag = math.sqrt(mag2)
        v = tuple(n / mag for n in v)
    return np.array(v)


def get_Normalized_Quaternion_from_euler_deg(roll,pitch,yaw):
    #DEG_2_RAD
    roll=roll*math.pi/180
    pitch = pitch * math.pi / 180
    yaw = yaw * math.pi / 180

    # 순서대로 x,y,z순이고 공식 참조해서 계산 https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    Original_Quaternion= [qx,qy,qz,qw ]
    Normalized_Quaternion=normalize(Original_Quaternion)
    return Normalized_Quaternion


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
    export_np_path='/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Demo_data/Expert_data_np_6dof'

    all_files=glob.glob(import_path+"/*.csv")
    new_axis=np.linspace(0,1,max_length)
    file_count=0
    iteration_count=0
    #모든열을같은길이로만들기
    for filename in all_files:
        data=pd.read_csv(filename)
        old_axis=np.linspace(0,1,np.column_stack([data.iloc[:,0],data.iloc[:,1]]).shape[0])#start,stop,num//array.shape[0]행의개수 --pandas 함수로 전체 행데이터 가져오기
        interp_robot_x = interp1d(old_axis,
                                  np.transpose(data.iloc[:, 0]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_y = interp1d(old_axis,
                                  np.transpose(data.iloc[:, 1]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_z = interp1d(old_axis,
                                  np.transpose(data.iloc[:, 2]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_rx = interp1d(old_axis,
                                   np.transpose(data.iloc[:, 3]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_ry = interp1d(old_axis,
                                   np.transpose(data.iloc[:, 4]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간
        interp_robot_rz = interp1d(old_axis,
                                   np.transpose(data.iloc[:, 5]))  # np.transpose--행과열을바꾸는거임,interp1d(x,y,z)--1차원보간

        data_phase_array_x = interp_robot_x(new_axis)/ws_x_scale
        data_phase_array_y = interp_robot_y(new_axis)/ws_y_scale
        data_phase_array_z = interp_robot_z(new_axis)/ws_z_scale
        data_phase_array_rx = interp_robot_rx(new_axis)
        data_phase_array_ry = interp_robot_ry(new_axis)
        data_phase_array_rz = interp_robot_rz(new_axis)
        ax.plot(data_phase_array_x,data_phase_array_y,data_phase_array_z,label=str(file_count))
        #np.save(r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\Expert_data_np_3dof"+"/"+str(file_count)+ "번째_데이터"+".npy",np.column_stack([data_phase_array_x,data_phase_array_y,data_phase_array_z]))
        np.save( export_np_path+"/"+str(file_count)+".npy",np.column_stack([data_phase_array_x,data_phase_array_y,data_phase_array_z,data_phase_array_rx,data_phase_array_ry,data_phase_array_rz]))
        # 우분투에서는 마찬가지로 저렇게!
        #np.save(export_path+"/"+str(file_count)+ "번째_데이터"+".npy",np.column_stack([data_phase_array_x,data_phase_array_y,data_phase_array_z]))
        file_count+=1

    export_quat_path = '/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Demo_data/Expert_data_np_7dof_quat'
    # 교시 당시 via points들

    ax.plot(-2/ws_x_scale ,-3.81/ws_y_scale ,-9.18/ws_z_scale ,label="start",color='yellow',marker='o',markersize=9)
    ax.plot(-1.483/ws_x_scale ,0/ws_y_scale ,-1.16/ws_z_scale ,label="end",color='red',marker='s',markersize=5)
    ax.plot(1.063/ws_x_scale ,0/ws_y_scale ,1.144/ws_z_scale ,label="end",color='red',marker='s',markersize=5)
    ax.plot(4.26/ws_x_scale ,4.76/ws_y_scale ,3.56/ws_z_scale ,label="end",color='blue',marker='o',markersize=9)

    all_files_np = glob.glob(export_np_path + "/*.npy")
    rep_qua = np.empty((max_length, 4,))
    rep_qua[:] = np.nan
    finalized_expert_data = np.empty((max_length, 7,))
    finalized_expert_data[:] = np.nan
    finalized_file_count = 1

    for filename in all_files_np:
        data_raw = np.load(filename)
        for i in range(0, max_length):  # a이상 b미만개수
            rep_qua[i] = get_Normalized_Quaternion_from_euler_deg(data_raw[i, 3], data_raw[i, 4], data_raw[i, 5])
        data_raw = np.delete(data_raw, slice(3, 6),1)  # 예를들어 slice (1,3)은 그배열의 처음부터 2번째 열에서 (1부터 갯수 샐때)3번째열까지 자른다 즉 2,3열 자르는거야
        finalized_expert_data = np.column_stack((data_raw, rep_qua))
        np.save(export_quat_path + "/" + str(finalized_file_count) + "번째_데이터" + ".npy", finalized_expert_data)
        #data_frame_csv=pd.DataFrame(finalized_expert_data)
        #data_frame_csv.to_csv('/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Mean_ProMPs_SAVE'+'/'+ str(finalized_file_count),header=False,index=False)

        finalized_file_count += 1

    plt.savefig(export_np_path+"/"+'test.png')

    plt.show(block=True)


    toc=timeit.default_timer()
    print('Finish! _ step1')
    print( 'it takse'+str(toc-tic)+'seconds' )
