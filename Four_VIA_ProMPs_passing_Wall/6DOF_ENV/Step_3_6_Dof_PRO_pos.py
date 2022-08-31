
import matplotlib as mpl
import matplotlib.pyplot as plt
import sys
import numpy as np # 행렬다루기 as 머이런건 애칭 allas name ,라이브러리나 다른 소스코드 쉽게불러오기ㅋ
import math # 제곱, 제곱근 등의 기초 사칙연산 및 쉬운함수들 sin cos 등
import glob
import timeit
from Step_1_6_Dof_DTW_NP import get_Normalized_Quaternion_from_euler_deg
from Step_2_6_Dof_Make_Initial_ProMPs import Quat_2_deg

## 먼저 position에 대해서
### --1) 초기 데모데이터에서 position부분만 가져온다 이때 promps는 2개를 생성 ㅎ ---> 각각 나눠서 따로 합쳐주도록합시다.
### --2) 그 position부분에 대해서만 ProMPs 학습 , via point도 x,y,z에 대해서만
### --3) 궤적의 길이를 담당해서 먼저하는것임
### --4) 학습다된 pos의 weight부분을 저장한다.
#### step 4에 이어서 ㅎ

if __name__ == '__main__':
    tic = timeit.default_timer()

    np.random.seed(42)

    initial_demo_path = '/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Demo_data/Expert_data_np_7dof_quat'

    ws_x_scale = 25;
    ws_y_scale = 30;
    ws_z_scale = 25;

    via1 = np.array([-2 / ws_x_scale, -3.81 / ws_y_scale, -9.18 / ws_z_scale])
    via1_quat = get_Normalized_Quaternion_from_euler_deg(62.34, -1.782, -35.842)
    via1_check_deg = Quat_2_deg(via1_quat[3], via1_quat[0], via1_quat[1], via1_quat[2])
    via1_pose = np.concatenate((via1, via1_quat), None)

    via2 = np.array([-1.483 / ws_x_scale, 0 / ws_y_scale, -1.16 / ws_z_scale])
    via2_quat = get_Normalized_Quaternion_from_euler_deg(-8.201, 41.702, -9.882)
    via2_check_deg = Quat_2_deg(via2_quat[3], via2_quat[0], via2_quat[1], via2_quat[2])
    via2_pose = np.concatenate((via2, via2_quat), None)

    via3 = np.array([1.063 / ws_x_scale, 0 / ws_y_scale, 1.144 / ws_z_scale])
    via3_quat = get_Normalized_Quaternion_from_euler_deg(-42.503, 33.64, -20.282)
    via3_check_deg = Quat_2_deg(via3_quat[3], via3_quat[0], via3_quat[1], via3_quat[2])
    via3_pose = np.concatenate((via3, via3_quat), None)

    via4 = np.array([4.26 / ws_x_scale, 4.76 / ws_y_scale, 3.56 / ws_z_scale])
    via4_quat = get_Normalized_Quaternion_from_euler_deg(-50.836, 64.567, -58.478)
    via4_check_deg = Quat_2_deg(via4_quat[3], via4_quat[0], via4_quat[1], via4_quat[2])
    via4_pose = np.concatenate((via4, via4_quat), None)

    Via_points = np.transpose(np.vstack((via1_pose[:7], via2_pose[:7], via3_pose[:7], via4_pose[:7])))  #:7 벡터 요소7개 다 인코딩한다는거임# [3X4]행렬
    #potent_Via_points=np.transpose( np.vstack((via1[:3], (via2[:3]+ via3[:3])/2, via4[:3])))
    # format_promp from initial trj
    #Init_relevnace_mat=np.load( Learned_model_folder + "/" +"initial_relevance.npy")
    # 교시데이터 추출
    demo_np_datas = []  # list
    all_demo_files = glob.glob(initial_demo_path + "/*.npy")
    for i in all_demo_files:  # 난 저 교시 폴더안에 있는 데이터 다 끄집어내서 하나의 행렬에 다 저장 ㅋ
        ith_data = np.load(i)
        demo_np_datas.append(ith_data)

    ########## Recall Initial ProMPs and Parameters######################################
    # Feature_map_num_index=10
    # Promp_set = MIP(demo_np_datas, Feature_map_num_index)  # promp 인스턴스 생성
    MoC_expertData = GME(dofs=3, Max_length=200, start_via_point=Via_points[:, 0],end_via_point=Via_points[:, 3])



    # format_promp from initial trj
    Dimensions = 3  # 3차원
    Num_via_points =  Via_points.shape[1]  # viapoint 4개
    #Num_via_points =potent_Via_points.shape[0]  # viapoint 3개
    #Promp_set = MIP(Demo_datas=MoC_expertData, n_dim=Dimensions, num_via_point=Num_via_points)  # promp 인스턴스 생성
    Promp_set = MIP(Demo_datas=demo_np_datas, n_dim=Dimensions, num_via_point=Num_via_points)  # promp 인스턴스 생성

    ######### Set random seed ################ np.random이 말이 랜덤이지 뭔가 좀 구린데

    np.random.seed(42)  # #-> random시 나오는거 저장할려고 seed 생성 42인거는 크게 의미없는데 자주 쓰임

    #######Learn relevance function ###################################################3#
    # 150 iterations with 200 trajectory samples <- 이건 논문의 6축 구멍통과 예쩨
    N_iterations =150

    n_samples = 200  # 최종promp에서 나올건 200개의 궤적다발이라서 이렇게 선정!!
    N_relfun = 6  # 4개 경유점 + length, jerk 각각 1개씩 총 6개
    ###### Main_PRO_GRP##################

    ### 지금 환경에 대해서 policy최적화 강화학습
    Learnt_pol,rewards,check_w_mu_check,check_w_Sigma_check,Weights_rel_check,Relvance_check=Optimize_Trj_distribution_PRO(promp=Promp_set,
                                                                                          via_points_Mat=Via_points,
                                                                                          n_iter=N_iterations,
                                                                                          N_trj=n_samples,
                                                                                          weight_index_num=Promp_set.cal_weight_index_num
                                                                                          )

    ### 위에서 학습한 policy 정보를 통해서 trj 최종추출
    Weight_final_samples, \
    traj_final_samples = Sample_trj_from_determined_ProMPs(N_TRJ_SAMPLE=n_samples, mu_W=Learnt_pol.mu_w,
                                                               Sigma_W=Learnt_pol.Sigma_w,
                                                               Block_psi_cal=Learnt_pol.Blck_PSI, n_dof=Promp_set.dof,
                                                               max_length=Promp_set.max_length,
                                                               )


    Learned_model_folder = r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\3D_Moving_cube\Four_Via_Points\Learned_model_parameter_3dof_four_via_points"  # /\no상관

    np.save(Learned_model_folder + "/Final_promp_trj.npy", traj_final_samples)  # 완전 극초기모델임

    ### Show_results#############################################
    # 3차원 입체 그래프 그리기 위해 필요 아래
    ####그림1 # 학습 결과 궤적 200개 출력 ######################
    mpl.rcParams['legend.fontsize'] = 10
    plt.figure(1)
    ax = plt.axes(projection='3d')

    ax.plot(Via_points[0,0],Via_points[1,0],Via_points[2,0], label="start", color='yellow', marker='o', markersize=12)
    ax.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="mid1", color='red', marker='o', markersize=5)
    ax.plot(Via_points[0, 2], Via_points[1, 2], Via_points[2, 2], label="mid2", color='red', marker='o', markersize=5)
    ax.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='blue', marker='o', markersize=12)

    # ax.plot(data_phase_array_x,data_phase_array_y,data_phase_ㅎarray_z,label=str(file_count))

    # ax.plot(Via_points[0,0], Via_points[1,0], Via_points[2,0], label="start", color='yellow', marker='o', markersize=12)
    #
    # ax.plot(Via_points[0,1],Via_points[1,1], Via_points[2,1], label="end", color='red', marker='s', markersize=5)

    #ax.plot(Via_points[0,3], Via_points[1,3], Via_points[1,3], label="end", color='blue', marker='o', markersize=12)
    # plot_distribution
    for i in range(0,200):
        # Initial_ProMPs.max_length
        ax.plot(traj_final_samples[i, 0:200].flatten(),traj_final_samples[i, 200:400].flatten(),traj_final_samples[i, 400:600].flatten(), label=str(i))

    plt.savefig(Learned_model_folder + "/" + "trk_distribution.png")
    ####그림2 # 학습 결과 평균 궤적 ######################

    plt.figure(2)
    # plot_actual_guidance_trj_cov
    ax1 = plt.axes(projection='3d')
    # ax.plot(data_phase_array_x,data_phase_array_y,data_phase_ㅎarray_z,label=str(file_count))
    ax1.plot(Via_points[0, 0], Via_points[1, 0], Via_points[2, 0], label="start", color='yellow', marker='o',
            markersize=12)
    ax1.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="mid1", color='red', marker='o', markersize=5)
    ax1.plot(Via_points[0, 2], Via_points[1, 2], Via_points[2, 2], label="mid2", color='red', marker='o', markersize=5)
    ax1.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='blue', marker='o', markersize=12)

    # ax1.plot(Via_points[0, 0], Via_points[1, 0], Via_points[2, 0], label="start", color='yellow', marker='o', markersize=12)
    # ax1.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="end", color='red', marker='s', markersize=5)
    # ax1.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='red', marker='s', markersize=5)

    #ax1.plot(Via_points[0, 3], Via_points[1, 3], Via_points[1, 3], label="end", color='blue', marker='o', markersize=12)

    # for i in range(0,Initial_ProMPs.max_length):
    #     #Initial_ProMPs.max_length
    #     ax1.plot( traj_init_samples_mean_xyz[0,3*i],traj_init_samples_mean_xyz[0,3*i+1],traj_init_samples_mean_xyz[0,3*i+2],label=str(i),color='black',marker='*',markersize=2 )
    mean_trj = np.mean(traj_final_samples, axis=0)[np.newaxis]
    ax1.plot(mean_trj[0, 0:200].flatten(),
             mean_trj[0, 200:400].flatten(),
             mean_trj[0, 400:600].flatten(),
             label=str(i))
    plt.savefig(Learned_model_folder + "/" + "mean_trj.png")
    ####그림3 # 학습 결과 궤적의 표준편차 ######################
    plt.figure(3)

    ax2 = plt.axes(projection='3d')
    #var_trj_200_xyz = np.sqrt(np.diag(MIP.Get_cov_from_signle_Mat(traj_final_samples)))  # 대각가져오기!
    var_trj_200_xyz_renew=np.std(traj_final_samples,axis=0)[np.newaxis]
    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz_renew[0,i] * 6)

    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz_renew[0,i+200] * 6)

    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz_renew[0,i+400] * 6) ## 눈에 보이게 만 참고

    ax2.plot(Via_points[0, 0], Via_points[1, 0], Via_points[2, 0], label="start", color='yellow', marker='o',
            markersize=12)
    ax2.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="mid1", color='red', marker='o', markersize=5)
    ax2.plot(Via_points[0, 2], Via_points[1, 2], Via_points[2, 2], label="mid2", color='red', marker='o', markersize=5)
    ax2.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='blue', marker='o', markersize=12)
    plt.savefig(Learned_model_folder + "/" + "var_trj.png")


    WI=np.array(range(Promp_set.cal_weight_index_num))

    ####그림4 # 학습 결과  relevance_x######################

    plt.figure(4) # relevance_ x
    plt.title("Relevance_x")
    plt.xlabel ("Weight_index")
    plt.ylabel("relevance")

    for i in range(0, N_relfun):
        plt.plot(WI,Learnt_pol.relevance_mat[i,range(0,Promp_set.cal_weight_index_num)] ,label="rel_"+str(i))
    plt.legend()
    plt.savefig(Learned_model_folder + "/" + "Relevance_x.png")

    ####그림5 # 학습 결과  relevance_ㅛ######################
    plt.figure(5)  # relevance_ y
    plt.title("Relevance_y")
    plt.xlabel("Weight_index")
    plt.ylabel("relevance")
    #for i in range(0,potent_Via_points.shape[0]):
    for i in range(0, N_relfun):
        plt.plot(WI,Learnt_pol.relevance_mat[i,range(Promp_set.cal_weight_index_num,Promp_set.cal_weight_index_num*2)] ,label="rel_"+str(i))
    plt.legend()
    plt.savefig(Learned_model_folder + "/" + "Relevance_y.png")
    ####그림6 # 학습 결과  relevance_ㅋ######################
    plt.figure(6)  # relevance_ z
    plt.title("Relevance_z")
    plt.xlabel("Weight_index")
    plt.ylabel("relevance")
    for i in range(0, N_relfun):
        plt.plot(WI,Learnt_pol.relevance_mat[i,range(Promp_set.cal_weight_index_num*2,Promp_set.cal_weight_index_num*3)] ,label="rel_"+str(i))
    plt.legend()
    plt.savefig(Learned_model_folder + "/" + "Relevance_z.png")

    ####그림7 # 학습 결과 _ 학습하면서 via point 1 까지 거리######################

    plt.figure(7)
    plt.title("Start_via_distance")
    plt.xlabel("Iteration")
    plt.ylabel("Euclidean_distance")
    plt.plot(range(0,N_iterations), np.mean(Learnt_pol.all_objectives_vals[0][range(0,N_iterations), :], axis=1))
    plt.fill_between(range(0,N_iterations),
                     np.min(Learnt_pol.all_objectives_vals[0][range(0,N_iterations), :], axis=1),
                     np.max(Learnt_pol.all_objectives_vals[0][range(0, N_iterations), :], axis=1),
                     alpha=0.5
                     ) # alpha는 투명도,0.5= 50%라고 선정
    plt.savefig(Learned_model_folder + "/" + "Start_via_distance.png")
    ####그림8 # 학습 결과 _ 학습 하면서  via point 2 까지 거리######################
    plt.figure(8)
    plt.title("Infront_hole_via_distance")
    plt.xlabel("Iteration")
    plt.ylabel("Euclidean_distance")
    plt.plot(range(0, N_iterations), np.mean(Learnt_pol.all_objectives_vals[1][range(0, N_iterations), :], axis=1))
    plt.fill_between(range(0, N_iterations),
                     np.min(Learnt_pol.all_objectives_vals[1][range(0, N_iterations), :], axis=1),
                     np.max(Learnt_pol.all_objectives_vals[1][range(0, N_iterations), :], axis=1),
                     alpha=0.3
                     )  # alpha는 투명도,0.3= 30%라고 선정
    plt.savefig(Learned_model_folder + "/" + "Infront_hole_via_distance.png")

    plt.figure(9)
    plt.title("Back_hole_via_distance")
    plt.xlabel("Weight_index")
    plt.ylabel("Euclidean_distance")
    plt.plot(range(0, N_iterations), np.mean(Learnt_pol.all_objectives_vals[2][range(0, N_iterations), :], axis=1))
    plt.fill_between(range(0, N_iterations),
                     np.min(Learnt_pol.all_objectives_vals[2][range(0, N_iterations), :], axis=1),
                     np.max(Learnt_pol.all_objectives_vals[2][range(0, N_iterations), :], axis=1),
                     alpha=0.3
                     )  # alpha는 투명도,0.3= 30%라고 선정

    plt.savefig(Learned_model_folder + "/" + "Back_hole_via_distance.png")
    ####그림9 # 학습 결과 _ via point 3 까지 거리######################
    plt.figure(10)
    plt.title("End_via_distance")
    plt.xlabel("Iteration")
    plt.ylabel("Euclidean_distance")
    plt.plot(range(0, N_iterations), np.mean(Learnt_pol.all_objectives_vals[3][range(0, N_iterations), :], axis=1))
    plt.fill_between(range(0, N_iterations),
                     np.min(Learnt_pol.all_objectives_vals[3][range(0, N_iterations), :], axis=1),
                     np.max(Learnt_pol.all_objectives_vals[3][range(0, N_iterations), :], axis=1),
                     alpha=0.3
                     )  # alpha는 투명도,0.3= 30%라고 선정

    plt.savefig(Learned_model_folder + "/" + "End_via_distance.png")
    ####그림10 # 학습 결과 _ 학습하면서 reward 추이######################
    plt.figure(11)
    ##compute mean ans std of the returns
    plt.title("Maximizing_Returns")
    plt.xlabel("iterations")
    plt.ylabel("Returns")
    rewards=rewards  # 학습 수렴한 이후임 ㅎ
    plt.plot(range(0,rewards.shape[1]), np.mean(rewards,axis=0).flatten()[range(0,rewards.shape[1])] )
    plt.fill_between(range(0, N_iterations),
                     np.min(rewards,axis=0).flatten()[range(0,rewards.shape[1])],
                     np.max(rewards,axis=0).flatten()[range(0,rewards.shape[1])],
                     alpha=0.3
                     )  # alpha는 투명도,0.3= 30%라고 선정
    #plt.plot(range(0, rewards.shape[1]), np.std(rewards, axis=0).flatten()[range(0, rewards.shape[1])])
    plt.savefig(Learned_model_folder+"/"+"Maximizing_Returns.png")
    plt.show()

    print('Finish! _ step4')  # PRO_GPR 다구함 ㅎ
    toc = timeit.default_timer()
    print('it takse' + str(toc - tic) + 'seconds')


