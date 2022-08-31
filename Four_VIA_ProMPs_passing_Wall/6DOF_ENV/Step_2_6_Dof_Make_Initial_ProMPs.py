import numpy as np # 행렬다루기 as 머이런건 애칭 allas name ,라이브러리나 다른 소스코드 쉽게불러오기ㅋ
import scipy as sp # 고급 통계 계산
import math # 제곱, 제곱근 등의 기초 사칙연산 및 쉬운함수들 sin cos 등
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import timeit
import math
from Step_1_6_Dof_DTW_NP import get_Normalized_Quaternion_from_euler_deg




class ProMPs : # (weight의 평균,분산 | PSI_Matrix ) 라는 pickle파일로 저장!!

    def __init__(self,Demo_datas,n_dim,num_via_point):
        self.max_length =1000 # 생성할 궤적의 point 수
        self.noise=1e-5 # tau=psi*w+noise
        self.regularizer=1e-12 #hyperparameter lamda in linear ridge regression-> determine weight components
        self.demo_trial_num=10 # 교시데이터 수
        self.dof=7 # 3축 포지션+쿼터니언4요소 어차피 궤적은 x,y,z 니
        self.bfs_sigma= 0.0286  #width of basis function ==1.06*std*1000^-1/5 std는 젤 큰걸로 넣는다.
        self.num_centers_outside_range=2
        # self.cal_weight_index_num =int(num_via_point*3*math.pow(2,n_dim-2))
        self.cal_weight_index_num = 30
        self.num_basis = self.cal_weight_index_num  # gaussian basis function의 개수
        self._dt=1/self.max_length # 1e-3
        self._bfs_centres = np.linspace(0, 1, self.num_basis) #centers of the basis function 0~1을 30칸으로 나뉨

        self.phase_speed=1
        #Phase_variable
        Dz=np.ones(self.max_length)*self.phase_speed
        z=np.cumsum(Dz)*self._dt
        DDz=np.diff(Dz)/self._dt
        DDz=np.hstack([DDz,DDz[-1]])
        DDDz = np.diff(Dz) / self._dt
        self._z=z
        self._Dz = Dz
        self._DDz = DDz
        self._DDDz = DDDz

        #self.cal_weight_index_num= self.dof^n_dim+self.dof*num_via_point #trj_pose_dof^dimension +traj_pose_dof*via_point_num
        # 헐 print(4^3) 이면 7이 나옴 c랑 다르게 math.pow를 써야하네 , 혹은 **인데 직관적이지 못하네 ㅎ
        # PSI MATRIX 생성 - Normalized Gaussian basis function임
        PSI_matrix = self.get_PSI_mat(self.num_basis, self.max_length) #10행, 200열--> #21행, 200열
        # compute the basis functions (이거 7자유도에서 저 regular term 넣는것도 해결하고 적용하자 젠
        # #se장lf._Phi, self._PhiD, self._PhiDD = self.generate_basis_function(phase_z=self._z,
        #                                                                   phase_zd=self._Dz,
        #                                                                   phase_zdd=self._DDz,
        #                                                                                )
        # # Weight*PSI= Trajectory distriubtion임 block_psi로  (psi o 0; 0 psi 0 ; 0 0 psi )꼴로 해야 weight 차원과 연산되어 trjectory distribution나옴!!
        # Trajectory sample from promp 되려면
        # N_trj 여기선 200개 즉 궤적 분포로 부터 200개의 궤적이 출력이 된다.
        # 그 궤적분포의 차원은 200(1000_x |1000_y|1000_z|1000_qx |1000_qy|1000_qz|1000_qw)로 200행 7000열이 된다.
        self.block_PSI = sp.linalg.block_diag(*[PSI_matrix,PSI_matrix,PSI_matrix,PSI_matrix,PSI_matrix,PSI_matrix,PSI_matrix])  # 200*3행,21*3열 ex) block_diag (*[A,A]) = A 0 ; 0 A
        # Weight Matrix 생성 <-- 1trj 대해서 [1 x N*n_dof] =1자유도당 10개의 basis function 할당함 그게 3개고 그러면 n개의 trj를 출력할거면
        # ==> weight matrix를 [ n x N*n_dof]가 나와야함
        self.weight_matrix = np.zeros((self.demo_trial_num,self.num_basis * self.dof), dtype="float") # 10 행 30열 (교시횟수 ,basis_feature 수 *자유도)

        for i in range(self.demo_trial_num): #weight 행렬 차원 생성
            self.weight_matrix[i,:] = self.lin_ridge_reg(PSI_matrix, Demo_datas[i], self.regularizer)# i번째 열 다할당/가져오기
            print("processing_"+str(i)+"th_trj_weight_row")

        #mu_w=sp.mean(weight_matrix,axis=1) # axis=0은 행에 대한 , axis=1은 열에 대한 --> 근데 이제 잘안쓰지 np.mean으로 ㅎ
        self.mu_w=np.transpose(np.mean(self.weight_matrix,axis=0).reshape(1,self.num_basis*self.dof)) # (30,) 튜플 아니라 직관적인 배열형태로 [21x1] --열에 대한 평균 --> [1x21]
        self.Sigma_INN_Hyper_lamda=1e-7
        # self.Sigma_w=np.cov(self.weight_matrix) # 이거 오류
        self.Sigma_w = Get_cov_from_signle_Mat(self.weight_matrix)
        self.Sigma = np.identity(self.Sigma_w.shape[0]) * self.Sigma_INN_Hyper_lamda

    def generate_basis_function(self, phase_z, phase_zd, phase_zdd):


        # basis functions
        phase_minus_centre = np.array(map(lambda x: x - self._bfs_centres, np.tile(phase_z, (self.num_basis, 1)).T)).T

        # basis function
        Phi = np.exp(-0.5 * np.power(phase_minus_centre / self.bfs_sigma, 2)) / (np.sqrt(2 * np.pi) * self.bfs_sigma)

        # first derivative of basis function
        PhiD = np.multiply(Phi, -phase_minus_centre / (self.bfs_sigma ** 2))


        # second derivative of basis function
        PhiDD = Phi / (-self.bfs_sigma ** 2) + np.multiply(-phase_minus_centre / (self.bfs_sigma ** 2), PhiD)


        # for normalization purposes
        sum_bfs = np.sum(Phi, axis=0)
        sum_bfsD = np.sum(PhiD, axis=0)
        sum_bfsDD = np.sum(PhiDD, axis=0)

        # normalize
        PhiD_normalized = ((PhiD * sum_bfs - Phi * sum_bfsD) * 1. / np.power(sum_bfs, 2))

        Phi_normalized = Phi / sum_bfs[None, :]

        tmp1 = Phi * (2 * np.power(sum_bfsD, 2) - np.multiply(sum_bfs, sum_bfsDD))

        tmp2 = tmp1 + PhiDD * np.power(sum_bfs, 2) - 2 * PhiD * sum_bfs * sum_bfsD

        # normalize acceleration
        PhiDD_normalized = tmp2 * (1. / (np.power(sum_bfs, 3)))

        # adding phase dependency
        PhiDD_normalized = PhiDD_normalized * np.power(phase_zd, 2) + PhiD_normalized * phase_zdd
        PhiD_normalized = PhiD_normalized * phase_zd

        PhiDDD_normalized=0



        return Phi_normalized, PhiD_normalized,PhiDD_normalized

    def get_PSI_mat(self, n_basis, max_Time_step): # PSI 행렬 만들기
        # gaussian basis function = trjectories Feature=basis function


        def gabafun(basis_index,phase_index):
            return np.exp(-1/2 *math.pow( (phase_index - (basis_index - 1) * max_Time_step / (n_basis - 1)) ,2 )/ max_Time_step)

        # def gabafun(basis_index,phase_index):
        #     return np.exp(-1/2 *math.pow( (phase_index - (basis_index - 1) * max_Time_step / (n_basis - 1)) ,2 )/ max_Time_step)

        gaba_matrix=np.zeros((max_Time_step,self.num_basis),dtype="float") # 200행 10열의 차원
        # gabafun 이 phase_index-(basis_index-1) 형태라 basis_index는 1~10 , phase index는 1~200 형식으로 들어가야함 물론 코딩에서는 끝숫자 -1
        for time_step in range(1,max_Time_step+1): # c++에서 i=0 ; for i< array.length i++ 과 같음 --> range(5) 면 0~4임 ㅋ =range(0,5)
            for n_basis_index in range(1,self.num_basis+1):
                gaba_matrix[time_step-1,n_basis_index-1]=gabafun(n_basis_index,time_step) #역행렬,전치행렬 아니야 gaba_fun 형태때문에 저래요

        normalizer = np.sum(gaba_matrix, axis=1) # 열별로 전부 더하기 -- 전체  phase 200개 데이터의 총합 =크기알려고
        Gaba_matrix = np.divide(gaba_matrix, normalizer.reshape(np.size(normalizer, 0), 1)) # 이로써 정규화 --어차피 feature map 형태만 필요

        return Gaba_matrix # 1000행 30열이 만들어짐


    # Linear ridge regression for computing ProMP weights
    def lin_ridge_reg(self, feature_input, target_trj,hyper_lamda):
        # feature_input=PSI_MATRIX is a (T, N) matrix(정확히는 PSI'). target_data is a (max_trj, trial_num=N matrix =weight 의 차원임
        PSI_SAM=feature_input # 200행 10열
        PSI_T_X_PSI=np.dot(np.transpose(PSI_SAM),PSI_SAM ) #10행 10열 #m행 k열 * k행 n열 = m행 n열 - 즉 k열 k행으로 같아야 연산가능 # dot안해고 *,product하면 요소별로 곱해짐 근데 행렬 크기 다르면 망함.
        #PSI_jerk=feat_ddd
        #PSI_jerk_X_PSI_jerk=np.dot(np.transpose(PSI_jerk),PSI_jerk)
        PSI_T_X_PSI_ADD_LAMDA_I_NN=(PSI_T_X_PSI+hyper_lamda*np.identity(np.size(PSI_T_X_PSI,0))) #10행 10열
        #PSI_T_X_PSI_ADD_LAMDA_I_NN = (PSI_T_X_PSI + hyper_lamda*PSI_jerk_X_PSI_jerk)
        weights_row=( np.dot(np.dot( np.linalg.inv(PSI_T_X_PSI_ADD_LAMDA_I_NN) ,np.transpose(PSI_SAM)),target_trj )).reshape((1,self.num_basis*self.dof),order='F') # A*B*C=A*(B*C)=(A*B)*C  ㅋ [10*3]=10행 3열 # 1자유도에 우리가 선정한 Featuremap_basis=10을 부여
        #weights_row = np.dot( np.dot(np.linalg.inv((np.dot(np.transpose(feature_input), feature_input) + hyper_lamda * np.identity(np.size(np.dot(np.transpose(feature_input), feature_input),0)))),np.transpose(feature_input)),np.transpose(target_trj))

        return weights_row # 각자유도별로 weight 10개씩 할당한다는 뜻 -PSI basis가 10개니 30행 1열 즉 30개의 weight이지만 실제 차원은 weight matrix로 해야함---> 근데 나는 최종 폼때문에 1*30로함

    def Sample_weights_and_trj(self,N_trj,Sigma_weight=None,rel=False):


        # weight sample은 차원이 (n_samples, dofs*num_gaba)=(N_trj,n_dof*N)=10행 30열
        # trj Sample은 PSI*W 로 각자유도에 다 psi를 할당해야하기에 diag(PSI*n_DOF)임 즉 차원은 (N_trj,n_dof*n)=(100행 600열) (x|y|z|_200)
        if Sigma_weight is None: # 초기 promp는 아직 weight 읎다.0행렬 게다가 relevance learning 안해서 좀 지저분한값
            #mvnrnd : 다변량 정규분포 난수생성 = M,SIGMA 인 정규분포에서 임의의 값들을 도출
            if rel is False :
                weight_samples = np.random.multivariate_normal(self.mu_w.flatten(),self.Sigma,N_trj ) #relevance learning x
                print("******* Sample_weight_and_trj ******_non__****************")
            elif rel is True:
                weight_samples = np.random.multivariate_normal(self.mu_w.flatten(),self.Sigma_w,N_trj )  # relevance learning시
                print("******* Sample_weight_and_trj ******_rel_update__****************")
            trj_samples=np.dot(weight_samples,np.transpose(self.block_PSI)).reshape(N_trj,self.max_length*self.dof)

        else:
            weight_samples=np.random.multivariate_normal(self.mu_w.flatten(),Sigma_weight,N_trj)
            trj_samples=np.dot(weight_samples,np.transpose(self.block_PSI)).reshape(N_trj,self.max_length*self.dof)
            print("******* Sample_weight_and_trj ********additional__weight_rel**************")


        return weight_samples,trj_samples




# index의 복수형을 indices라 하네


def Get_cov_from_signle_Mat(ARR_MAT_FCOV ): # row =observation sets , column = features 뭐든 열을 기준으로 나는 feature을 맞춘다.
    Puncha =ARR_MAT_FCOV-np.mean(ARR_MAT_FCOV,axis=0)
     # 이게 모집단을 쓸거라 자유도 -1 안할거임
    Result_cov= np.dot(Puncha.T,Puncha)/(ARR_MAT_FCOV.shape[0])


    return Result_cov

def Known_env_cha_Vec(V1,V2,V3,V4):
    Known_env_vec_now = np.array([np.linalg.norm(V2 - V1), np.linalg.norm(V3 - V1),np.linalg.norm(V4 - V1)])

    #Known_env_vec_now=np.array([ np.linalg.norm(V2-V1),np.linalg.norm(V3-V1) ,np.linalg.norm(V4-V1)])

    return Known_env_vec_now

def Quat_2_deg(qw,qx,qy,qz ):
    """
            Convert a quaternion into euler angles (roll, pitch, yaw)
            roll is rotation around x in radians (counterclockwise)
            pitch is rotation around y in radians (counterclockwise)
            yaw is rotation around z in radians (counterclockwise)
            """
    t0 = +2.0 * (qw * qx + qy * qz)
    t1 = +1.0 - 2.0 * (qx * qx + qy * qy)
    roll_x = math.atan2(t0, t1)

    t2 = +2.0 * (qw * qy - qz * qx)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)

    t3 = +2.0 * (qw * qz + qx * qy)
    t4 = +1.0 - 2.0 * (qy *qy + qz * qz)
    yaw_z = math.atan2(t3, t4)

    return roll_x/math.pi*180, pitch_y/math.pi*180, yaw_z/math.pi*180  # in deg



def Sample_trj_from_determined_ProMPs(N_TRJ_SAMPLE,mu_W,Sigma_W,Block_psi_cal,n_dof,max_length ):  # 4개인수는 궤적분포로 부터 도출할 궤적수 , weight평균,분산, PSI형태임
    # 1D length N weight,
    # 2D Size NxN Cov
    # 평균이 matrix 형태인데 여기서 data_array 로 그냥 []이거정의는 list화니  mvnrnd 형태때문에 어쩔수없음

    W_sample=np.random.multivariate_normal(mu_W.flatten(),Sigma_W ,N_TRJ_SAMPLE) # M,SIGMA 정규분포로 부터 원하는 개수의 랜덤값 도출, 근데 다변량정규분포 난수생성 이함수는 1 rank의 mu만 받음
    # [N_Trj,n_dof*N] 인 1차원 벡터가나옴 이걸 2차원으로 확장해야 trj_sample사용가능
    # np,dot(a,b) = a*b 이순서고 np.dot(a,b)!=np.dot(b,a) 둘은 다르다. 내가 만들어준 차원 그대로 곱한다.
    #trj_samples = np.dot( W_sample, np.transpose(Block_psi_cal)) # w*psi=traj , 1차원 벡터끼리는 내적으로 하나의 스칼라 하지만 저렇게 배열끼리하면 행렬곱이됨 blkpsi는   [600x30] 계속 쓸거야
    #trj_samples=[]
    trj_samples=np.dot(W_sample[np.newaxis],np.transpose(Block_psi_cal)).reshape((N_TRJ_SAMPLE,n_dof*max_length)) # 이게 정답 이제 x~200,y~200,z~200을 ,x,y,z별로 추려내


    #    trj_samples_xyz[:, 3 * i:3 * i + 3] = np.array([np.transpose(trj_samples[:,3*i]) ,np.transpose(trj_samples[:,3*i+200]),np.transpose(trj_samples[:,3*i+400]) ])
    #     #trj_samples_xyz[:, 3 * i:3 * i + 2] = trj_samples[:, (3 * i,3 * i+200,3 * i+400)]
        #trj_samples_xyz[:,3*i :3*i+2]=(trj_samples[:, 3 * i], trj_samples[:, 3 * i + 1],trj_samples[:, 3 * i + 2])
        #8]:data = np.random.randn(2, 3) --[2x3] 행렬 ㅋ
    # np.array(A,B) A,B 둘다 [2X3]일때 [4X3]이됨 계속 행(아래로 추가 그러니 append(A,B,axis=1나 concatenate,axis=1 해야 열로추가가능



    return W_sample, trj_samples


if __name__ == '__main__' :
    tic= timeit.default_timer()

    ws_x_scale = 25;
    ws_y_scale = 30;
    ws_z_scale = 25;
    via1 = np.array([-2 / ws_x_scale, -3.81 / ws_y_scale, -9.18 / ws_z_scale])
    via1_quat= get_Normalized_Quaternion_from_euler_deg(62.34,-1.782,-35.842)
    via1_check_deg=Quat_2_deg(via1_quat[3],via1_quat[0],via1_quat[1],via1_quat[2] )
    via1_pose=np.concatenate((via1,via1_quat),None)

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

    # 교시데이터 추출
    # format_promp from initial trj

    demo_np_datas = []  # list
    initial_demo_path = '/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Demo_data/Expert_data_np_7dof_quat'

    #export_quat_path = '/home/kch/kch_work/Phcharm_project/LA_KCH_ProMPs_Linux/Four_VIA_ProMPs_passing_Wall/6DOF_ENV/Demo_data/Expert_data_np_7dof_quat'

    all_demo_files = glob.glob(initial_demo_path + "/*.npy")
    for i in all_demo_files:  # 난 저 교시 폴더안에 있는 데이터 다 끄집어내서 하나의 행렬에 다 저장 ㅋ
        ith_data = np.load(i)
        demo_np_datas.append(ith_data)


    Dimensions=3 # 3차원
    Num_via_points=4 #viapoint4개 출발점, 구멍앞뒤, 도착점
    N_trj=200
    Initial_ProMPs = ProMPs(Demo_datas=demo_np_datas, n_dim=Dimensions, num_via_point=Num_via_points)  # promp 인스턴스 생성

    Weight_initial_samples, \
    traj_initial_samples=Initial_ProMPs.Sample_weights_and_trj( N_trj=N_trj,Sigma_weight=Initial_ProMPs.Sigma_w,rel=False)



    # 3차원 입체 그래프 그리기 위해 필요 아래
    mpl.rcParams['legend.fontsize'] = 10
    plt.figure(1)
    ax = plt.axes(projection='3d')

    # ax.plot(data_phase_array_x,data_phase_array_y,data_phase_ㅎarray_z,label=str(file_count))
    ax.plot(Via_points[0,0],Via_points[1,0],Via_points[2,0], label="start", color='yellow', marker='o', markersize=9)
    ax.plot(Via_points[0,1],Via_points[1,1],Via_points[2,1], label="via1", color='red', marker='s', markersize=5)
    ax.plot(Via_points[0,2],Via_points[1,2],Via_points[2,2], label="via2", color='red', marker='s', markersize=5)
    ax.plot(Via_points[0,3],Via_points[1,3],Via_points[2,3], label="end", color='blue', marker='o', markersize=9)

    #ax.plot(Via_points[3,0],Via_points[3,1],Via_points[3,2], label="end", color='blue', marker='o', markersize=12)
    # plot_distribution
    for i in range(0, N_trj):
        # Initial_ProMPs.max_length
        ax.plot(traj_initial_samples[i, 0:Initial_ProMPs.max_length].flatten(),
                traj_initial_samples[i, Initial_ProMPs.max_length:Initial_ProMPs.max_length*2].flatten(),
                traj_initial_samples[i, Initial_ProMPs.max_length*2:Initial_ProMPs.max_length*3].flatten(),
                label=str(i))

    plt.figure(2)
    # plot_actual_guidance_trj_cov
    ax1 = plt.axes(projection='3d')
    # ax.plot(data_phase_array_x,data_phase_array_y,data_phase_ㅎarray_z,label=str(file_count))
    ax1.plot(Via_points[0, 0], Via_points[1, 0], Via_points[2, 0], label="start", color='yellow', marker='o',
            markersize=9)
    ax1.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="via1", color='red', marker='s', markersize=5)
    ax1.plot(Via_points[0, 2], Via_points[1, 2], Via_points[2, 2], label="via2", color='red', marker='s', markersize=5)
    ax1.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='blue', marker='o', markersize=9)

    # for i in range(0,Initial_ProMPs.max_length):
    #     #Initial_ProMPs.max_length
    #     ax1.plot( traj_init_samples_mean_xyz[0,3*i],traj_init_samples_mean_xyz[0,3*i+1],traj_init_samples_mean_xyz[0,3*i+2],label=str(i),color='black',marker='*',markersize=2 )
    mean_trj = np.mean(traj_initial_samples, axis=0)[np.newaxis]
    ax1.plot(traj_initial_samples[i, 0:Initial_ProMPs.max_length].flatten(),
            traj_initial_samples[i, Initial_ProMPs.max_length:Initial_ProMPs.max_length * 2].flatten(),
            traj_initial_samples[i, Initial_ProMPs.max_length * 2:Initial_ProMPs.max_length * 3].flatten())
    var_trj_200_xyz = np.sqrt(np.diag(Get_cov_from_signle_Mat(traj_initial_samples))) # 대각가져오기!


    plt.show(block=True)


# parameter_saving
    # Learned_model_folder = r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\3D_Moving_cube\Three_Via_Points\Learned_model_prameter_3dof_three_via_points"  # /\no상관
    # np.save(Learned_model_folder + "/inital_promp_blk_psi_N_10.npy", Initial_ProMPs.block_PSI)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    # np.save(Learned_model_folder + "/inital_weight_mat" + str(Known_characteristic_vec) + ".npy",
    #         Initial_ProMPs.weight_matrix)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    # np.save(Learned_model_folder + "/inital_env_chavec" + str(Known_characteristic_vec) + ".npy",
    #         Known_characteristic_vec)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    # # np.save( '폴더위치+저장할 이름 형태 ', 할당할 값 )
    # 걸린시간 및 코드종료확인
    toc = timeit.default_timer()
    print('Finish! _ step2')
    print('it takse' + str(toc - tic) + 'seconds')