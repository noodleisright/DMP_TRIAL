import numpy as np # 행렬다루기 as 머이런건 애칭 allas name ,라이브러리나 다른 소스코드 쉽게불러오기ㅋ
import scipy as sp # 고급 통계 계산
from scipy import linalg # 내부 패키지 꼬임방지 -->함수를 못불러오는 이유가그래
import math # 제곱, 제곱근 등의 기초 사칙연산 및 쉬운함수들 sin cos 등
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import timeit


class ProMPs :

    def __init__(self,Demo_datas,n_dim,num_via_point):
        self.max_length =200 # 생성할 궤적의 point 수
        #self.number_features=num_gaussian_basis # trajectory feature 수 =gaussian basis
        self.noise=0.001 # tau=psi*w+noise
        self.regularizer=1e-12 #hyperparameter lamda in linear ridge regression-> determine weight components
        self.demo_trial_num=10 # 교시데이터 수
        self.dof=3 # 3축 포지션 어차피 궤적은 x,y,z 니

        #self.cal_weight_index_num= self.dof^n_dim+self.dof*num_via_point #trj_pose_dof^dimension +traj_pose_dof*via_point_num
        # 헐 print(4^3) 이면 7이 나옴 c랑 다르게 math.pow를 써야하네 , 혹은 **인데 직관적이지 못하네 ㅎ
        self.cal_weight_index_num =int(num_via_point*3*math.pow(2,n_dim-2))
        #self.cal_weight_index_num = int(math.pow( self.dof, n_dim)) + self.dof * num_via_point
        self.number_features = self.cal_weight_index_num# trajectory feature 수 =gaussian basis
        # PSI MATRIX 생성 - Normalized Gaussian basis function임
        PSI_matrix = self.get_PSI_mat(self.number_features, self.max_length) #10행, 200열--> #21행, 200열

        # Weight*PSI= Trajectory distriubtion임 block_psi로  (psi o 0; 0 psi 0 ; 0 0 psi )꼴로 해야 weight 차원과 연산되어 trjectory distribution나옴!!
        # Trajectory sample from promp 되려면
        # N_trj 여기선 100개 즉 궤적 분포로 부터 100개의 궤적이 출력이 된다.
        # 그 궤적분포의 차원은 100(200_x |200_y|200_z)로 100행 600열이 된다.

        #self.block_PSI=sp.linalg.block_diag(*([PSI_matrix]*self.dof) ) # 200*3행,10*3열 원본코드에서 넘 희안하게 쓴 코드들 직관적으로 바꿈
        self.block_PSI = sp.linalg.block_diag(*[PSI_matrix,PSI_matrix,PSI_matrix])  # 200*3행,21*3열 ex) block_diag (*[A,A]) = A 0 ; 0 A
        # Weight Matrix 생성 <-- 1trj 대해서 [1 x N*n_dof] =1자유도당 10개의 basis function index 할당함 그게 3개고 그러면 n개의 trj를 출력할거면
        # ==> weight matrix를 [ n x N*n_dof]가 나와야함
        self.weight_matrix = np.zeros((self.demo_trial_num,self.number_features * self.dof), dtype="float") # 10 행 30열 (교시횟수 ,basis_feature 수 *자유도)

        for i in range(self.demo_trial_num): #weight 행렬 차원 생성
            self.weight_matrix[i,:] = self.lin_ridge_reg(PSI_matrix, Demo_datas[i], self.regularizer)# i번째 열 다할당/가져오기
            print("processing_"+str(i)+"th_trj_weight_row")

        #mu_w=sp.mean(weight_matrix,axis=1) # axis=0은 행에 대한 , axis=1은 열에 대한 --> 근데 이제 잘안쓰지 np.mean으로 ㅎ
        self.mu_w=np.transpose(np.mean(self.weight_matrix,axis=0).reshape(1,self.number_features*self.dof)) # (30,) 튜플 아니라 직관적인 배열형태로 [21x1] --열에 대한 평균 --> [1x21]
        self.Sigma_INN_Hyper_lamda=1e-7
        # self.Sigma_w=np.cov(self.weight_matrix) # 이거 오류
        self.Sigma_w = Get_cov_from_signle_Mat(self.weight_matrix)
        self.Sigma = np.identity(self.Sigma_w.shape[0]) * self.Sigma_INN_Hyper_lamda







    def get_PSI_mat(self, n_basis, max_Time_step): # PSI 행렬 만들기
        # gaussian basis function = trjectories Feature=basis function


        def gabafun(basis_index,phase_index):
            return np.exp(-1/2 *math.pow( (phase_index - (basis_index - 1) * max_Time_step / (n_basis - 1)) ,2 )/ max_Time_step)

        Feature_matrix=np.zeros((max_Time_step,self.number_features),dtype="float") # 200행 10열의 차원
        # gabafun 이 phase_index-(basis_index-1) 형태라 basis_index는 1~10 , phase index는 1~200 형식으로 들어가야함 물론 코딩에서는 끝숫자 -1
        for time_step in range(1,max_Time_step+1): # c++에서 i=0 ; for i< array.length i++ 과 같음 --> range(5) 면 0~4임 ㅋ =range(0,5)
            for n_basis_index in range(1,self.number_features+1):
                Feature_matrix[time_step-1,n_basis_index-1]=gabafun(n_basis_index,time_step) #역행렬,전치행렬 아니야 gaba_fun 형태때문에 저래요

        normalizer = np.sum(Feature_matrix, axis=1) # 열별로 전부 더하기 -- 전체  phase 200개 데이터의 총합 =크기알려고
        Feature_matrix = np.divide(Feature_matrix, normalizer.reshape(np.size(normalizer, 0), 1)) # 이로써 정규화 --어차피 feature map 형태만 필요

        return Feature_matrix # 200행 10열이 만들어짐


    # Linear ridge regression for computing ProMP weights
    def lin_ridge_reg(self, feature_input, target_trj,hyper_lamda):
        # feature_input=PSI_MATRIX is a (T, N) matrix(정확히는 PSI'). target_data is a (max_trj, trial_num=N matrix =weight 의 차원임
        PSI_SAM=feature_input # 200행 10열
        PSI_T_X_PSI=np.dot(np.transpose(PSI_SAM),PSI_SAM ) #10행 10열 #m행 k열 * k행 n열 = m행 n열 - 즉 k열 k행으로 같아야 연산가능 # dot안해고 *,product하면 요소별로 곱해짐 근데 행렬 크기 다르면 망함.
        PSI_T_X_PSI_ADD_LAMDA_I_NN=(PSI_T_X_PSI+hyper_lamda*np.identity(np.size(PSI_T_X_PSI,0))) #10행 10열
        weights_row=( np.dot(np.dot( np.linalg.inv(PSI_T_X_PSI_ADD_LAMDA_I_NN) ,np.transpose(PSI_SAM)),target_trj )).reshape((1,self.number_features*self.dof),order='F') # A*B*C=A*(B*C)=(A*B)*C  ㅋ [10*3]=10행 3열 # 1자유도에 우리가 선정한 Featuremap_basis=10을 부여
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


def Gen_Moc_expert_data (dofs,Max_length,  start_via_point, end_via_point):


    direct_way = np.zeros((Max_length, dofs))
    for i in range(dofs):
        direct_way[:,i]=np.linspace(start_via_point[i], end_via_point[i], Max_length)
    MoC_expert_Data=[]

    for j in range(10):
        MoC_expert_Data.append(direct_way)


    return MoC_expert_Data

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



if __name__ == '__main__':
    tic = timeit.default_timer()

    ws_x_scale = 25;
    ws_y_scale = 30;
    ws_z_scale = 25;

    Demo_env_info = []
    # 주어진 환경알기
    # Given_properties

    via1 = np.array([-2/ws_x_scale, -3.81/ws_y_scale, -9.18/ws_z_scale])
    via2 = np.array([-1.483/ws_x_scale, 0/ws_y_scale, -1.16/ws_z_scale])
    via3 = np.array([1.063/ws_x_scale, 0/ws_y_scale, 1.144/ws_z_scale])
    via4 = np.array([4.26/ws_x_scale, 4.76/ws_y_scale, 3.56/ws_z_scale])

    # via1 = np.array([-2 / ws_x_scale, -7.06 / ws_y_scale, -9.18 / ws_z_scale])
    # via2 = np.array([-1.5818 / ws_x_scale, 0 / ws_y_scale, -1.16 / ws_z_scale])
    # via3 = np.array([1.2154 / ws_x_scale, 0 / ws_y_scale, 1.144 / ws_z_scale])
    # via4 = np.array([4.26 / ws_x_scale, -1.97 / ws_y_scale, 3.56 / ws_z_scale])

    Via_points =np.transpose( np.vstack((via1[:3], via2[:3], via3[:3], via4[:3]))) #:3 벡터 3개 다 인코딩한다는거임# [3X4]행렬



    Known_characteristic_vec = Known_env_cha_Vec(via1, via2, via3, via4)


    # 교시데이터 추출
    # format_promp from initial trj
    demo_np_datas = []  # list
    initial_demo_path = r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\Expert_data_np_3dof"  # /\no상관
    all_demo_files = glob.glob(initial_demo_path + "/*.npy")
    for i in all_demo_files:  # 난 저 교시 폴더안에 있는 데이터 다 끄집어내서 하나의 행렬에 다 저장 ㅋ
        ith_data = np.load(i)
        demo_np_datas.append(ith_data)



    Dimensions=3 # 3차원
    Num_via_points=4 #viapoint4개

    MoC_expertData=Gen_Moc_expert_data(dofs=3, Max_length=200, start_via_point=Via_points[:,0]  , end_via_point=Via_points[:,3])

    Initial_ProMPs = ProMPs(Demo_datas=MoC_expertData, n_dim=Dimensions, num_via_point=Num_via_points)  # promp 인스턴스 생성

    #    Initial_ProMPs = ProMPs(Demo_datas= demo_np_datas,n_dim=Dimensions,num_via_point=Num_via_points)  # promp 인스턴스 생성

    N_trj = 10
    # for trj_index in range(N_trj):

    Weight_init_samples, \
        traj_init_samples= Initial_ProMPs.Sample_weights_and_trj( N_trj=N_trj,Sigma_weight=Initial_ProMPs.Sigma_w,rel=False)
    # Weight_init_samples, \
    # traj_init_samples = Sample_trj_from_determined_ProMPs(N_trj, Initial_ProMPs.mu_w, Initial_ProMPs.Sigma_w,
    #                                                       Initial_ProMPs.block_PSI,n_dof=Initial_ProMPs.dof,max_length=Initial_ProMPs.max_length)

    # Weight_init_samples, \
    # traj_init_samples =Initial_ProMPs.Sample_weights_and_trj(N_trj=N_trj,Sigma_weight=None,rel=False )

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
        ax.plot(traj_init_samples[i, 0:200].flatten(),
                traj_init_samples[i, 200:400].flatten(),
                traj_init_samples[i, 400:600].flatten(),
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
    mean_trj = np.mean(traj_init_samples, axis=0)[np.newaxis]
    ax1.plot(mean_trj[0, 0:200].flatten(),
             mean_trj[0, 200:400].flatten(),
             mean_trj[0, 400:600].flatten(),
             label=str(i))
    plt.figure(3)

    ax2 = plt.axes(projection='3d')
    var_trj_200_xyz = np.sqrt(np.diag(Get_cov_from_signle_Mat(traj_init_samples))) # 대각가져오기!
    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz[i] * 300)

    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz[i + 200] * 300)

    for i in range(0, 200):
        ax2.plot(mean_trj[0, 0:200].flatten()[i],
                 mean_trj[0, 200:400].flatten()[i],
                 mean_trj[0, 400:600].flatten()[i],
                 color='green', marker='o', markersize=var_trj_200_xyz[i + 400] * 300)

    ax2.plot(Via_points[0, 0], Via_points[1, 0], Via_points[2, 0], label="start", color='yellow', marker='o',markersize=9)
    ax2.plot(Via_points[0, 1], Via_points[1, 1], Via_points[2, 1], label="via1", color='red', marker='s', markersize=5)
    ax2.plot(Via_points[0, 2], Via_points[1, 2], Via_points[2, 2], label="via2", color='red', marker='s', markersize=5)
    ax2.plot(Via_points[0, 3], Via_points[1, 3], Via_points[2, 3], label="end", color='blue', marker='o', markersize=9)
    plt.show()

    # parameter_saving
    Learned_model_folder = r"C:\김창현작업\Kch_project\Pycharmproject\MAT_REP_RWR_moving_cube\3D_Moving_cube\Four_Via_Points\Learned_model_parameter_3dof_four_via_points"  # /\no상관
    np.save(Learned_model_folder + "/inital_promp_blk_psi_N_10.npy", Initial_ProMPs.block_PSI)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    np.save(Learned_model_folder + "/inital_weight_mat" + str(Known_characteristic_vec) + ".npy",
            Initial_ProMPs.weight_matrix)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    np.save(Learned_model_folder + "/inital_env_chavec" + str(Known_characteristic_vec) + ".npy",
            Known_characteristic_vec)  # 사실상 psi,w,특성벡터만있으면되 ㅋ
    # np.save( '폴더위치+저장할 이름 형태 ', 할당할 값 )
    # 걸린시간 및 코드종료확인
    toc = timeit.default_timer()
    print('Finish! _ step2')
    print('it takse' + str(toc - tic) + 'seconds')
