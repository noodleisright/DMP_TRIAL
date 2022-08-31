import numpy as np # 행렬다루기 as 머이런건 애칭 allas name ,라이브러리나 다른 소스코드 쉽게불러오기ㅋ
import scipy as sp # 고급 통계 계산
import math # 제곱, 제곱근 등의 기초 사칙연산 및 쉬운함수들 sin cos 등
import matplotlib as mpl
import matplotlib.pyplot as plt
import glob
import sys
import timeit
import math
from Step_1_6_Dof_DTW_NP import get_Normalized_Quaternion_from_euler_deg



class ProMPs_dof_select : # (weight의 평균,분산 | PSI_Matrix ) 라는 pickle파일로 저장!!

    def __init__(self,Demo_datas,_dof):
        self.max_length =1000 # 생성할 궤적의 point 수
        self.noise=1e-5 # tau=psi*w+noise
        self.regularizer=1e-12 #hyperparameter lamda in linear ridge regression-> determine weight components
        self.demo_trial_num=10 # 교시데이터 수
        self.dof=_dof # 선택 Position인지 orientation인지.
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

class Learnt_policy: # optimal policy 생성
    #결국 최종 적으로 이 3가지를 이용해 weight matrix만들고 psi랑 합침
    def __init__(self,all_objectives_vals,mu_w,Sigma_w,Blck_PSI,relevances):
        self.all_objectives_vals=all_objectives_vals
        self.mu_w=mu_w
        self.Sigma_w=Sigma_w
        self.Blck_PSI = Blck_PSI
        self.relevance_mat=relevances


def Optimize_Trj_distribution_PRO(promp,via_points_Mat,n_iter,N_trj,weight_index_num): # promp 와 mu,sigma 그리고200 iteration당 200개 궤적 ㅋ

    # hyper_parameter
    beta =1 # 실험 값임. 1~300사이라고했음 하지만 너무 R이작아서 바꾼다 . 근데 어떻게든 0으로 되니 원
    len_beta = 0.1  # len가 젤큰값임 그러니 젤 작게
    jerk_beta = 1e5  # beta나 파라미터 가중치나 같은말 논문에선 10^5 근데 좀 바꾸자

    n_dof = promp.dof  # 3임
    # 거리구한걸 scale 다운하면 아예 궤적이 쪼그러듬
    # the values for beta_l and beta_j were determined by tryingA different powers of 10
    # 논문에선 6자유도 문제에서 beta= 200, len_beta=10^-1 , jerk_beta=10^5 ---> 근데 e^-x 인꼴은 어떻게든 저값들을 작게하면서 0~1사이로 하는거니
    reward_weight = np.zeros((N_trj, 1))
    # n_relfun = via_points_Mat.shape[1]  # 4 # relevance learning의 Ojectives는 position 차이임 그때 via point수 4개  # via_mat은지금 [3x4] shape1해도되는데 걍 값을 아니깐 ㅎ
    n_relfun = 6
    all_objective_vals = [np.zeros((n_iter, N_trj)) for i in range(n_relfun)]  # 5쌍의 [150X200]리스트임 ㅋ

    # Env_dim = via_points_Mat.shape[0]  # 3임 3자유도 ㅎ
    returns = np.zeros((N_trj, n_iter))  # [200X150]
    # 피어슨 상관계수 구하고 뭐등등 relevance learning

    updated_promp_Sigma = np.zeros((weight_index_num * n_dof, weight_index_num * n_dof))
    new_promp_Sigma = np.zeros((weight_index_num * n_dof, weight_index_num * n_dof))
    sigma_epsilon = sys.float_info.epsilon  # sigma가 무진장 작게됨 2.220446049250313e-16
    weights_rel = np.zeros((weight_index_num * n_dof, weight_index_num * n_dof))
    C_k_i = np.zeros((weight_index_num * n_dof, weight_index_num * n_dof))
    # Sigma_w = np.identity(n=30,  dtype=np.int8) * sigma_epsilon  # 30x30 에서 대각 추출은 추후 relevnace learning 에서... 일단은 초기 promp배운 sigma w
    mu_w = promp.mu_w  # 초기 promp에서 배운 mu_w

    converged = False
    Converging_checking_term = 5  # 5번씩 학습하면서 확인 ㅋ
    Epsilon = 1e-3  # converge 즉  수렴, 학습끝인지알기

    n = weights_rel.shape[0]  # 30->63

    ######## FOR DEBUG####################
    mu_check = np.zeros((n_iter, weight_index_num * n_dof))
    Sigma_w_check = [np.zeros((weight_index_num * n_dof, weight_index_num * n_dof)) for i in range(n_iter)]
    Weights_rel_check = [np.zeros((weight_index_num * n_dof, weight_index_num * n_dof)) for i in range(n_iter)]
    Relvance_check = [np.zeros((n_relfun, weight_index_num * n_dof)) for i in range(n_iter)]


    ### Main_training#################

    for iteration in range(n_iter):  # 150 iteration임

        print("iteration_" + str(iteration + 1) + "_begin")
        print("***")
        print(str(iteration + 1) + "번째 episode 학습중")

        if iteration >= Converging_checking_term:
            converged = True
            for rel_fun_index in range(n_relfun):  # 4개의 via point 에대한 rel func 연산
                tmp_object_vals = all_objective_vals[rel_fun_index]
                # 각 via point당 objective val 구하기 = 거리차이 값
                tmp_objectives_slect_sec = tmp_object_vals[:,
                                           iteration - Converging_checking_term: iteration]  # -5 :200 즉 1,2,3,4,이렇게 텀 가져가
                max_val = np.max(tmp_objectives_slect_sec)
                min_val = np.min(tmp_objectives_slect_sec)
                dist_err = max_val - min_val
                if (dist_err) > Epsilon:  # 최대=최소 거리차가 아직도 멀었다.
                    converged = False  # 될때까지 학습하기

        if converged == True:
            print("Converged!!_in_this_env")

            for rel_fun_index in range(n_relfun):
                all_objective_vals[rel_fun_index] = all_objective_vals[rel_fun_index][:, 0:iteration]  # 지금까지 추려낸거
            break  # 루프를 끊고나와 ## 이건 분모가 0되는거 막아야 넣을수있어.

        # 매 iteration 마다 RELEVANCE 업데이트

        relevance = learn_relevances (promp, via_points_Mat,N_trj)  # learn_relevances(n_relfun, dofs, env, sample_trajectories, n_samples, mu_w, var_w, num_gaba,                                     block_PSI, start, end, get_target)

        # 이제 현제 궤적분포를 via point 수만큼 도입
        for rel_fun_index in range(n_relfun):  # 우선 4개 via point 및 그외 objectives는 2개 니깐 6
            # 해당 via point 각가에 대한 relevance를 파악하여 weight의 variance 를 rescale한다.

            previous_Sigma_w = promp.Sigma_w
            previous_mu_W = promp.mu_w
            # print( previous_Sigma_w)

            weights_rel[range(n), range(n)] = np.diag(previous_Sigma_w) * np.transpose(relevance[rel_fun_index,:])  # 이래야 f*rel 형태 #대각 성분 가지고 행렬형태만들기1 # np.diag 요소가 행렬이면 대각성분추출 ,요소가 벡터값이면 그걸로 새로운 행렬만듬

            for i in range(0, n):
                if weights_rel[i, i] <= sigma_epsilon:
                    weights_rel[i, i] = sigma_epsilon

            # print(weights_rel)
            weight_samples, traj_samples = promp.Sample_weights_and_trj(N_trj=N_trj, Sigma_weight=weights_rel, rel=True)

            if rel_fun_index == 0:

                current_objective_val = compute_dist_to_initial_via_point(traj_samples,via_point_mat_index=via_points_Mat[:, [ rel_fun_index]]).flatten()
                print("get_Via_point1_dist_start")
                print("*********")

            elif rel_fun_index == 1:

                current_objective_val = compute_dist_to_between_via_point(traj_samples,via_point_mat_index=via_points_Mat[:, [rel_fun_index]]).flatten()
                print("get_Via_point_2_dist")
                print("************")
            elif rel_fun_index == 2:

                current_objective_val = compute_dist_to_between_via_point(traj_samples,via_point_mat_index=via_points_Mat[:, [rel_fun_index]]).flatten()
                print("get_Via_point_3_dist")
                print("************")
            elif rel_fun_index == 3:

                current_objective_val = compute_dist_to_end_via_point(traj_samples,via_point_mat_index=via_points_Mat[:,[rel_fun_index]]).flatten()
                print("get_Via_point_4_dist_end")
                print("******************")

            elif rel_fun_index == 4:

                current_objective_val = compute_lens(traj_samples, N_trj)
                print("get_trj_length")
                print("******************")

            elif rel_fun_index == 5:

                current_objective_val = compute_jerks(traj_samples, N_trj, promp.max_length)
                print("get_trj_length")
                print("******************")

            all_objective_vals[rel_fun_index][iteration, :] = current_objective_val.flatten()

            N_TRJ_SAMPLES = traj_samples
            Via_points_mat = via_points_Mat
            Max_length_of_trj = promp.max_length
            BETA_L = len_beta
            BETA_J = jerk_beta

            # o_1= - compute_dist_to_initial_via_point(traj_samples,via_point_mat_index=via_points_Mat[:,[0]]).flatten()
            # o_2 =- compute_dist_to_between_via_point(traj_samples, via_point_mat_index=via_points_Mat[:,[1]]).flatten()
            # o_3=  - compute_dist_to_end_via_point(traj_samples, via_point_mat_index=via_points_Mat[:,[2]]).flatten()
            # o_4 =- (np.dot(BETA_L, compute_lens(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0]).flatten()))
            # o_5 = - (np.dot(BETA_J,  compute_jerks(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0], Max_length_of_trj).flatten()))
            # R = o_1 + o_2 + o_3 + o_4 + o_5

            R = Get_Reward(N_TRJ_SAMPLES=traj_samples, Via_points_mat=via_points_Mat,
                           Max_length_of_trj=promp.max_length, BETA_L=len_beta, BETA_J=jerk_beta)
            reward_weight = np.exp(beta * R)  # [N_sample X 1] # relevance의 학습 200 x1] 이부분에서 nan 에러가 나왔음 e^(-27*200) 이면 0이나오네?

            # for i in range(0, n_samples):
            #     if reward_weight[i] <= sigma_epsilon:  # exp(-5)=6.737946*10^-3
            #         reward_weight[i] = sigma_epsilon

            print("get_Reward_  " + str(rel_fun_index + 1) + " 번째 via point")
            print("_____________________")

            # Update_Promps_###########################
            # Update mu_W
            W_minus_mu = np.subtract(weight_samples, np.transpose(previous_mu_W))  # np.subtract 는 row 행별로 빼는 함수 그러니 다시 돌려주면됨.

            new_mu_w = np.dot(np.transpose(weight_samples), reward_weight) / np.sum(
                reward_weight)  # SIGMA(R*W)/R # 이코드의 주된에러가 여기서 발생! ,원인은 hyper parameter 튜닝때문!
            ##########################################################################################

            # Update_Sigma_W
            new_promp_Sigma = np.zeros((weight_index_num * n_dof, weight_index_num * n_dof))  # 매 iteration 마다 reset

            for i in range(N_trj):  # C_ik---> 200 개  sample trj에 대해 구함
                new_promp_Sigma = new_promp_Sigma + reward_weight[i] * np.transpose(W_minus_mu[i, :]) * W_minus_mu[i,
                                                                                                        :]  ## 이전 iteration의 sigma_w가져옴 , 이래야 30 30 됨?

            new_promp_Sigma = new_promp_Sigma / np.sum(reward_weight)

            C_k_i = new_promp_Sigma
            C_k_i = (C_k_i + np.transpose(C_k_i)) / 2  # symmetric하게

            updated_promp_Sigma = np.zeros((C_k_i.shape[0], C_k_i.shape[0]))  # 마찬가지이유
            updated_promp_Sigma[range(n), range(n)] = (1 - relevance[rel_fun_index, :]) * np.diag(
                previous_Sigma_w) + relevance[rel_fun_index, :] * np.diag(C_k_i)  # 이래야 diag가 됨

            C_k_i = updated_promp_Sigma

            # update_ promp
            promp.mu_w = new_mu_w
            promp.Sigma_w = C_k_i
            print("Update_promp___" + str(iteration + 1) + "________" + str(rel_fun_index))

        returns[:, iteration] = reward_weight.flatten()  # [200x150] dataarray
        mu_check[iteration, :] = promp.mu_w.flatten()
        Blck_PSI = promp.block_PSI
        Sigma_w_check[iteration] = C_k_i
        Weights_rel_check[iteration] = weights_rel
        Relvance_check[iteration] = relevance
        print("******************" + str(iteration + 1) + "_iteration_PRO_done    converged? :" + str(converged))

    return Learnt_policy(all_objective_vals,promp.mu_w,promp.Sigma_w,Blck_PSI,relevance) ,returns, mu_check,Sigma_w_check,Weights_rel_check,Relvance_check


# Learns relevance functions
def learn_relevances(promp_draft, viapoints ,Num_Trj):
    n_relfun =6 # 4+1+1

    #Sigma_weight = np.identity(Sigma_weight.shape[0]) * 1e-7  # [30x30] 의 Sigma_Weight에서 가져와서 틀을 만드는거야 ---> lamda=10^-7 인건 임의로 생성한 weight _sigma이고 이걸 튜닝ㅎ나다.
    # 임의로 틀을 정해주는거임 ㅋ
    # Trajectory Sample from ProMP

    #weight_samples, traj_samples = promp.Sample_weights_and_trj(100, None,True)  ## 학습할때도 똑같이 100개 궤적만들어내네 -->relevance elarning 없을때 ,초기 weight만들기

    weight_samples, \
    traj_samples = promp_draft.Sample_weights_and_trj(N_trj=Num_Trj, Sigma_weight=None, rel=True)
    # weight_samples,\
    # traj_samples=MIP.Sample_trj_from_determined_ProMPs( N_TRJ_SAMPLE=200,mu_W=mu_W,Sigma_W=Sigma_weight,Block_psi_cal=bLOCK_PSI, n_dof=3,max_length=200)
    # # Compute distances
    dists_viapoints = np.zeros((Num_Trj, viapoints.shape[1]))  # relevance learning에서의 objective는 거리차이임. 즉 dof별로 인근 via point 간의 pose 맞추는거임. === [200X4]
    length_trjs = np.zeros((Num_Trj, 1))  # relevance learning에서의 objective는 거리차이임. 즉 dof별로 인근 via point 간의 pose 맞추는거임.
    jerk_trjs = np.zeros((Num_Trj, 1))  # relevance learning에서의 objective는 거리차이임. 즉 dof별로 인근 via point 간의 pose 맞추는거임.
    obs_val =np.zeros((Num_Trj,dists_viapoints.shape[1]+length_trjs.shape[1]+jerk_trjs.shape[1] ))

    #viapoint는 행은 dof, 열은 via point수를 나타냄
    for rel_fun in range(n_relfun):  # relevance_function을 4가지 VIA 포인트 각각연산하며 만들어내

        # rel_obj = 궤적의 길이

        if rel_fun==0:
            obs_val[:, rel_fun] = compute_lens(traj_samples,Num_Trj).flatten()
            print('objectives_1_trj_length_')

        # rel_obj = 궤적의 변화 == jerk

        elif rel_fun==1:
            obs_val[:, rel_fun] = compute_jerks(traj_samples,Num_Trj,promp_draft.max_length).flatten()
            print('objectives_2_trj_jerk_')
        # rel_obj = 그 via point와 궤적의 최단 거리
        elif rel_fun ==3:

            obs_val[:, rel_fun] = compute_dist_to_initial_via_point(traj_samples, via_point_mat_index=viapoints[:, [rel_fun-2]]).flatten()
            print( 'objectives_3_via1_dist_')


        elif rel_fun == 4 :

            obs_val[:, rel_fun] = compute_dist_to_between_via_point(traj_samples, via_point_mat_index=viapoints[:, [rel_fun-2]]).flatten()
            print( 'objectives_4_via2_dist_')


        elif rel_fun == 5 :

            obs_val[:, rel_fun] = compute_dist_to_between_via_point(traj_samples, via_point_mat_index=viapoints[:, [rel_fun-2]]).flatten()
            print( 'objectives_5_via3_dist_')


        elif rel_fun == 6:

            obs_val[:, rel_fun] = compute_dist_to_end_via_point(traj_samples, via_point_mat_index=viapoints[:, [rel_fun-2]]).flatten()
            print( 'objectives_6_via4_dist_')



    # Compute Pearson's correlation coefficient
    cols_w = weight_samples.shape[1]  # 200개 궤적이니 200x21*3 에서 21*3이 나옴
    cols_v = obs_val.shape[1]  # 6임
    relevances = np.zeros((cols_v, cols_w))  # [4X30] 형태 행은 rel_거리== 어떻게든 viapoint 근방 phase는 다 거치게 하는게 주 목표 , 그리고 열은 10개 index를 x,y,z로 거침
    for i in range(cols_w):
        for j in range(cols_v): # 4 via_po
            #relevances[j,i] = np.corrcoef(weight_samples[:, i],obs_val[:, j], rowvar=False)[0][1] # 이래서 한줄만에 구해진다는거임 ㅋ np.corrcoef(x, y)[0, 1] 이게 맞는데 --> 두 x,y 상관계수만 확인하기 ㅋ 이게 미친 오류 다시
            relevances[j, i] = cor_m(weight_samples[:, i],obs_val[:, j])

    relevances = np.abs(relevances)

    print("relevance_learning_done!_this_iteration")
    print("******")
    return relevances



def compute_dist_to_initial_via_point(traj_samples, via_point_mat_index):

    n_sample = traj_samples.shape[0]  # 여기선 200개

    via_point_mat_index = np.transpose(via_point_mat_index)

    max_length = 200  # 난 200개의 점 도출
    actual_dist = np.zeros((200, 1))

    actual_dist = traj_samples[:, [0, max_length, max_length * 2]] - via_point_mat_index
    actual_dist = np.sqrt(np.sum(np.power(actual_dist , 2), axis=1))
    return actual_dist

def compute_dist_to_between_via_point(traj_samples, via_point_mat_index):


    n_via_dof = 3  # 3 viapoint pose의 자유도임 3임
    via_point_mat_index=np.transpose(via_point_mat_index) # [3X4] via1,2,3,4를 [4X3]으로 바꾸어 np.subtract맞게
    n_sample = traj_samples.shape[0]  # 여기선 200개

    max_length = 200  # 난 200개의 점 도출

    Expected_via_points_dist = np.zeros((n_sample,n_via_dof))  # [200X3] 각 궤적 다발에서 맞추기로 예상되는 via point 거리 <- 확실히 어느 phase일지 모르지만 대강 처음 위치에서 어느정도  거리있을지 추려내느거임
    All_n_Trj_points_dist = np.zeros((n_sample, max_length))  # [200x200]
    actual_dist = np.zeros((200, 1))

    for phase in range(max_length):
        All_n_Trj_points_dist[:,phase]=np.sqrt(np.sum(np.power( traj_samples[:,[phase+d_index*max_length for d_index in range(n_via_dof)]]-via_point_mat_index ,2) ,axis=1))

    min_distance_phase_index = np.argmin(All_n_Trj_points_dist, axis=1) # 이 궤적이 200개의 점으로 이루저져있는데 경유점과의 거리가 가장 작은 지점이 대표 phase로 한다.

    for Trj_i_th in range(n_sample):  # 각각 sample 궤적에 대해서
        Expected_via_points_dist[Trj_i_th,:]=traj_samples[Trj_i_th,[min_distance_phase_index[Trj_i_th]+d_index*max_length for d_index in range(n_via_dof)]]

    actual_dist = Expected_via_points_dist - via_point_mat_index
    actual_dist = np.sqrt(np.sum(np.power(actual_dist, 2), axis=1))

    # for Trj_i_th in range(n_sample):  # 각각 sample 궤적에 대해서
    #     min_dist_in_via_phase = min_distance_phase_index[Trj_i_th]  # 100개의 궤적들중 해당 viapoint에 젤 가까운지점 찾기
    #     Expected_via_points_dist[Trj_i_th, :] = traj_samples[Trj_i_th, [min_dist_in_via_phase + d * max_length for d in range(3)]]  # 200x3
    #     actual_dist[Trj_i_th]=np.sqrt(np.sum(np.power( np.subtract( Expected_via_points_dist[Trj_i_th] , via_point_mat_index     )   ,2)   )) # 200x1

    return actual_dist




def compute_dist_to_end_via_point(traj_samples, via_point_mat_index):

    via_point_mat_index = np.transpose(via_point_mat_index) # [3X4] via1,2,3,4를 [4X3]으로 바꾸어 np.subtract맞게
    n_sample = traj_samples.shape[0]  # 여기선 100개
    max_length = 200  # 난 200개의 점 도출
    actual_dist = np.zeros((200, 1))

    actual_dist=traj_samples[:, [max_length-1, max_length*2-1, max_length * 3-1]]-via_point_mat_index
    actual_dist=np.sqrt( np.sum( np.power(actual_dist,2),axis=1))

    # for Trj_i_th in range(n_sample):
    #     #actual_dist[Trj_i_th] = np.linalg.norm(np.subtract( traj_samples[Trj_i_th, [max_length-1, max_length*2-1, max_length * 3-1]],via_point_mat_index))
    #     actual_dist[Trj_i_th] = np.sqrt( np.sum(np.power( np.subtract(traj_samples[Trj_i_th, [max_length-1, max_length*2-1, max_length * 3-1]], via_point_mat_index  )   ,2),axis=1    ))# 200x1 m to cm



    return actual_dist




def compute_lens(trajectory_samples_from_promp, n_samples):
    lengths = np.zeros((n_samples, 1))
    for traj_sample_index in range(n_samples):
        trajectory = trajectory_samples_from_promp[traj_sample_index, :].reshape((200, 3), order='F')  # 100X600에서 1X600추려내서 200x3 형태로 200페이즈의 x,y,z로 추려내 F는 포트란 형식이란 뜻으로 무조건 앞차원 위주로 배열
        velocities = np.diff(trajectory, axis=0)  ## 앞서 행이 ith번째 phase의 pose값이니 지금행과 아래행간의 차이를 velocity로 도출
        lengths[traj_sample_index, 0] = np.sum(np.sqrt(np.sum(np.power(velocities, 2), axis=1)), axis=0)
    return lengths

# Computes pairwise difference n times
def compute_diff(vals, n):
    i = 0
    arr1 = []
    arr2 = []
    ret = []
    while i < n: # 총 100개 데이터중 0부터 99까지 데이터와 1부터 100까지 데이터의 차이임 이걸 몇번구하느냐가 ㅎ
        arr1 = vals[0:len(vals) - 1] if i == 0 else ret[0:len(ret) - 1]
        arr2 = vals[1:len(vals)] if i == 0 else ret[1:len(ret)]
        ret = np.subtract(arr2, arr1)  # 행의 기준이니깐 다른
        i += 1
    return np.array(ret)

def compute_jerks(trajectory_samples_from_promp, n_samples, max_length):
    jerks = np.zeros((n_samples, 1))
    for traj_sample in range(n_samples):
        curr_jerks = 0
        rng = range(3) # x,y,z 니깐
        for i in rng:  # rng stands for range.
            curr_vals = trajectory_samples_from_promp[traj_sample][i * max_length:(i + 1) * max_length - 1]
            curr_jerks += np.sum(np.abs(compute_diff(curr_vals, 3)))
        jerks[traj_sample, 0] = (curr_jerks / (3 * max_length))
    return jerks  # (n_samples, 1)



def Get_Reward( N_TRJ_SAMPLES,Via_points_mat,Max_length_of_trj,BETA_L,BETA_J ):
    ### R= exp(-b(o_1+o_2+o_3+b_l*l+b_j*j) 즉 저것과 별개로 계산해서 합치는거야 !!!
    ### o represents the distances to each of the four via points ===> 이 코드에선 3개인데 암튼 합일것임
    ### l represent the length of trj
    ### j represent the average jerk magnitude of the trj
    #######  즉 relevance의 요소는 5개로 이전 iteration으로 만들어낸 relevance를 적용해서 만든 weight로 새로 궤적분포(200개의 궤적다발)을 생성하고
    ####### 각 궤적별로 via point들과 궤적간의 거리 그리고 궤적들의 length, jerk를 확인한다.
    ####### 그리고 policy의 reward를 보면 각 objectives들은 다 최소화 해야 하는 항으로 구성되면서 학습
    ####### 이걸 objectives 5개니

    # jerks = compute_jerks(traj_samples, N_trj, promp.max_length)
    # lens = compute_lens(traj_samples, N_trj)
    # R = -current_objective_val - (np.dot(len_beta, lens)).reshape(N_trj, 1) - (
    #     np.dot(jerk_beta, jerks)).reshape(N_trj, 1)

    # current_objective_val = compute_dist_to_initial_via_point(traj_samples,
    #                                                           via_point_mat_index=Via_points_mat[:,
    #                                                                               [0]])
    #
    # current_objective_val = compute_dist_to_between_via_point(traj_samples,
    #                                                           via_point_mat_index=Via_points_mat[:,
    #                                                                               [1]])
    #
    # current_objective_val = compute_dist_to_end_via_point(traj_samples,
    #                                                           via_point_mat_index=Via_points_mat[:,
    #                                                                               [2]])

    # o_1= - compute_dist_to_initial_via_point(traj_samples,via_point_mat_index=via_points_Mat[:,[0]]).flatten()
    # o_2 =- compute_dist_to_between_via_point(traj_samples, via_point_mat_index=via_points_Mat[:,[1]]).flatten()
    # o_3=  - compute_dist_to_end_via_point(traj_samples, via_point_mat_index=via_points_Mat[:,[2]]).flatten()
    # o_4 =- (np.dot(BETA_L, compute_lens(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0]).flatten()))
    # o_5 = - (np.dot(BETA_J,  compute_jerks(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0], Max_length_of_trj).flatten()))

    Reward_fun=- compute_dist_to_initial_via_point(N_TRJ_SAMPLES,via_point_mat_index=Via_points_mat[:,[0]]).flatten()\
               - compute_dist_to_between_via_point(N_TRJ_SAMPLES,via_point_mat_index=Via_points_mat[:,[1]]).flatten()\
               - compute_dist_to_end_via_point(N_TRJ_SAMPLES, via_point_mat_index=Via_points_mat[:, [2]]).flatten()\
               - (np.dot(BETA_L, compute_lens(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0]).flatten()))\
               - ( np.dot(BETA_J, compute_jerks(N_TRJ_SAMPLES, N_TRJ_SAMPLES.shape[0],Max_length_of_trj).flatten()))

    return Reward_fun



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

#