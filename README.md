# STATE_BASED_RDKF_DATA_FDDBASED

## lambda_val은 증가할수록 필터가 더 보수적으로 상태를 추정하도록 유도하고, 과도한 오차 반영을 줄인다.
## eps는 감소할수록 로버스트성 영향 축소, 대신 노이즈 모델 신뢰도 증가
## psi_A,psi_B 모델의 불확실성이 높아질수록 크게 잡아야함.
# 모델을 정확히 알고 있으면 0.1~0.5 /// 모델이 약간 부정확하면 1.0 /// 모델이 매우 불확실하면 2.0~ 5.0

def state_based_rdkf(X, U, Y, R, Q, P0, lambda_val, psi_A, psi_B, eps):
    n, N = X.shape             # n: X의 상태의 개수/ N: data의 개수
    m = U.shape[0]             # m: 입력할 변수에 대한 개수
    p = Y.shape[0]             # Y: 출력값에 대한 개수

    # 1. Regression matrices
    X_p = X[:, :-1]             # (n, N-1)
    U_p = U[:, :-1]             # (m, N-1)
    Y_f = Y[:, 1:]              # (p, N-1)

    XU_p = np.vstack([X_p, U_p])   # (n+m, N-1)

    # 2. Z = C [A B] 추정
    Z = Y_f @ np.linalg.pinv(XU_p)   # (p, n+m)

    # 3. C 추정
    C_hat = Z[:, :n]                 # (p, n)

    # 4. [A B] 추정
    AB = np.linalg.pinv(C_hat) @ Z  # (n, n+m)
    A_t = AB[:, :n]                 # (n, n)
    B_t = AB[:, n:]                 # (n, m)

    # 5. 초기화
    R_hat = R - (eps / lambda_val) * (C_hat @ C_hat.T)
    P_hat = np.linalg.inv(np.linalg.inv(P0) + lambda_val * psi_A**2 * eps * np.eye(n))
    A_hat = A_t @ (np.eye(n) - lambda_val * psi_A**2 * eps * P_hat)
    B_hat = B_t - lambda_val * psi_B**2 * eps * (P_hat @ B_t)        #차원이 일치하지 않는 문제 발생해서 이렇게 정사영을 하여 차원을 보정한다.

    x_est = X[:, 0]
    X_hat_hist = np.zeros((n, N))
    X_hat_hist[:, 0] = x_est

    for k in range(N - 1):
        u_k = U[:, k].reshape(m, 1)
        y_kp1 = Y[:, k + 1].reshape(p, 1)

        # Predict
        x_pred = A_hat @ x_est.reshape(n, 1) + B_hat @ u_k
        y_pred = C_hat @ x_pred

        # Kalman Gain
        S = R_hat + C_hat @ P_hat @ C_hat.T
        L_k = P_hat @ C_hat.T @ np.linalg.inv(S)

        # Update
        x_est = (x_pred + L_k @ (y_kp1 - y_pred)).flatten()
        X_hat_hist[:, k + 1] = x_est

        # Covariance Update
        P_bar = A_hat @ P_hat @ A_hat.T + Q
        S_bar = R_hat + C_hat @ P_bar @ C_hat.T
        P_hat = P_bar - P_bar @ C_hat.T @ np.linalg.inv(S_bar) @ C_hat @ P_bar

    return X_hat_hist
