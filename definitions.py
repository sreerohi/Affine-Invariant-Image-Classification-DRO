def project_tau(tau0,tau1, M,N, printer=0):
  # c0 = np.floor((M-1)/2)
  # c1 = np.floor((N-1)/2)
  c0 = (M-1)/2
  c1 = (N-1)/2
  if(printer==1):
    print(c0,c1)


  mVector = (np.arange(M) -c0).reshape(M,1)
  m1Vector = (np.ones(M).reshape(M,1))
  nVector = (np.arange(N) -c1).reshape(N,1)
  n1Vector = np.ones(N).reshape(N,1)

  

  b1 = np.matmul(mVector, n1Vector.T)
  b2 = np.matmul(m1Vector, nVector.T)
  b3 = np.matmul(m1Vector, n1Vector.T)
  if(printer==1):
    print("******")
    print(b1)
    print(b2)
    print(b3)
    print("******")
  p1 = np.matmul( mVector.T, tau0)
  p1 = np.matmul(p1, n1Vector)
  p1 = p1/ np.linalg.norm(mVector)**2
  p1 = p1/N

  p2 = np.matmul( m1Vector.T, tau0)
  p2 = np.matmul(p2, nVector  )
  p2 = p2/np.linalg.norm(nVector)**2
  p2 = p2/M

  p3 = np.matmul ( m1Vector.T, tau0 )
  p3 = np.matmul(p3, n1Vector)
  p3 = p3/(M*N )

  p4 = np.matmul( mVector.T, tau1)
  p4 = np.matmul(p4, n1Vector)
  p4 = p4/ (np.linalg.norm(mVector))**2
  p4 = p4/N

  p5 = np.matmul( m1Vector.T, tau1)
  p5 = np.matmul(p5, nVector  )
  p5 = p5/np.linalg.norm(nVector)**2
  p5 = p5/M

  p6 = np.matmul ( m1Vector.T, tau1 )
  p6 = np.matmul(p6, n1Vector)
  p6 = p6/(M*N ) 

  tau0 = p1*b1 + p2*b2 + (p3 *b3 )
  tau1 = p4*b1 + p5*b2 + (p6 *b3 )
  if(printer==1):
    print(p1,p2,p3,p4,p5,p6)

  return (tau0 , tau1   ) 


def affine_to_vf2(A, b, M, N):
    #Original function from [1] modified.
    A0 = A[:,0]
    A1 = A[:,1]

    c0 = (M-1)/2
    c1 = (N-1)/2


    eu = np.dot( (np.arange(0,M) -c0)[:,np.newaxis], np.ones(N)[:,np.newaxis].T)
    ev = np.dot(np.ones(M)[:,np.newaxis], (np.arange(0,N )-c1)[:,np.newaxis].T)
    # print(eu)
    # print(eu.shape)
    # print(A0[np.newaxis, np.newaxis, :] )
    # print(A0[np.newaxis, np.newaxis, :].shape )
    # print(eu[..., np.newaxis])
    # print(eu[..., np.newaxis].shape)
    v = np.moveaxis(A0[np.newaxis, np.newaxis, :] * eu[..., np.newaxis], 0,0)
    # print(v)
    # print(v.shape)

    tau = A0[np.newaxis, np.newaxis, :] * eu[..., np.newaxis] + \
            A1[np.newaxis, np.newaxis, :] * ev[..., np.newaxis] + \
            b[np.newaxis, np.newaxis, :] * np.ones((M, N, 1))
    # print("$$$$")
    # print(tau[:,:,0])
    return (tau[:,:,0], tau[:,:,1])

def identity_vf(M, N, RM=None, RN=None):
    #Original function from [1] .
    if RM is None:
        RM = M
    if RN is None:
        RN = N

    m_vec = np.linspace(0, M-1, RM)
    n_vec = np.linspace(0, N-1, RN)

    eu = np.dot(m_vec[:,np.newaxis], np.ones(RN)[:,np.newaxis].T)
    ev = np.dot(np.ones(RM)[:,np.newaxis], n_vec[:,np.newaxis].T)

    return (eu, ev)

def image_interpolation_bicubic(x,tau1,tau2):
    #Original function from [1] modified.
    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau1 - t1_0
    a0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau1 - t1_1
    a1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau1 - t1_2
    a2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau1 - t1_3
    a3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau2 - t2_0
    b0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau2 - t2_1
    b1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau2 - t2_2
    b2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau2 - t2_3
    b3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    x_pr = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    return x_pr

def dimage_interpolation_bicubic_dtau1(x,tau1,tau2):
    #Original function from [1] modified.
    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #

    # phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
    #              { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (sgn(u) = 1)
    u = tau1 - t1_0
    a0 = -1.5 * u ** 2 + 5 * u - 4
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (sgn(u) = 1)
    u = tau1 - t1_1
    a1 = 4.5 * u ** 2 - 5 * u 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (sgn(u) = -1)
    u = tau1 - t1_2
    a2 = -4.5 * u ** 2 - 5 * u 
 
    # 3: here, we are in case (1)
    #          and u is negative (sgn(u) = -1)
    u = tau1 - t1_3
    a3 = 1.5 * u ** 2 + 5 * u + 4 
    
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau2 - t2_0
    b0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau2 - t2_1
    b1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau2 - t2_2
    b2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau2 - t2_3
    b3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2
    
    dx_pr_dtau1 = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    
    return dx_pr_dtau1
    
def dimage_interpolation_bicubic_dtau2(x,tau1,tau2):
    #Original function from [1] modified.
    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]

    N0 = x.shape[0]
    N1 = x.shape[1]
    N2 = x.shape[2]
    
    # embed with zeros at boundary
    xx = np.zeros((N0+2,N1+2,N2))
    xx[1:(N0+1),1:(N1+1),:] = x.copy()
    
    # shift tau1 and tau2 to account for this embedding
    tau1 = tau1 + 1 
    tau2 = tau2 + 1
    
    ## generate the 16 resampled slices that will be combined to make up our interpolated image 
    #
    # 
    ft1 = np.floor(tau1)
    ft2 = np.floor(tau2)
    
    t1_0 = ( np.minimum( np.maximum( ft1 - 1, 0 ), N0 + 1 ) ).astype(int)
    t1_1 = ( np.minimum( np.maximum( ft1, 0     ), N0 + 1 ) ).astype(int)
    t1_2 = ( np.minimum( np.maximum( ft1 + 1, 0 ), N0 + 1 ) ).astype(int)
    t1_3 = ( np.minimum( np.maximum( ft1 + 2, 0 ), N0 + 1 ) ).astype(int)

    t2_0 = ( np.minimum( np.maximum( ft2 - 1, 0 ), N1 + 1 ) ).astype(int)
    t2_1 = ( np.minimum( np.maximum( ft2, 0     ), N1 + 1 ) ).astype(int)
    t2_2 = ( np.minimum( np.maximum( ft2 + 1, 0 ), N1 + 1 ) ).astype(int)
    t2_3 = ( np.minimum( np.maximum( ft2 + 2, 0 ), N1 + 1 ) ).astype(int)
    
    x_00 = xx[ t1_0, t2_0 ]
    x_01 = xx[ t1_0, t2_1 ]
    x_02 = xx[ t1_0, t2_2 ]
    x_03 = xx[ t1_0, t2_3 ]
    x_10 = xx[ t1_1, t2_0 ]
    x_11 = xx[ t1_1, t2_1 ]
    x_12 = xx[ t1_1, t2_2 ]
    x_13 = xx[ t1_1, t2_3 ]
    x_20 = xx[ t1_2, t2_0 ]
    x_21 = xx[ t1_2, t2_1 ]
    x_22 = xx[ t1_2, t2_2 ]
    x_23 = xx[ t1_2, t2_3 ]
    x_30 = xx[ t1_3, t2_0 ]
    x_31 = xx[ t1_3, t2_1 ]
    x_32 = xx[ t1_3, t2_2 ]
    x_33 = xx[ t1_3, t2_3 ]
    
    # generate the 16 weights which will be used to combine the x_ij
    #
    # note:
    #    phi(u) = { 1.5 |u|^3 - 2.5 |u|^2 + 1           0 <= |u| <= 1   (0)
    #             { -.5 |u|^3 + 2.5 |u|^2 - 4 |u| + 2   1 <= |u| <= 2   (1)
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (|u| = u)
    u = tau1 - t1_0
    a0 = -.5 * u ** 3 + 2.5 * u ** 2 - 4 * u + 2
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (|u| = u)
    u = tau1 - t1_1
    a1 = 1.5 * u ** 3 - 2.5 * u ** 2 + 1 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (|u| = - u)
    u = tau1 - t1_2
    a2 = -1.5 * u ** 3 - 2.5 * u ** 2 + 1
 
    # 3: here, we are in case (1)
    #          and u is negative (|u| = - u)
    u = tau1 - t1_3
    a3 = .5 * u ** 3 + 2.5 * u ** 2 + 4 * u + 2

    # phi_dot(u) = {  4.5 sgn(u) u^2 - 5 u              0 <= |u| <= 1   (0)
    #              { -1.5 sgn(u) u^2 + 5 u - 4 sgn(u)   1 <= |u| <= 2   (1)    
    
    # 0: here, we are in case (1), because t1_0 + 1 <= tau1 <= t1_0 + 2
    #          and u is positive (sgn(u) = 1)
    u = tau2 - t2_0
    b0 = -1.5 * u ** 2 + 5 * u - 4 
    
    # 1: here, we are in case (0), because t1_1 <= tau1 <= t1_0 + 1 
    #          and u is positive (sgn(u) = 1)
    u = tau2 - t2_1
    b1 = 4.5 * u ** 2 - 5 * u 
    
    # 2: here, we are in case (0) because tau1 <= t1_2 <= tau1 + 1
    #          and u is negative (sgn(u) = -1)
    u = tau2 - t2_2
    b2 = -4.5 * u ** 2 - 5 * u
 
    # 3: here, we are in case (1)
    #          and u is negative (sgn(u) = -1)
    u = tau2 - t2_3
    b3 = 1.5 * u ** 2 + 5 * u + 4
    
    dx_pr_dtau2 = ((a0*b0)[...,None] * x_00 
            + (a0*b1)[...,None] * x_01
            + (a0*b2)[...,None] * x_02
            + (a0*b3)[...,None] * x_03
            + (a1*b0)[...,None] * x_10 
            + (a1*b1)[...,None] * x_11
            + (a1*b2)[...,None] * x_12
            + (a1*b3)[...,None] * x_13
            + (a2*b0)[...,None] * x_20 
            + (a2*b1)[...,None] * x_21
            + (a2*b2)[...,None] * x_22
            + (a2*b3)[...,None] * x_23
            + (a3*b0)[...,None] * x_30 
            + (a3*b1)[...,None] * x_31
            + (a3*b2)[...,None] * x_32
            + (a3*b3)[...,None] * x_33)
    
    return dx_pr_dtau2