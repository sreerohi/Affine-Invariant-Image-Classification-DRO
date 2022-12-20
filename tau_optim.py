#Class for batch image interpolation and Tau optimization
class BatchInterpolOptim(nn.Module):
  def __init__(self, batchSize, M, N, loss_fn, step_size, model, device):
    super(BatchInterpolOptim, self).__init__()

    self.batchSize = batchSize
    self.M = M
    self.N = N
    self.lossFn = loss_fn 
    self.stepSize = step_size
    self.model = model
    self.device = device
    self.nclasses = 10


    # self.taus  = np.array(10*[np.random.rand(2, M,N)]) # The 10 vector fields, one for each label. Stored as Numpy elements.
    ivf = affine_to_vf2(np.eye(2), np.zeros(2,), self.M, self.N)
    self.taus  = np.array(self.nclasses*[ivf]) # The 10 vector fields, one for each label. Stored as Numpy elements.
    self.finalGradTaus = np.zeros((self.nclasses, 2, self.M, self.N))

    # self.taus  = np.array(5*[[ivf[0], ivf[1]]] + 5*[[t0,t1]]) # Testing the batchInterpolation function

    #Storage containers necessary for one optimization step on all 10 taus:
    self.indices = [[] for _ in range(self.nclasses)]

    # self.predictions = torch.Tensor((self.batchSize, 10)).to(device)
    # self.predictions.requires_grad_ = False

    self.interpolated_gradients = torch.Tensor(np.array(self.batchSize*[np.zeros((1, self.M,self.N))])).to(device) # Average Gradients of the loss function w.r.t Interpolated Images
    self.interpolated_gradients.requires_grad = False

    self.interpolated_images = torch.Tensor(np.array(self.batchSize*[np.zeros((1,self.M,self.N))])).to(device) # Saving the interpolated images here
    ## requires_grad is set in the optimize function

    self.batchVectorField = np.array(self.batchSize*[np.zeros((2,self.M,self.N))])
    self.Y = None

    # Variables for logging
    self.lossEpoch = 0

  def assignVectorFields(self,  y):
    #y is a tensor of batch labels 
    #Returns numpy array of vector fields, one for each element in the batch #(batchsize, 2, M, N)
    if(self.device.type == "cuda"):
      y = y.cpu().numpy()
    else:
      y = y.numpy()
    batchVectorField = np.array(self.batchSize*[np.zeros((2,self.M,self.N))])  # shape is (batchsize, 2, M, N), used for storing vector fields for the whole batch
    for i in range(0,self.nclasses):
      indices  = np.where(y==i)[0]
      self.indices[i] = indices
      batchVectorField[indices] = self.taus[i] # Assigning one of the 10 vector fields depending on the image label
    self.batchVectorField = batchVectorField
    return self.batchVectorField

  def forward(self, batchX, batchY):
    # batchX is a tensor of dimensions (batchsize, 1, M, N)
    # batchY is a tensor of dimesnions (batchsize, )
      self.Y = batchY
      batchVectorField = self.assignVectorFields(batchY)
      self.interpolated_images = self.image_interpolation_bicubicBatch(batchX, batchVectorField).to(self.device)
      return self.interpolated_images

  def image_interpolation_bicubicBatch(self, batchX, batchTaus):
    # batchX is a tensor of dimensions (batchsize, M, N)
    # batchTaus is a numpy array of vector fields
    # Returns a tensor of interpolated images of size (batchsize, 1, M, N)
    batchX = batchX.permute(0,2,3,1)
    if(self.device.type == "cuda"):
      batchX = batchX.cpu().numpy()
    else:
      batchX = batchX.numpy()
    returnBatchX = []
    for i in range(0, self.batchSize ):
      img  = image_interpolation_bicubic(batchX[i], batchTaus[i][0] + 13.5, batchTaus[i][1] + 13.5)
      returnBatchX.append( img )
    return torch.Tensor(np.array(returnBatchX)).permute(0,3,1,2).to(self.device)
  
  def computeInterpolatedGradients(self): #computes gradient of loss function w.r.t interpolated images and stores them in an object variable
    self.interpolated_images.requires_grad_(True) 
    self.interpolated_images.grad = None
    self.model.eval()
    predictions = self.model(self.interpolated_images)

    l = 0
    for i in range(0 , self.nclasses):
      predictionsI = predictions[self.indices[i]]
      lI = self.lossFn(predictionsI, self.Y[self.indices[i]])
      l = l + lI.item()
      lI.backward(retain_graph = True) ## For every i, only a part of the networks output is going into the ith loss function. Therfore, in terms of the omputational graph, there are 10 different loss function nodes. So after the first loop (i=0), if we don't retain graph, the 
      #part of the graph sued to compute l0 will be deleted (?) and therefore, will be a problem to backpropagate gradients from l1. Therefore, we have to retain the graph.
      #I have checked that when we do lI.backward , the gradients for images not belonging to the the i th class are not affected.
    
    self.lossEpoch = l/self.batchSize
    self.interpolated_gradients = self.interpolated_images.grad
  
  def dimage_interpolation_bicubic_dtaus_Batch(self, batchX, batchTaus):
    # batchX is a tensor of dimensions (batchsize, M, N)
    # batchTaus is a numpy array of vector fields
    # Returns two tensors representing d(x \circ \tau) / dtau1 for the whole batch  of dimensions (batchsize, 1, M, N ) and  d(x \circ \tau2)/ dtau2 for the whole batch  of dimensions(batchsize, 1, M, N )
    #dimage_interpolation_bicubic_dtau1(x,tau1,tau2)

    batchX = batchX.permute(0,2,3,1)

    if(self.device.type == "cuda"):
      batchX = batchX.cpu().numpy()
    else:
      batchX = batchX.numpy()

    BatchGradTau1 = []
    BatchGradTau2 = []

    for i in range(0, self.batchSize):
      gradTau1 = dimage_interpolation_bicubic_dtau1(batchX[i], batchTaus[i][0] + 13.5, batchTaus[i][1] + 13.5)
      BatchGradTau1.append(gradTau1)
      gradTau2 = dimage_interpolation_bicubic_dtau2(batchX[i], batchTaus[i][0]+ 13.5, batchTaus[i][1] + 13.5)
      BatchGradTau2.append(gradTau2)

    return torch.Tensor(np.array(BatchGradTau1)).permute(0,3,1,2).to(self.device), torch.Tensor(np.array(BatchGradTau2)).permute(0,3,1,2).to(self.device)

  def computegradLoss_taus(self, batchX):
    # batchX is a tensor of dimensions (batchsize, M, N)
    #Code is written assuming forward() method is called before this method

    # At this point we an compute gradients of loss w.r.t interpolated images, and gradients of the interpolated images w.r.t taus. 
    #We neet to multiply them(chain rule), add sum them with appropriate sample indices for each tau (since loss is a sum of samples) and return gradients of shape (10,2,28,28)

    # Returns: Numpy array (10,2,28,28)
    gradientLossFinal = np.zeros((self.nclasses,2,self.M,self.N))

    self.computeInterpolatedGradients()
    batchTaus = self.batchVectorField

    batchGradTau1, batchGradTau2 = self.dimage_interpolation_bicubic_dtaus_Batch(batchX, batchTaus )

    GradTau1 = self.interpolated_gradients * batchGradTau1
    GradTau2 = self.interpolated_gradients * batchGradTau2

    if(self.device.type == "cuda"):
      GradTau1 = GradTau1.cpu().numpy()
      GradTau2 = GradTau2.cpu().numpy()
    else:
      GradTau1 = GradTau1.numpy()
      GradTau2 = GradTau2.numpy()

    for i in range(0, self.nclasses):
      gradTau1I = np.sum(GradTau1[self.indices[i]] , axis = 0 )
      gradTau2I = np.sum(GradTau2[self.indices[i]] , axis = 0 )
      gradientLossFinal[i][0] = gradTau1I[0]
      gradientLossFinal[i][1] = gradTau2I[0]

    self.finalGradTaus = gradientLossFinal

  def optimize(self):
    #Assumes that self.computegradLoss_taus is called externally
    self.taus = self.taus + self.stepSize*(self.finalGradTaus) #GRADIENT ASCENT

    for i in range(0, self.nclasses):
      self.taus[i][0],self.taus[i][1] = project_tau(self.taus[i][0],self.taus[i][1], self.M,self.N, printer=0)
    #Here