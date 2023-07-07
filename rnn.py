#Importing images from cifar-10
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torchvision
import numpy as np
import copy
import scipy as sc
import scipy.special
from sklearn.linear_model import LinearRegression
import torchvision.models as models
import time


from athena.active import ActiveSubspaces

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')



def get_train_test_data(dataset, batch_size, augm = False, train_amt = 50000):
    """
    dataset: string with name
    batch_size: scalar
    augm: boolean of whether you want augmented data
    returns a test and train loader
    """
    if dataset=='CIFAR':
        normalize = transforms.Normalize(mean = [0.49139968, 0.48215841, 0.44653091],
        std = [0.24703223, 0.24348513, 0.26158784])
    
    elif dataset=='SVHN':
        normalize = transforms.Normalize(mean = [0.45141584, 0.45141453, 0.45142587],
        std = [0.19929032, 0.1992932,  0.19929022])
    
    transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
    
    if augm==True:
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),normalize
                ])
                    
        test_transform = transforms.Compose([
                transforms.ToTensor(),normalize
                ])
    
    if dataset=='CIFAR':
        train_dataset = torchvision.datasets.CIFAR10(
        root= './data', train = True,
        download =True, transform = transform)
        
        test_dataset = torchvision.datasets.CIFAR10(
        root= './data', train = False,
        download =True, transform = test_transform)
    
    elif dataset=='SVHN':
        train_dataset = torchvision.datasets.SVHN(root='./data', split = 'train', download=True, transform = transform)
        
        test_dataset = torchvision.datasets.SVHN(root='./data', split = 'test', download=True, transform = test_transform)
        
    train_dataset = torch.utils.data.Subset(train_dataset,np.arange(train_amt))
        
    
    train_loader = torch.utils.data.DataLoader(train_dataset
    , batch_size = batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset
    , batch_size = batch_size)
    
    return train_loader, test_loader        
    #train_subset = torch.utils.data.Subset(train_dataset,np.arange(subset_amt))
    


def get_full_model(dataset, augm = False):
    """
    dataset: string with name of dataset
    augm: True/False dependent on whether you have augmented data
    returns the full model
    """

    vgg = torchvision.models.vgg16()
    input_lastLayer = vgg.classifier[6].in_features
    vgg.classifier[6] = nn.Linear(input_lastLayer,10)
    
    state_dict = torch.load('full_vgg16_'+str(dataset)+'_augm'+str(augm)+'.pth')
    
    vgg.load_state_dict(state_dict)
    
    return vgg


def get_pre_post_model(model, cutoff):
    """
    model:model
    cutoff: l+1 of the cutoff layer
    """
    
    seq_mod = nn.Sequential(*(list(model.features.children()) +
                                    [nn.AdaptiveAvgPool2d((7, 7))] +
                                    [nn.Flatten(1, -1)]+
                                    list(model.classifier.children())))
    pre = seq_mod[:cutoff]
    post = seq_mod[cutoff:]

    return pre, post

def get_output_matrix(model, train_loader):
    """
    input: model, train loader
    output: a matrix where the columns denote the samples and the rows denote the output of the neurons of a layer.
    can also be used for predictions.
    This works as a full model output as well
    """
    
    batch_size = train_loader.batch_size

    params = torch.tensor(model(torch.zeros(1,3,32,32)).shape).prod().item()

    container = torch.empty(params,len(train_loader.dataset))

    with torch.no_grad():
        if str(type(model))!="<class 'function'>":
            model.eval()
            model.to(device)

        for i, (images , labels) in enumerate(train_loader):

            images = images.to(device)

            container[:,i*batch_size:(i*batch_size)+ batch_size] = (model(images).reshape(len(images),params).T)


    return container


def POD(pre, train_loader, r=50):
    
    container = get_output_matrix(pre, train_loader)
    
    #torch is faster. Using numpy for stability reasons
    #U = torch.linalg.svd(container, full_matrices = False) [0]
    U = torch.from_numpy(np.linalg.svd(container,full_matrices = False)[0])

    U_reduced = U[:,:r]

    z = (U_reduced.T@container)

    return z, U_reduced


def AS(pre, train_loader, post, r=50):

    asubs = ActiveSubspaces(dim = r, method='exact')
    
    container = get_output_matrix(pre, train_loader)
    
    container_AS = torch.empty(container.shape)
    
    batch_size = train_loader.batch_size
    pre_size = len(container)
    
    if str(type(pre))!="<class 'function'>":
        pre.train()
        pre.to(device)

    for i, (train_images , train_labels) in enumerate(train_loader):

        train_images = train_images.to(device)
        train_labels = train_labels.to(device)


        the_input = pre(train_images)
        
        #if the_input.requires_grad==False:
        #    the_input.requires_grad_()
        
        output = post(the_input)

        loss_of_output = nn.functional.nll_loss(output,train_labels)

        container_AS[:,i*batch_size:(i*batch_size)+ batch_size] = (torch.autograd.grad(loss_of_output,the_input, grad_outputs=torch.ones(loss_of_output.size()).to(dtype=the_input.dtype,
                                                           device=the_input.device),
                                                           create_graph = True,
                                                           retain_graph = True,
                                                           allow_unused=True))[0].reshape(len(train_images), pre_size).T

    asubs.fit(gradients = container_AS.detach().cpu().numpy().T)

    proj_mat = torch.tensor(asubs.evects)

    z = proj_mat.T@container

    return z, proj_mat

def get_PCE_container(z,r=50,p=2):

    container_PCE = torch.empty(z.shape[1],int(np.math.factorial(r+p)/(np.math.factorial(p)*np.math.factorial(r))))

    #zeroth basis
    container_PCE[:, 0] = 1

    #first basis
    container_PCE[:,1:r+1] = (sc.special.hermite(1)(z).T)

    if p==2:

        #second basis
        container_PCE[:,-r:] = sc.special.hermite(2)(z).T

        #upper diagonal matrix additionally
        #why is it 1225 alltogether? need to look at
        inside = torch.empty(z.shape[1],int(np.math.factorial(r+p)/(np.math.factorial(p)*np.math.factorial(r)))-2*r-1)

        idx=0
        for i in range(r-1):

            inside[:,idx:(idx+r-1-i)] = ((sc.special.hermite(1)(z.T[:,i]).unsqueeze(1)* sc.special.hermite(1)(z.T[:,i+1:])))

            idx+=r-1-i

        container_PCE[:,r+1:-r] = inside

    return container_PCE

def get_PCE_coeffs(container, full_output):
    
    LR = LinearRegression(fit_intercept=False).fit(container,full_output.T)
    coeff = LR.coef_.transpose() 
    
    print('Approximation score: ',LR.score(container,full_output.T))
    coeff = LR.coef_.transpose() 
    y_PCE = container@coeff
    
    return coeff

class FNN(nn.Module):
    def __init__(self, input_dim = 50, hidden_dim = 20, output_dim = 10):
        super(FNN, self).__init__()
        # Linear function
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        # Non-linearity
        self.softplus = nn.Softplus()

        # Linear function (readout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Linear function  # LINEAR
        out = self.fc1(x)

        # Non-linearity  # NON-LINEAR
        out = self.softplus(out)

        # Linear function (readout)  # LINEAR
        out = self.fc2(out)
        return out
    
    
def training_fnn(fnn_net, epochs, inputs_net, real_out):
    '''
    Training phase for a Feed Forward Neural Network (FNN).
    :param nn.Module fnn_net: FNN model
    :param int epochs: epochs for the training phase.
    :param tensor inputs_net: matrix of inputs for the network
        with dimensions n_input x n_images.
    :param tensor real_out: tensor representing the real output
        of the network.
    '''
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(fnn_net.parameters(), lr=0.0001)
    correct = 0
    total = 0

    fnn_net = fnn_net.to(device)
    inputs_net = inputs_net.to(device)
    for i in range(len(real_out)):
        real_out[i] = real_out[i].to(device)

    #print(inputs_net.shape)

    final_loss = []
    batch_size = 64
    print('FNN training initialized')
    for epoch in range(epochs):  # loop over the dataset multiple times
        correct=0
        total=0
        for i in range(inputs_net.size()[0] // batch_size):
            #print(inputs_net.size()[0] // batch_size)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = fnn_net((inputs_net[i * batch_size:(i + 1) *
                                         batch_size, :]).to(device))
            loss = criterion(
                outputs,
                torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                          batch_size]).to(device))
            #dokumentasjon?
            loss.backward(retain_graph=True)
            optimizer.step()


            _, predicted = torch.max(outputs.data, 1)
            labels = torch.LongTensor(real_out[i * batch_size:(i + 1) *
                                               batch_size]).to(device)
            total += labels.size(0)



            correct += (predicted == labels).sum().item()
        #print(epoch)
        #print('correct: ', labels[predicted==labels])
        #print('incorrect: thought it was this \n', predicted[predicted!=labels], '\nbut was actually \n', labels[predicted!=labels])
        #print('loss:',loss)
        if (epoch%10==0):
            print(epoch,'loss:',loss)
        #print('accuracy:',correct/total)

    print(correct/total)


    print('FNN training completed', flush = True)
    
    
def reduced_model(dataset, augm, cutoff, red_type, inout_type, r=50, batch_size = 64, train_amt = 50000):
    
    train_loader, test_loader = get_train_test_data(dataset, batch_size, augm, train_amt)
    full_model = get_full_model(dataset, augm)
    full_output = get_output_matrix(full_model, train_loader)
    
    pre, post = get_pre_post_model(full_model,cutoff=cutoff)

    if red_type == 'POD':
        z, proj_mat = POD(pre,train_loader, r=r)
    else:
        z, proj_mat = AS(pre,train_loader,post, r=r)
        
    print(z.shape)
    print(proj_mat.shape)

    if inout_type=='FNN':

        inout_map = FNN(input_dim = r)
        training_fnn(inout_map, 500, z.T.to(device), full_output.argmax(axis=0).to(device))

    else:
        #For tomorrow: consider making this into linear regression due to rounding errors.
        container_PCE = get_PCE_container(z,r=r,p=2)
        inout_map = get_PCE_coeffs(container_PCE, full_output)
        
        #c = torch.from_numpy(np.linalg.inv(container_PCE.T @ container_PCE))@(container_PCE.T)@full_output
        #inout_map=c
        
    reduced_dict = {'pre' : pre, 'proj_mat' : proj_mat, 'inout_map' : inout_map}
    return reduced_dict
    #torch.save(reduced_dict,'reduced_model_dicts/' +dataset +str(augm) + '_idx'+str(cutoff)+'_'+str(red_type)+'_'+str(inout_type)+'.pth')


def get_loss_val(dataset, augm):
    
    num_epochs = 100
    batch_size = 64
    learning_rate = 0.0001
    train_amt = 50000
    
    train_loader, test_loader = get_train_test_data(dataset, batch_size=batch_size, augm=augm, train_amt=train_amt)
    model = models.vgg16()
    input_lastLayer = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(input_lastLayer,10)
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9,weight_decay=5e-4)


    losses = torch.zeros(num_epochs)
    vals = torch.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):

        model.train()

        for i, (imgs , labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            labels_hat = model(imgs)
            n_corrects = (labels_hat.argmax(axis=1)==labels).sum().item()
            loss_value = criterion(labels_hat, labels)
            loss_value.backward()
            optimizer.step()
            optimizer.zero_grad()

            #losses[epoch] += loss_value.detach().item()*len(labels)

            if (i+1) % 250 == 0:
              print(f'epoch {epoch+1}/{num_epochs}, step: {i+1}: loss = {loss_value:.5f}, acc = {100*(n_corrects/labels.size(0)):.2f}%, time spent:{time.time() - start_time}')

        losses[epoch] = loss_value.detach().item()

        with torch.no_grad():
            model.eval()

            for i, (imgs, labels) in enumerate(test_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                labels_hat = model(imgs)

                loss_value = criterion(labels_hat, labels)

                #vals[epoch]+= loss_value.item()*len(labels)
            vals[epoch] = loss_value.item()

    #losses = losses/len(train_loader.dataset)
    #vals = vals/len(train_loader.dataset)
    
    return losses, vals
