import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision 
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import models
# from torchsummary import summary
# from tqdm import tqdm_notebook as tqdm
# import os
from timeit import default_timer as timer
from tqdm import tqdm
import warnings
import copy
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(confusion_matrix,save_path,model_name):
  
    for i in range(len(confusion_matrix)):
        epoch = i+1
        file_name = save_path + model_name + "_Epoch_"+ str(epoch) + "_Confusion_Matrix.png"
        cfm = confusion_matrix[i]
        cm = np.array([[cfm[0],cfm[1]],[cfm[2],cfm[3]]])
        df_cm = pd.DataFrame(cm, range(2), range(2))
        sns.set(font_scale=1.4) # for label size
        cfm_fig = sns.heatmap(df_cm, annot=True, annot_kws={"size": 16},cmap="YlGnBu") # font size
        x_labels = ["Actual Positive","Actual Negative"]
        y_labels = ["Predict Positive","Predict Negative"]
        title = "Epoch_" + str(epoch) + " Confusion Matrix"
        cfm_fig.set(title=title, xticklabels=x_labels, yticklabels=y_labels)
        # plt.show()
        fig = cfm_fig.get_figure()
        fig.savefig(file_name)

def plot_curve(data, save_path, title, model_name):
    plt.clf()
    x_data = [i+1 for i in range(len(data))]
    y_data = [data[i] for i in range(len(data))]
    file_name = save_path + model_name + "_" + title + ".png"
    plt.plot(x_data,y_data)
    plt.title(title)
    plt.xlabel("Epoch")
    # plt.show()
    plt.savefig(file_name)

def saveLogfile(data, save_path, file_name):
    header = "Epoch,Accuracy\n"
    save_path = save_path + file_name + ".csv"
    with open(save_path,'w') as file:
        file.write(header)
        for i in range(len(data)):
            file.write(str(i+1) + ',' + str(data[i])+'\n')

def saveConfusionMatrix(data, save_path, file_name):
    header = "Epoch,TP,FP,FN,TN\n"
    save_path = save_path + file_name + ".csv"
    with open(save_path,'w') as file:
        file.write(header)
        for i in range(len(data)):
            s = "{},{},{},{},{}\n".format(i+1,data[i][0],data[i][1],data[i][2],data[i][3])
            file.write(s)


def get_pretrained_model(model_name,n_classes,freeze=True):

    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        if freeze:
          for param in model.parameters():
              param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        if freeze:
          for param in model.parameters():
              param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        if freeze:
          for param in model.parameters():
              param.requires_grad = False
        n_inputs = model.classifier[6].in_features

        # Add on classifier
        model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))
    elif model_name == 'inception_v3':
        model = models.inception_v3(aux_logits=False,pretrained=False)
        if freeze:
          for param in model.parameters():
              param.requires_grad = False

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, n_classes))
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)

        if freeze:
          for param in model.parameters():
              param.requires_grad = False
        n_inputs = model.classifier[1].in_features

        # Add on classifier
        model.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, n_classes))
            
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=False)

        if freeze:
          for param in model.parameters():
              param.requires_grad = False
        n_inputs = model.classifier[1].in_features

        # Add on classifier
        model.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, n_classes))
    return model



def images_transforms(phase,IMAGE_SIZE):
    if phase == 'training':
        data_transformation =transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomEqualize(10),
            transforms.RandomRotation(degrees=(-25,20)),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    else:
        data_transformation=transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    
    return data_transformation

def imshow(img):
    plt.figure(figsize=(20, 20))
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def Convlayer(in_channels,out_channels,kernel_size,padding=1,stride=1):
    conv =  nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    return conv

class NeuralNet(nn.Module):
    def __init__(self,num_classes):
        super(NeuralNet,self).__init__()
        
        self.conv1 = Convlayer(in_channels=3,out_channels=32,kernel_size=3)
        self.conv2 = Convlayer(in_channels=32,out_channels=64,kernel_size=3)
        self.conv3 = Convlayer(in_channels=64,out_channels=128,kernel_size=3)
        self.conv4 = Convlayer(in_channels=128,out_channels=256,kernel_size=3)
        self.conv5 = Convlayer(in_channels=256,out_channels=512,kernel_size=3)
        
        self.classifier = nn.Sequential(
            nn.Linear(512*4*4, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=x.view(-1,512*4*4)
        x=self.classifier(x)

        return x

class ResNet50(nn.Module):
   def __init__(self,num_class,pretrained_option=False):
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained_option)
        
        if pretrained_option==True:
            for param in self.model.parameters():
                param.requires_grad=False

        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)
        
   def forward(self,X):
        out=self.model(X)
        return out

def training(model, train_loader, test_loader, Loss, optimizer, epochs, device, num_class, save_model_path):
    model.to(device)
    best_model_wts = None
    best_evaluated_acc = 0
    best_epoch = 1
    train_acc = []
    test_acc = []
    test_Recall = []
    test_Precision = []
    test_F1_score = []
    test_Confusion_matrix = []
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer , gamma = 0.96)
    overall_start = timer()
    for epoch in range(1, epochs+1):
        print ("Epoch : {}/{}".format(epoch,epochs))            
        with torch.set_grad_enabled(True):
            model.train()
            start = timer()
            total_loss=0
            correct=0
            print("Training Phase")
            print("==================================================")
            for idx,(data, label) in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                        
                data = data.to(device,dtype=torch.float)
                label = label.to(device,dtype=torch.long)

                predict = model(data)      

                loss = Loss(predict, label.squeeze())

                total_loss += loss.item()
                pred = torch.max(predict,1).indices
                correct += pred.eq(label).cpu().sum().item()
                        
                loss.backward()
                optimizer.step()

            total_loss /= len(train_loader.dataset)
            correct = (correct/len(train_loader.dataset))*100.
            print ("Training Loss : " , total_loss)
            print ("Training Accuracy : " , correct ,"%")
            #print(epoch, total_loss, correct)     
        scheduler.step()
        print("Testing Phase")
        print("==================================================")         
        accuracy  , Recall , Precision , F1_score, Confusion_matrix = evaluate(model, device, test_loader)
        train_acc.append(correct)  
        test_acc.append(accuracy)
        test_Recall.append(Recall)
        test_Precision.append(Precision)
        test_F1_score.append(F1_score)
        test_Confusion_matrix.append(Confusion_matrix)

        if accuracy > best_evaluated_acc:
            print("Updating(copying) Weights of Best model")
            best_evaluated_acc = accuracy
            best_model_wts = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            torch.save(best_model_wts, save_model_path+"_epoch" + str(best_epoch) + ".pt")
        elapsed_time = timer() - start
        print(f'Elapsed time :{elapsed_time:.2f} (sec)')
        print(" ")
        
    total_time = timer() - overall_start    
    print("Training Completed!!")
    print(f'Total time:{total_time:.2f} (sec)')
    print(f'Time per epoch:{total_time / (epochs):.2f} (sec/epoch)') 
    print("Best Model was saved at epoch {} with testing accuracy {}".format(best_epoch,best_evaluated_acc))

    #save model
    torch.save(best_model_wts, save_model_path+"_epoch" + str(best_epoch) + ".pt")
    # model.load_state_dict(best_model_wts)

    return train_acc , test_acc , test_Recall , test_Precision , test_F1_score , test_Confusion_matrix

def evaluate(model, device, test_loader):
    correct=0
    TP=0
    TN=0
    FP=0
    FN=0
    with torch.set_grad_enabled(False):
        model.eval()
        for idx,(data,label) in enumerate(test_loader):
            data = data.to(device,dtype=torch.float)
            label = label.to(device,dtype=torch.long)
            predict = model(data)
            pred = torch.max(predict,1).indices
            #correct += pred.eq(label).cpu().sum().item()
            for j in range(data.size()[0]):
                #print ("{} pred label: {} ,true label:{}" .format(len(pred),pred[j],int(label[j])))
                if (int (pred[j]) == int (label[j])):
                    correct +=1
                if (int (pred[j]) == 1 and int (label[j]) ==  1):
                    TP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  0):
                    TN += 1
                if (int (pred[j]) == 1 and int (label[j]) ==  0):
                    FP += 1
                if (int (pred[j]) == 0 and int (label[j]) ==  1):
                    FN += 1

        Confusion_matrix = [TP,FP,FN,TN]

        print ("TP : " , TP)
        print ("TN : " , TN)
        print ("FP : " , FP)
        print ("FN : " , FN)

        print ("num_correct :",correct ," / " , len(test_loader.dataset))
        Recall = TP/(TP+FN)
        print ("Recall : " ,  Recall )

        Precision = TP/(TP+FP)
        print ("Precision : " ,  Precision )

        F1_score = 2 * Precision * Recall / (Precision + Recall)
        print ("F1 - score : " , F1_score)

        correct = (correct/len(test_loader.dataset))*100.
        print ("Testing Accuracy : " , correct ,"%")

    return correct , Recall , Precision , F1_score, Confusion_matrix

if __name__=="__main__":
    img_size=(128, 128)
    batch_size= 32
    learning_rate = 0.02
    epochs=100
    num_classes=2
    root = "/home/10711007/covid19/"
    model_name = "efficientnet_b7"
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print (device)

    train_path = root + 'data/train'
    test_path = root + 'data/test'


    trainset=datasets.ImageFolder(train_path,transform=images_transforms('train',img_size))
    testset=datasets.ImageFolder(test_path,transform=images_transforms('test',img_size))
    
    train_loader = DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=1)
    test_loader = DataLoader(testset,batch_size=batch_size,shuffle=True,num_workers=1)
    
    examples=iter(train_loader)
    images,labels=examples.next()
    print(images.shape)
    # imshow(torchvision.utils.make_grid(images[:56],pad_value=20))

    # model = model = ResNet50(2, True)
    model = get_pretrained_model(model_name=model_name, n_classes=num_classes, freeze=False)
    Loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # print (summary(model,(3,128,128)))

    print (train_loader)
    dataiter = iter(train_loader)
    images , labels = dataiter.next()
    print (type(images) , type(labels))
    print (images.size(),labels.size())
    
    save_model_path = root + "models/" + model_name # saving path of models
    train_acc , test_acc , test_Recall , test_Precision , test_F1_score , test_Confusion_matrix = training(model, train_loader, test_loader, Loss, optimizer, epochs, device, 2, save_model_path)
    
    
    save_results_path = root + "results/" # saving path of results
    
    # Saving training and testing results as log file
    saveLogfile(train_acc, save_results_path, "Train_Accuracy_"+model_name)
    saveLogfile(test_acc, save_results_path, "Test_Accuracy_"+model_name)
    saveLogfile(test_Recall, save_results_path, "Test_Recall_"+model_name)
    saveLogfile(test_Precision, save_results_path, "Test_Precision_"+model_name)
    saveLogfile(test_F1_score, save_results_path, "Test_F1_score_"+model_name)
    saveConfusionMatrix(test_Confusion_matrix, save_results_path, "Test_Confusion_Matrix_"+model_name)

    # Visualization of training and testing results
    
    plot_curve(train_acc, save_results_path, "Train_Accuracy", model_name)
    plot_curve(test_acc, save_results_path, "Test_Accuracy", model_name)
    plot_curve(test_Recall, save_results_path, "Test_Recall", model_name)
    plot_curve(test_Precision, save_results_path, "Test_Precision", model_name)
    plot_curve(test_F1_score, save_results_path, "Test_F1_score", model_name)
    # plot_confusion_matrix(test_Confusion_matrix, save_results_path, model_name)