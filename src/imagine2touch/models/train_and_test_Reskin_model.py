from ftplib import error_reply
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from calibration import dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from models import vanilla_model
from tqdm import tqdm
import sys
from torch.utils.tensorboard import SummaryWriter


def load_xyFz_model_and_scaling():
    model = vanilla_model(
        15,
        feature_dim=40,
        feat_hidden=[200, 200],
        activation_fn=nn.ReLU,
        feat_activation=None,
        output_hidden=[200, 200],
        output_activation=nn.ReLU,
    )

    input_scaling = np.loadtxt("./reskin/models/input_scaling.txt")
    output_scaling = np.array([1.0 / 16, 1.0 / 16, 1 / 3.0])
    model.load_state_dict(torch.load("./reskin/models/weights"))

    return model, input_scaling, output_scaling


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(
            "pass path to experiment directory followed by a true or false value for binary script experiment"
        )
        exit()

    # Hyperparameters
    input_size = 15  # 5 magnetometers each has bx,by and bz
    learning_rate = 0.0001
    batch_size = 32
    num_epochs = 40

    input, targets = dataset.prepare_reskin_data(sys.argv[1], sys.argv[2])
    # input_binary, targets_binary = dataset.prepare_reskin_data(sys.argv[3],sys.argv[4])
    # input = np.vstack((input,input_binary))
    # targets = np.vstack((targets,targets_binary))

    data = np.hstack((input, targets))
    np.random.shuffle(data)
    input = torch.FloatTensor(data[:, :15])
    targets = torch.FloatTensor(data[:, 15:])

    # set device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    inputs_train = input[: int(len(input) * 0.8)]
    targets_train = targets[: int(len(targets) * 0.8)]
    inputs_test = input[int(len(input) * 0.8) :]
    targets_test = targets[int(len(targets) * 0.8) :]
    TrainSet = dataset.Touch2imageSet(inputs_train, targets_train)
    TestSet = dataset.Touch2imageSet(inputs_test, targets_test)
    TrainLoader = DataLoader(TrainSet, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(TestSet, batch_size=batch_size, shuffle=True)

    model = vanilla_model(
        input_size,
        5,
        feature_dim=40,
        feat_hidden=[100, 100],
        activation_fn=nn.ReLU,
        feat_activation=None,
        output_hidden=[100, 100],
        output_activation=nn.LeakyReLU,
        pred_Fz=True,
    ).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    comment = (
        "LR_"
        + str(learning_rate)
        + "_BS_"
        + str(batch_size)
        + "_epochs_"
        + str(num_epochs)
    )
    writer = SummaryWriter(comment=comment)

    # fig=plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    for epoch in tqdm(range(num_epochs)):
        for batch_idx, (data, targets) in enumerate(TrainLoader):
            data = data.to(device)
            targets = targets.to(device)
            scores = model(data)
            # if (epoch == num_epochs -1):
            #     # graph = scores.cpu().detach().numpy()

            #     # ax.scatter(xs=graph[:,0],ys=graph[:,1],zs=graph[:,2])
            #     # ax.set_xlabel('x')
            #     # ax.set_ylabel('y')
            #     # ax.set_zlabel('Fz')
            loss = criterion(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        writer.add_scalar("Loss/train", loss, epoch)
    # plt.show()


def check_accuracy(
    loader, model
):  # accuracy here defined for an output class as percentage of no. of correctly predicted points/total no. of points with a a manually defined tolerance
    # counters for calculating accuracy
    num_correct_x = 0
    num_correct_y = 0
    # num_correct_F_z =0
    num_correct_z = 0
    num_correct_l = 0
    num_correct_m = 0
    num_samples = 0

    # set model to evaluation
    model.eval()

    # set up visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    # ax = fig.add_subplot()

    # Arrays to store date for further analysis
    # predictions = []
    # targets_x =[]
    # targets_y=[]
    # targets_z=[]
    # inputs = []
    # errors_x =[]
    # errors_y =[]
    # errors_z =[]

    # variable to sum all losses an finally print its mean
    loss = 0

    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # main evaluation procedure
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            probs = torch.softmax(scores, -1)
            # loss = loss + criterion(scores,y)

            # counters for accuracy calculations
            num_correct_x += torch.sum(
                torch.where(abs(y[:, 0] - probs[:, 0]) < 0.1, 1, 0)
            )
            num_correct_y += torch.sum(
                torch.where(abs(y[:, 1] - probs[:, 1]) < 0.1, 1, 0)
            )
            num_correct_z += torch.sum(
                torch.where(abs(y[:, 2] - probs[:, 2]) < 0.1, 1, 0)
            )
            num_correct_l += torch.sum(
                torch.where(abs(y[:, 3] - probs[:, 3]) < 0.1, 1, 0)
            )
            num_correct_m += torch.sum(
                torch.where(abs(y[:, 4] - probs[:, 4]) < 0.1, 1, 0)
            )

            num_samples += scores.size(
                0
            )  # number_of_samples_forwarded_through_model_till_now

            # detach inputs and outpurs from gpu and turn to numpy
            graph = probs.cpu().detach().numpy()  # predicted x,y, and force
            # targets = y.cpu().detach().numpy() # true x , y , and force
            # bs = x.cpu().detach().numpy() # 15 magnetic field data

            # calculate explicit errors
            # error_x = abs(targets[:,0] - graph[:,0])
            # error_y = abs(targets[:,1] - graph[:,1])
            # error_z = abs(targets[:,2] - graph[:,2])
            # for ex in error_x:
            #     errors_x.append(ex)
            # for ey in error_y:
            #     errors_y.append(ey)
            # for ez in error_z:
            #     errors_z.append(ez)

            # fill graph to show it later
            # ax.scatter(xs=graph[:,0],ys=graph[:,1],zs=graph[:,2])

            # record data
            # for element in graph:
            #     predictions.append(element)
            # for x in targets[:,0]:
            #     targets_x.append(x)
            # for y in targets[:,1]:
            #     targets_y.append(y)
            # for z in targets[:,2]:
            #     targets_z.append(z)
            # for input in bs :
            #     inputs.append(input)

        # print calculated info
        # print(loss/i)
        print("total number of test samples is {}".format(num_samples))
        print("T precision is {}".format(num_correct_x / num_samples))
        print("C precision is {}".format(num_correct_y / num_samples))
        print("V precision is {}".format(num_correct_z / num_samples))
        print("M precision is {}".format(num_correct_l / num_samples))
        print("S precision is {}".format(num_correct_m / num_samples))

    # turn test true and predicted data to numpy array for storeage
    # predictions=np.array(predictions)
    # xs =np.array(targets_x)
    # ys =np.array(targets_y)
    # zs =np.array(targets_z)
    # inputs =np.array(inputs)
    # errors_x = np.array(errors_x)
    # errors_y = np.array(errors_y)
    # errors_z = np.array(errors_z)

    # funish model eval
    model.train()

    # visualize data
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # # ax.set_zlabel('Fz')
    # plt.show()
    # store test data
    # with open('./predictions','wb') as predictions_file :
    #     np.save(predictions_file,predictions)
    # with open('./xs','wb') as xs :
    #     np.save(xs,targets_x)
    # with open('./ys','wb') as ys :
    #     np.save(ys,targets_y)
    # with open('./zs','wb') as zs :
    #     np.save(zs,targets_z)
    # with open('./bs','wb') as bs :
    #     np.save(bs,inputs)
    # with open('./ex','wb') as ex :
    #     np.save(ex,errors_x)
    # with open('./ey','wb') as ey :
    #     np.save(ey,errors_y)
    # with open('./ez','wb') as ez :
    #     np.save(ez,errors_z)


check_accuracy(TestLoader, model)


# Inputs to the model must be (Change in Magnetic Field/input_scaling)
# Model output must be scaled as (output/output_scaling). Units are (mm,mm,N).
