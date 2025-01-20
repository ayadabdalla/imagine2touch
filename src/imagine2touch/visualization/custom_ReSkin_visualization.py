# standard libraries
import pygame
from pygame.locals import *
import sys
import math
import time
import numpy as np
from argparse import ArgumentParser
import os


# repo modules
from src.imagine2touch.reskin_sensor import ReSkinBase
from src.imagine2touch.reskin_calibration import dataset


def init_pygame():
    """
    initialize pygame
    """
    time.sleep(1)
    pygame.init()  # initialize pygame
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((420, 600))
    bg = pygame.image.load(f"{dir_path}/images/3D.PNG")
    pygame.mouse.set_visible(1)

    pygame.display.set_caption("5X Board Visual")
    return clock, screen, bg


def get_baseline(sens, num_samples):
    """
    get average of <num_samples> from a <sens>
    """
    print("Leave the sensor in a rest state")
    time.sleep(2.0)

    baseline_samples = sens.get_data(num_samples)
    baseline = [s.data for s in baseline_samples]
    baseline = np.array(baseline)
    baseline = np.mean(baseline, axis=0)
    print("Resting data collected.")

    return baseline


def draw_vectors(input_data):
    """
    draw measured magnetic vector at each of the 5 magnetometers on ReSkin, input_data is a flattened 1-D array of 3D vectors
    """
    for idx in range(5):
        center_arrow = chip_locations[
            idx
        ]  # center vector tail at magnetometers locations
        angle = math.atan2(
            input_data[3 * idx + 1], input_data[3 * idx]
        )  # angle between x and y coordinates of the target vector
        z = math.sqrt(
            (
                input_data[3 * idx] * input_data[3 * idx]
                + input_data[3 * idx + 1] * input_data[3 * idx + 1]
            )
        )  # 2D target vector magnitude
        x = center_arrow[0] + math.sin(angle) * z  # vector head point x
        y = center_arrow[1] + math.cos(angle) * z  # vector head point y
        r = (
            abs(input_data[3 * idx + 2]) / scale
        )  # circle representing the magnitude of the 3rd co-ordinate of the 3D vector
        pygame.draw.line(
            screen, (0, 0, 0), center_arrow, (x, y), 5
        )  # draw 2D vector representing first 2 co-ordinates
        if input_data[3 * idx + 2] <= 0:
            pygame.draw.circle(
                screen, (0, 0, 255), center_arrow, r, 70
            )  # draw purple circle representing 3rd co-ordinate
        else:
            pygame.draw.circle(
                screen, (255, 0, 0), center_arrow, r, 70
            )  # draw purple circle representing 3rd co-ordinate
    pygame.display.update()


def isolate_data_vector(data, i):
    """
    get one 5-3Dvector (reading) from a collection of 5-3Dvectors, scale it & rotate it to fit pygame frame
    """
    input_data = data[i] * 100  # scale vector
    input_data[0:3] = [
        input_data[1],
        -1 * input_data[0],
        -1 * input_data[2],
    ]  # magnetometer 1 data
    input_data[3:6] = [
        input_data[4],
        -1 * input_data[3],
        -1 * input_data[5],
    ]  # magnetometer 2 data
    input_data[6:9] = [
        input_data[6],
        input_data[7],
        -1 * input_data[8],
    ]  # magnetometer 3 data
    input_data[9:12] = [
        -1 * input_data[9],
        -1 * input_data[10],
        -1 * input_data[11],
    ]  # magnetometer 4 data
    input_data[12:15] = [
        -1 * input_data[13],
        input_data[12],
        -1 * input_data[14],
    ]  # magnetometer 5 data
    return input_data


def draw_point(x_pose, y_pose, i, color):
    """
    draw target1 and target2 of data point at index <i> as x & y co-ordinates
    """

    pygame.draw.circle(
        screen,
        color,
        [np.asscalar(x_pose[i]) * 24.5, -1 * np.asscalar(y_pose[i]) * 24.5 + 410],
        10,
        100,
    )
    pygame.display.update()


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.realpath(__file__))
    parser = ArgumentParser(description="Training Script")
    parser.add_argument("experiment_directory", help="Experiment directory")
    parser.add_argument("object_name", help="Object Name")

    args = parser.parse_args()

    # #model predictions
    # predicted_targets = np.load('./custom_ReSkin_visualization_example_ReSkin_data/predictions', allow_pickle=True)

    # #ground truth labels
    # xs = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/xs',allow_pickle=True),(-1,1))
    # ys = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/ys',allow_pickle=True),(-1,1))
    # zs = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/zs',allow_pickle=True),(-1,1))

    # #input data
    # data = np.load('./custom_ReSkin_visualization_example_ReSkin_data/bs',allow_pickle=True)
    data, _ = dataset.prepare_reskin_data(
        f"{args.experiment_directory}/{args.object_name}_tactile", True
    )

    # #error between predictions and ground truth
    # ex = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/ex',allow_pickle=True),(-1,1))
    # ey = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/ey',allow_pickle=True),(-1,1))
    # ez = np.reshape(np.load('./custom_ReSkin_visualization_example_ReSkin_data/ez',allow_pickle=True),(-1,1))

    # stack all data
    # stack = np.hstack((predictions,xs,ys,zs,data,ex,ey,ez))
    # print(stack.shape)

    # Define constants
    WHITE = pygame.Color(255, 255, 255)
    RED = pygame.Color(255, 0, 0)
    BLACK = pygame.Color(0, 0, 0)
    # viz_sensor = ReSkinBase(num_mags=5, port='/dev/ttyACM0', baudrate=115200) #sensor
    scale = 10  # reading to size on image factor

    # filter out temperature readings
    # temp_mask = np.ones((20,),dtype=bool)
    # temp_mask[::4] = False

    # instantiate pygame objects
    clock, screen, bg = init_pygame()

    # # read first 100 samples as baseline
    # numBaselineSamples = 100
    # baseline = get_baseline(viz_sensor, numBaselineSamples)

    # # chip locations in pixels on the game board
    # # in order of center, top, right, bottom, left to match incoming data stream
    chip_locations = np.array(
        [[211, 204], [211, 60], [357, 206], [211, 353], [67, 204]]
    )

    # preprocess data from another script
    # data, targets = dataset.prepare_reskin_data(sys.argv[1],sys.argv[2])

    i = -1  # used as counter for data and labels

    # x_pose = predicted_targets [:,0] #predicted target 1
    # y_pose = predicted_targets [:,1] #predicted target 2
    # f = predicted_targets[:,2] #predicted target 3
    while True:
        ## process data from sensor directly
        # raw_data = viz_sensor.get_data(1)
        # input_data = raw_data[0].data - baseline
        # #rotation of chip axes to pygame coordinate system
        # input_data = input_data[temp_mask]

        screen.blit(bg, (0, 0))  # refresh gui
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            ## get a new baseline on demand
            # elif event.type == KEYDOWN:
            #     if event.key == ord('b'):
            #         baseline = get_baseline(viz_sensor, numBaselineSamples)
            elif event.type == KEYDOWN:
                if event.key == ord("n"):
                    i += 1  # get next target point
                    draw_vectors(
                        isolate_data_vector(data, i)
                    )  # example data vector input

                    # rename true labels
                    # x_pose_true = xs
                    # y_pose_true = ys
                    # f_true = zs

                    # predicte y, true y, and error between them
                    # font1 = pygame.font.SysFont(str(y_pose_true[i]), 30)
                    # text = font1.render("ty "+str(y_pose_true[i]), True, (0,0,0))# true y (target 1)
                    # screen.blit(text, (180, 20))
                    # pygame.display.update()

                    # font2 = pygame.font.SysFont(str(y_pose[i]), 30)
                    # text2 = font2.render("py "+str(y_pose[i]), True, (0,0,0))# predicted y (target 2)
                    # screen.blit(text2, (180, 40))
                    # pygame.display.update()

                    # font3 = pygame.font.SysFont(str(ey[i]), 30)
                    # text3 = font3.render("ey "+str(ey[i]), True, (0,0,0))# error in y (target 3)
                    # screen.blit(text3, (180, 60))
                    # pygame.display.update()

                    # predicted x, true x, and error between them
                    # font4 = pygame.font.SysFont(str(x_pose_true[i]), 30)
                    # text4 = font4.render("tx "+str(x_pose_true[i]), True, (0,0,0)) # true x (target 1)
                    # screen.blit(text4, (20, 20))
                    # pygame.display.update()

                    # font5 = pygame.font.SysFont(str(x_pose[i]), 30)
                    # text5 = font5.render("px "+str(x_pose[i]), True, (0,0,0)) # predicted x (target 1)
                    # screen.blit(text5, (20, 40))
                    # pygame.display.update()

                    # font6 = pygame.font.SysFont(str(ex[i]), 30)
                    # text6 = font6.render("ex "+str(ex[i]), True, (0,0,0)) # error in x (target 1)
                    # screen.blit(text6, (20, 60))
                    # pygame.display.update()

                    # predicte, f true f, and error between them
                    # font7 = pygame.font.SysFont(str(f_true[i]), 30)
                    # text7 = font7.render("tf "+str(f_true[i]), True, (0,0,0)) # true force (target 3)
                    # screen.blit(text7, (20, 120))
                    # pygame.display.update()

                    # font8 = pygame.font.SysFont(str(f[i]), 30)
                    # text8 = font8.render("pf "+str(f[i]), True, (0,0,0)) # predicted force (target 3)
                    # screen.blit(text8, (20, 140))
                    # pygame.display.update()

                    # font9 = pygame.font.SysFont(str(ez[i]), 30)
                    # text9 = font9.render("ef "+str(ez[i]), True, (0,0,0)) # error in force (target 3)
                    # screen.blit(text9, (20, 160))
                    # pygame.display.update()

                    # draw_point(x_pose,y_pose,i,(255,255,0)) # predicted pose (target1,target2)
                    # draw_point(x_pose_true,y_pose_true,i,(255,0,0)) # true pose (true target1, true target2)
