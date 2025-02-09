import torch

from lib.config import Config
from lib.runner import Runner
from lib.experiment import Experiment

import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2


class Prediction():
    def __init__(self):
        # The image path is /datasets/tusimple-test/clips 
        # it works special with test_label.json
        # you need to spesify the file name in the json along with the test data.
        # i am overwriting 1.jpg to change photo each time.
        #
        # The cv2 visualization is commented out from lib/runner.
        # if Runner's view parameter is all the prediction data comes in type of pixel (640,360)
        # other wise it is in between 0 and 1

        self.exp = Experiment("laneatt_r122_tusimple", mode="test")
        self.cfg = Config(self.exp.cfg_path)  
        self.exp.set_cfg(self.cfg, override=False)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        

    def predict(self):
        runner = Runner(self.cfg, self.exp, self.device, view="all")
        try:
            prediction = runner.eval(self.exp.get_last_checkpoint_epoch(), save_predictions=False)
            return prediction
        except Exception as e:
            print(e)
        
    

    def get_center_lanes(self, lanes,center_point=330, num_lanes=2):
        
        lane_centers = []
        for lane in lanes[0]:
            if hasattr(lane, 'points'):
                x_coords = [p[0] for p in lane.points]
            else:
                x_coords = lane[:, 0] if isinstance(lane, np.ndarray) else [p[0] for p in lane]
        
            avg_x = np.mean(x_coords)
            lane_centers.append((avg_x, lane))
        
        
        lane_centers.sort(key=lambda x: abs(x[0] - center_point))
        return [lane for _, lane in lane_centers[:num_lanes]]

    def middle_lane(self,preds):
        if preds:
            
            Lane1 = preds[0].points
            Lane2 = preds[1].points
            
            equalizer=0
            if len(Lane1) != len(Lane2):
                    
                if len(Lane1) > len(Lane2):
                    big_one = Lane1
                    lil_one = Lane2      
                elif len(Lane2) >= len(Lane1):
                    big_one = Lane2
                    lil_one = Lane1

                try:
                    while lil_one[0][1] != big_one[equalizer][1]:
                       equalizer += 1
                    
                except Exception as e:
                    equalizer=0
                
            else:
                big_one = Lane1
                lil_one = Lane2

            Lane_middle = []
            for i, ln1 in enumerate(lil_one):
                ln2 = big_one[equalizer+i]
                ln_middle = (ln1[0]+ln2[0])/2
                Lane_middle.append((ln_middle,ln2[1]))

            return Lane_middle
    
    def plot_lanes(self, Lane1, Lane2, Lane_middle,center_point=320, image_path=None):
        plt.figure(figsize=(10, 6))

        if image_path:
            img = mpimg.imread(image_path)
            plt.imshow(img, aspect='auto', extent=[0, 640, 0, 360], alpha=0.9)  # alpha sets transparency
        

        Lane1_x, Lane1_y = zip(*Lane1)
        Lane2_x, Lane2_y = zip(*Lane2)
        Lane_middle_x, Lane_middle_y = zip(*Lane_middle)

        plt.plot(Lane1_x, Lane1_y, label='Lane 1', color='b')
        plt.plot(Lane2_x, Lane2_y, label='Lane 2', color='r')
        plt.plot(Lane_middle_x, Lane_middle_y, label='Middle Lane', color='g', linestyle='--')
        plt.axvline(x=center_point, color='purple', linestyle='--', linewidth=2, label='X = Center')

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('')
        plt.legend()
        plt.gca().invert_yaxis()

        
        plt.show()
    
    def calculate_angle(self,Lane_middle):
        (x1, y1), (x2, y2) = Lane_middle[:2]  
    
        if x2 - x1 != 0:
            m_middle = (y2 - y1) / (x2 - x1)
        else:
            m_middle = float('inf')  
        angle_rad = np.arctan(m_middle)  
        angle_deg = np.degrees(angle_rad)
        return angle_deg
    

    def take_photo(self):
        url = 'http://192.168.112.134:8080/video'
        cam = cv2.VideoCapture(url)
        cv2.namedWindow("test")

        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            return "its bad"
    
        img_name = "datasets/tusimple-test/clips/1.jpg"
        cv2.imwrite(img_name, frame)
        
        #maybe these lines can be commented out.
        #cam.release()
        cv2.destroyAllWindows()
    
    def run(self):
        try:
            self.take_photo()
            raw = self.predict()
            center_lanes = self.get_center_lanes(raw,center_point=350)
            middle = self.middle_lane(center_lanes)
        
            angle = self.calculate_angle(middle)
            if angle > 0:
                print(angle, "Go Right")
            else:
                print(angle, "Go Left")
        except Exception as e:
            #print(e)
            pass

if __name__ == "__main__":
    while True:
        Prediction().run()
        

