import numpy as np
import matplotlib.pyplot as plt
#from scipy.misc import imresize, imread
from cv2 import imread,resize
from human_pose_nn import HumanPoseIRNetwork
from scipy.spatial.distance import euclidean

class Gait1D:
    def __init__(self,model_path='/mnt/sda2/Spider/Kin-Verification/gait_Recognition/models/MPII+LSP.ckpt'):
        self.path=model_path
        self.joint_names = [
                            'right ankle ',
                            'right knee ',
                            'right hip',
                            'left hip',
                            'left knee',
                            'left ankle',
                            'pelvis',
                            'thorax',
                            'upper neck',
                            'head top',
                            'right wrist',
                            'right elbow',
                            'right shoulder',
                            'left shoulder',
                            'left elbow',
                            'left wrist']
        self.key_poses=16
        self.model_setup()
    def model_setup(self):
        self.model=HumanPoseIRNetwork()
        self.model.restore(self.path)
        pass

    def estimate_pose(self,frame):
        frame=resize(frame,(299,299))
        frame=np.expand_dims(frame,0)
        Y,X,Prob=self.model.estimate_joints(frame)
        Y,X,Prob=np.squeeze(Y),np.squeeze(X),np.squeeze(Prob)
        for i in range(16):
            print('%s: %.02f%%' % (joint_names[i], a[i] * 100))
        return (Y,X,Prob)

    def Extract_gaitfeature(self,Positions):
        Y,X=Positions
        feature=[]
        pelvis=[X[6],Y[6]]
        for i in range(self.key_poses):
            if i==6:
                continue
            else:
                feature.append(euclidean([X[i],Y[i]],pelvis))
        feature=np.array(feature)
        return feature


    def plot_poses(self,Positions):
        colors = ['r', 'r', 'b', 'm', 'm', 'y', 'g', 'g', 'b', 'c', 'r', 'r', 'b', 'm', 'm', 'c']
        for i in range(16):
            if i < 15 and i not in {5, 9}:
                plt.plot([x[i], x[i + 1]], [y[i], y[i + 1]], color = colors[i], linewidth = 5)
        plt.imshow(img)
        plt.show()
        plt.savefig('/mnt/sda2/Spider/Kin-Verification/dummy_pose.jpg')


if __name__ == "__main__":
    Gait=Gait1D()
    image=imread('/mnt/sda2/Spider/Kin-Verification/gait_Recognition/images/test5.jpg')
    prediction=Gait.estimate_pose(image)
    feature=Gait.Extract_gaitfeature((prediction[0],prediction[1]))
    print(feature)