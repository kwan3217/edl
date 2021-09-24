"""
Draw a heads-up-display
"""
from edl import Trajectory
from picturebox import PictureBox
import matplotlib

class HUD:
    def __init__(self,trajectory:Trajectory,oufn_pat:str):
        self.trajectory=trajectory
        self.oufn_pat=oufn_pat
    def draw_frame(self,i,et,pb:PictureBox):
        raise NotImplementedError
    def write_frames(self, *, et0:float, et1:float, fps: float, step0: int = 0, step1: int = None):
        print("Interpolating trajectory...")
        self.trajectory.tabulate()
        print("Writing HUD frames...")
        if step1 is None:
            step1 = int((et1 - et0) * fps)
        n_steps = step1 - step0 + 1
        matplotlib.use('agg')
        pb=PictureBox(1920,1080)
        for i_step in range(step0, step1):
            pb.clear()
            et=et0+i_step/fps
            self.draw_frame(i_step,et,pb)
            oufn=self.oufn_pat%i_step
            pb.savepng(oufn,transparent=True)
            print('.',end='')
            if i_step%100==0:
                print(f"{i_step}/{n_steps}")


