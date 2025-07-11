import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class EyeParameter:
    def __init__(self,
                eye_size=(0.5, 0.5),
                eye_pos=(0.5, 0.5),
                cheek_size=(1.0, 0.2),
                cheek_pos=(0.5, 1.0),
                eyebrow_size=(1.0, 0.2),
                eyebrow_pos=(0.5, 0.0)):
        self.eye_size = eye_size
        self.eye_pos = eye_pos
        self.cheek_size = cheek_size
        self.cheek_pos = cheek_pos
        self.eyebrow_size = eyebrow_size
        self.eyebrow_pos = eyebrow_pos

    def draw(self, img, w, h, is_left = True, eye_color = (255,255,255), skin_color = (0,0,0)):
        w_eff = w/2.0
        x_offset = 0
        if is_left: 
            x_offset = 0

            eye_pos = (
                int(self.eye_pos[0]*w_eff + x_offset),
                int(self.eye_pos[1]*h)
            )
            eye_axeslength = (
                int(self.eye_size[0]*0.5*w_eff),
                int(self.eye_size[1]*0.5*h)
            )

            cheek_pos = (
                int(self.cheek_pos[0]*w_eff + x_offset),
                int(self.cheek_pos[1]*h)
            )
            cheek_axeslength = (
                int(self.cheek_size[0]*0.5*w_eff),
                int(self.cheek_size[1]*0.5*h)
            )

            eyebrow_pos = (
                int(self.eyebrow_pos[0]*w_eff + x_offset),
                int(self.eyebrow_pos[1]*h)
            )
            eyebrow_axeslength = (
                int(self.eyebrow_size[0]*0.5*w_eff),
                int(self.eyebrow_size[1]*0.5*h)
            )

        else: 
            x_offset = w_eff

            eye_pos = (
                int((1.0-self.eye_pos[0])*w_eff + x_offset),
                int(self.eye_pos[1]*h)
            )
            eye_axeslength = (
                int(self.eye_size[0]*0.5*w_eff),
                int(self.eye_size[1]*0.5*h)
            )

            cheek_pos = (
                int((1.0-self.cheek_pos[0])*w_eff + x_offset),
                int(self.cheek_pos[1]*h)
            )
            cheek_axeslength = (
                int(self.cheek_size[0]*0.5*w_eff),
                int(self.cheek_size[1]*0.5*h)
            )

            eyebrow_pos = (
                int((1.0-self.eyebrow_pos[0])*w_eff + x_offset),
                int(self.eyebrow_pos[1]*h)
            )
            eyebrow_axeslength = (
                int(self.eyebrow_size[0]*0.5*w_eff),
                int(self.eyebrow_size[1]*0.5*h)
            )

        cv2.ellipse(img, eye_pos, eye_axeslength, 0, 0, 360, eye_color, -1)
        cv2.ellipse(img, cheek_pos, cheek_axeslength, 0, 0, 360, skin_color, -1)
        cv2.ellipse(img, eyebrow_pos, eyebrow_axeslength, 0, 0, 360, skin_color, -1)

def fuse_EyeParameters(parameter_a:EyeParameter, parameter_b:EyeParameter, alpha:float):
    out = EyeParameter()
    out.eye_size = (
        parameter_a.eye_size[0]*(1-alpha) + parameter_b.eye_size[0]*alpha,
        parameter_a.eye_size[1]*(1-alpha) + parameter_b.eye_size[1]*alpha
    )
    out.eye_pos = (
        parameter_a.eye_pos[0]*(1-alpha) + parameter_b.eye_pos[0]*alpha,
        parameter_a.eye_pos[1]*(1-alpha) + parameter_b.eye_pos[1]*alpha
    )
    out.cheek_size = (
        parameter_a.cheek_size[0]*(1-alpha) + parameter_b.cheek_size[0]*alpha,
        parameter_a.cheek_size[1]*(1-alpha) + parameter_b.cheek_size[1]*alpha
    )
    out.cheek_pos = (
        parameter_a.cheek_pos[0]*(1-alpha) + parameter_b.cheek_pos[0]*alpha,
        parameter_a.cheek_pos[1]*(1-alpha) + parameter_b.cheek_pos[1]*alpha
    )
    out.eyebrow_size = (
        parameter_a.eyebrow_size[0]*(1-alpha) + parameter_b.eyebrow_size[0]*alpha,
        parameter_a.eyebrow_size[1]*(1-alpha) + parameter_b.eyebrow_size[1]*alpha
    )
    out.eyebrow_pos = (
        parameter_a.eyebrow_pos[0]*(1-alpha) + parameter_b.eyebrow_pos[0]*alpha,
        parameter_a.eyebrow_pos[1]*(1-alpha) + parameter_b.eyebrow_pos[1]*alpha
    )  
    return out

class Eyes:
    def __init__(self):
        self.img = None
        self.x = 600
        self.y = 1024

        self.skin_color = (0,0,0)
        self.eye_color = (255,255,255)

        self.eye_x = [int(self.x*0.25), int(self.x*0.75)]
        self.eye_y = [int(self.y*0.5), int(self.y*0.5)]
        self.eye_size_x = [int(self.x*0.3), int(self.x*0.3)]
        self.eye_size_y = [int(self.y*0.4), int(self.y*0.4)]

        self.emotions = {
            'normal'    : (EyeParameter(), EyeParameter()),
            'curious'   : (EyeParameter(eye_size = (0.5, 0.7)), EyeParameter(eye_size = (0.5, 0.7))),
            'happy'     : (EyeParameter(cheek_pos=(0.5,0.7)), EyeParameter(cheek_pos=(0.5,0.7))),
            'angry'     : (EyeParameter(eyebrow_pos=(0.7, 0.3)), EyeParameter(eyebrow_pos=(0.7, 0.3))),
            'sad'       : (EyeParameter(eyebrow_pos=(0.3, 0.3)), EyeParameter(eyebrow_pos=(0.3, 0.3))),
            'sleep'    : (EyeParameter(eye_size = (0.5, 0.1), eyebrow_pos=(0.5, 0.43)), EyeParameter(eye_size = (0.5, 0.1), eyebrow_pos=(0.5, 0.43))),
            'happy2' : (EyeParameter(eye_size = (0.5, 0.1), cheek_pos=(0.5,0.57)), EyeParameter(eye_size = (0.5, 0.1), cheek_pos=(0.5,0.57))),
            'worried' : (EyeParameter( cheek_pos=(0.5,0.8), eyebrow_pos=(0.5, 0.2)), EyeParameter(cheek_pos=(0.5,0.8), eyebrow_pos=(0.5, 0.2)))
        }

    def draw_eyes(self, left_eyes:EyeParameter, right_eye:EyeParameter):
        if self.img is None:
            self.img = np.zeros((self.y, self.x, 3), dtype = np.uint8)
        
        h, w, _ = self.img.shape
        if h != self.y or w != self.x:
            self.img = np.zeros((self.y, self.x, 3), dtype = np.uint8)

        self.img[:,:] = np.asarray(self.skin_color)
        right_eye.draw(self.img, h=self.y, w=self.x, is_left=False, eye_color=self.eye_color, skin_color=self.skin_color)
        left_eyes.draw(self.img, h=self.y, w=self.x, is_left=True, eye_color=self.eye_color, skin_color=self.skin_color)
    
    def set_emotion(self, emotion:str):
        l, r = self.emotions[emotion]
        self.draw_eyes(l, r)

    def set_emotion_mix(self, emotion_a:str, emotion_b:str, alpha:float):
        la, ra = self.emotions[emotion_a]
        lb, rb = self.emotions[emotion_b]
        l = fuse_EyeParameters(la, lb, alpha)
        r = fuse_EyeParameters(ra, rb, alpha)
        self.draw_eyes(l, r)

if __name__ == "__main__":

    eyes = Eyes()
    eyes.eye_color = (255, 255, 0)
    eyes.skin_color = (125, 125, 255)

    def get_img(t):

        alpha = (t-0.3) / 0.1
        if( alpha > 1):
            alpha = 1.0
        if( alpha < 0):
            alpha = 0

        eyes.set_emotion_mix('normal', 'worried', alpha)
        return eyes.img
    
    fig = plt.figure()
    im = plt.imshow(get_img(0))
    
    fps = 30.0
    duration = 5
    def animate_func(i):
        t = (i)/fps/duration
        im.set_array(get_img(t))
        return [im]

    
    anim = animation.FuncAnimation(
                                fig, 
                                animate_func, 
                                frames = int(duration * fps),
                                interval = 1000 / fps, # in ms
                                )
    writergif = animation.PillowWriter(fps=fps)
    anim.save('worried.gif', writer=writergif)