import numpy as np
import math
import tensorflow.compat.v1 as tf

class CosineAnnealer:

    def __init__(self, start, end, steps):
        self.start = start
        self.end = end
        self.steps = steps
        self.n = 0

    def step(self):
        self.n += 1
        cos = tf.math.cos(np.pi * (self.n / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos

    def getval(self,step):
        cos = tf.math.cos(np.pi * (step / self.steps)) + 1
        return self.end + (self.start - self.end) / 2. * cos

class OneCycleScheduler():
    """
    From https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/, by Andrich van Wyk modified from
    fastai lib. Modified again to apply for TPU code.
    """

    def __init__(self, lr_max, steps, mom_min=0.85, mom_max=0.95, phase_1_pct=0.3, div_factor=25.):
        super(OneCycleScheduler, self).__init__()
        lr_min = lr_max / div_factor
        final_lr = lr_max / (div_factor * 1e4)
        phase_1_steps = steps * phase_1_pct
        phase_2_steps = steps - phase_1_steps

        self.phase_1_steps = phase_1_steps
        self.phase_2_steps = phase_2_steps
        self.phase = 0
        self.step = 0

        self.phases = [[CosineAnnealer(lr_min, lr_max, phase_1_steps), CosineAnnealer(mom_max, mom_min, phase_1_steps)],
                       [CosineAnnealer(lr_max, final_lr, phase_2_steps),
                        CosineAnnealer(mom_min, mom_max, phase_2_steps)]]

        self.lrs = []
        self.moms = []

    @tf.function
    def getlrmom(self,step):
        if step<self.phase_1_steps:
        # return tf.where([step<self.phase_1_steps,step<self.phase_1_steps],
            # tf.constant(self.phases[0][0].getval(step),dtype=tf.float32),tf.constant(self.phases[0][1].getval(step),dtype=tf.float32),
            # tf.constant(self.phases[1][0].getval(step-self.phase_1_steps),dtype=tf.float32),tf.constant(self.phases[1][1].getval(step-self.phase_1_steps),dtype=tf.float32)
            return [self.phases[0][0].getval(step),self.phases[0][1].getval(step)]
        else:
            return [self.phases[1][0].getval(step-self.phase_1_steps),self.phases[1][1].getval(step-self.phase_1_steps)]


def lrs(step,total_step):
    low = math.log2(1e-3)
    high = math.log2(50)
    return 2**(low+(high-low)*step/total_step)