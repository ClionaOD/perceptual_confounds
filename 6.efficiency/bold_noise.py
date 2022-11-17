import numpy as np
from matplotlib import pyplot as plt

class bold_noise:
    noisetype=None
    nsamples=None
    nscans=None
    tr=None

    # first noise parameter is constant across frequency, second scales with period
    NOISE_PARMS={'adult': [4.86, 0.1636],
                }

    def __init__(self, noisetype='adult', nsamples=1, nscans=256, tr=2):
        self.noisetype = noisetype
        self.nscans = int(nscans)
        self.tr = tr
        self.scaling = 200
        if not self.nscans%2==0:
            raise('nscans must be a multiple of two') 

        
    def generate(self, nsamples=1):
        nyquist = 0.5 / self.tr
        freq = np.linspace(0, nyquist, int(self.nscans/2)+1).reshape(int(self.nscans/2)+1,1)
        freq[0] = np.nan
        period = 1.0 / freq
        noise_power = self.NOISE_PARMS[self.noisetype][0] + period * self.NOISE_PARMS[self.noisetype][1]
        noise_power = self.scaling * noise_power * np.exp(1j * np.random.uniform(low=0.0, high=np.pi*2, size=(noise_power.shape[0], nsamples)))
        noise_mirror = np.concatenate( (np.zeros((1,nsamples)), noise_power[1:-1,:], np.flip(noise_power[1:,:], axis=0).conjugate()))
        self.noise_power = noise_power
        self.sample = np.abs(np.fft.ifft(noise_mirror, axis=0))
        self.freq = freq

        return self.sample

    def show(self):
        fig, ax = plt.subplots( ncols=2)
        x=np.tile(self.freq,(1,self.noise_power.shape[1]))
        ax[0].plot(x, self.noise_power)
        ax[0].set_xlabel('Frequency (Hz)')
        ax[0].set_ylabel('Spectrum level')
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        ax[1].plot(self.sample)
        ax[1].set_xlabel('Time (s)')
        

if __name__=='__main__':
    bn=bold_noise()
    sample = bn.generate(10)
    bn.show()
    plt.savefig('bold_power.jpg')