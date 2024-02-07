"""
Utility functions for various audio features
"""

import librosa
import numpy as np

def get_madmom_onsets(filename, sr, hop_length):
    """
    Call madmom to get the onsets based on a dynamic bayes network on
    top of an audio novelty function from a recurrent neural network

    Parameters
    ----------
    filename: string
        Path to audio file
    sr: int
        Audio sample rate
    hop_length: int
        Frame hop length

    Returns
    -------
    beats: ndarray(nbeats, dtype=int)
        An array of beat onsets times, in factors of hopSize samples
    """
    from madmom.features.beats import RNNBeatProcessor, DBNBeatTrackingProcessor
    proc = DBNBeatTrackingProcessor(fps=100)
    act = RNNBeatProcessor()(filename)
    b = proc(act)
    beats = np.array(np.round(b*sr/hop_length), dtype=int)
    return beats


def normalize_feat(X):
    """
    Normalize features so that their norm is 1

    Parameters
    ----------
    X: ndarray(dim, n_frames)
        Features over time
    
    Returns
    -------
    ndarray(dim, n_frames)
        Normalized version of features
    """
    norm = np.sqrt(np.sum(X**2, axis=0))
    norm[norm == 0] = 1
    return X/norm[None, :]

def aggregate_intervals(X, fac, aggregate_fns=[np.mean]):
    """
    Aggregate intervals by a constant factor

    Parameters
    ----------
    X: ndarray(dim, N)
        Original features
    fac: int
        Factor by which to downsample
    aggregate_fns: List of F functions
        List of functions to use to aggregate the features in the intervals

    Returns
    -------
    ndarray(dim*F, ceil(N/fac))
    """
    dim = X.shape[0]
    Ys = []
    N = int(np.ceil(X.shape[1]/fac))
    for fn in aggregate_fns:
        Y = np.zeros((dim, N))
        for j in range(N):
            Y[:, j] = fn(X[:, j*fac:(j+1)*fac], axis=1)
        Ys.append(Y)
    return np.concatenate(Ys, axis=0)

def get_agg_features(y, sr, hop_length, avg_fac): 
    """
    Compute audio features aggregated over intervals

    Parameters
    ----------
    y: ndarray(n_samples)
        Audio samples
    sr: int
        Audio sample rate
    hop_length: int
        Hop length between frames
    avg_fac: int
        Factor by which to aggregate features

    Returns
    -------
    [
        chroma: ndarray(12, < n_samples//hop_length)
            Chroma features
        mfcc: ndarray(20, < n_samples//hop_length)
            MFCC features
        tempogram: ndarray(768, < n_samples//hop_length)
            Tempogram features
    ]
    """   
    # 1) CQT chroma with 3x oversampling in pitch
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    
    # 2) Exponentially liftered MFCCs (TODO: Check for loudness normalization)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop_length)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=20)
    lifterexp = 0.6
    coeffs = np.arange(mfcc.shape[0])**lifterexp
    coeffs[0] = 1
    mfcc = coeffs[:, None]*mfcc

    # 3) Tempograms
    #  Use a super-flux max smoothing of 5 frequency bands in the oenv calculation
    SUPERFLUX_SIZE = 5
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length,
                                        max_size=SUPERFLUX_SIZE)
    tempogram = librosa.feature.tempogram(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    
    ## Step 2: Aggregate features
    n_frames = np.min([chroma.shape[1], mfcc.shape[1], tempogram.shape[1]])
    chroma_agg = np.array(chroma[:, 0:n_frames])
    mfcc_agg = np.array(mfcc[:, 0:n_frames])
    tempogram_agg = np.array(tempogram[:, 0:n_frames])
    chroma_agg = aggregate_intervals(chroma_agg, avg_fac, [np.median])
    mfcc_agg = aggregate_intervals(mfcc_agg, avg_fac, [np.mean])
    tempogram_agg = aggregate_intervals(tempogram_agg, avg_fac, [np.mean])
    
    return [normalize_feat(chroma_agg), normalize_feat(mfcc_agg), normalize_feat(tempogram_agg)]


def chrompwr(X, P=.5):
    """
    Y = chrompwr(X,P)  raise chroma columns to a power, preserving norm
    2006-07-12 dpwe@ee.columbia.edu
    -> python: TBM, 2011-11-05, TESTED
    """
    nchr, nbts = X.shape
    # norms of each input col
    CMn = np.tile(np.sqrt(np.sum(X * X, axis=0)), (nchr, 1))
    CMn[CMn == 0] = 1
    # normalize each input col, raise to power
    CMp = np.power(X/CMn, P)
    # norms of each resulant column
    CMpn = np.tile(np.sqrt(np.sum(CMp * CMp, axis=0)), (nchr, 1))
    CMpn[np.where(CMpn==0)] = 1.
    # rescale cols so norm of output cols match norms of input cols
    return CMn * (CMp / CMpn)

def get_oti(C1, C2, do_plot = False):
    """
    Get the optimial transposition of the first chroma vector
    with respect to the second one
    Parameters
    ----------
    C1: ndarray(n_chroma_bins)
        Chroma vector 1
    C2: ndarray(n_chroma_bins)
        Chroma vector 2
    do_plot: boolean
        Plot the agreements over all shifts
    Returns
    -------
    oit: int
        An index by which to rotate the first chroma vector
        to match with the second
    """
    NChroma = len(C1)
    shiftScores = np.zeros(NChroma)
    for i in range(NChroma):
        shiftScores[i] = np.sum(np.roll(C1, i)*C2)
    if do_plot:
        import matplotlib.pyplot as plt
        plt.plot(shiftScores)
        plt.title("OTI")
        plt.show()
    return np.argmax(shiftScores)


def get_hpcp(x, hop_length, sr, 
        frameSize=4096,
        windowType='blackmanharris62',
        harmonicsPerPeak=8,
        magnitudeThreshold=0,
        maxPeaks=100,
        whitening=True,
        referenceFrequency=440,
        minFrequency=100,
        maxFrequency=3500,
        nonLinear=False,
        numBins=12):
    """
    HPCP code, copied from datacos with minor modifications
    https://github.com/ctralie/acoss/blob/b8cf1346d9bb2e3ff22493e9c54242665c050359/preprocess/features.py#L209
    Requires essentia (pip install essentia)
    
    Compute Harmonic Pitch Class Profiles (HPCP) for the input audio files using essentia standard mode using
    the default parameters as mentioned in [1].
    Please refer to the following paper for detailed explanantion of the algorithm.
    [1]. Gómez, E. (2006). Tonal Description of Polyphonic Audio for Music Content Processing.
    For full list of parameters of essentia standard mode HPCP 
    please refer to http://essentia.upf.edu/documentation/reference/std_HPCP.html
    
    Parameters
    ----------
    x: ndarray(n_samples)
        Audio samples
    hop_length: int
        Hop length between frames
    sr: int
        Sample rate
    
    Returns
    hpcp: ndarray(12, n_frames)
        The HPCP coefficients at each time frame
    """
    from essentia import Pool, array
    import essentia.standard as estd

    audio = array(x)
    frameGenerator = estd.FrameGenerator(audio, frameSize=frameSize, hopSize=hop_length)
    # framecutter = estd.FrameCutter(frameSize=frameSize, hopSize=self.hop_length)
    windowing = estd.Windowing(type=windowType)
    spectrum = estd.Spectrum()
    # Refer http://essentia.upf.edu/documentation/reference/streaming_SpectralPeaks.html
    spectralPeaks = estd.SpectralPeaks(magnitudeThreshold=magnitudeThreshold,
                                        maxFrequency=maxFrequency,
                                        minFrequency=minFrequency,
                                        maxPeaks=maxPeaks,
                                        orderBy="frequency",
                                        sampleRate=sr)
    # http://essentia.upf.edu/documentation/reference/streaming_SpectralWhitening.html
    spectralWhitening = estd.SpectralWhitening(maxFrequency= maxFrequency,
                                                sampleRate=sr)
    # http://essentia.upf.edu/documentation/reference/streaming_HPCP.html
    hpcp = estd.HPCP(sampleRate=sr,
                    maxFrequency=maxFrequency,
                    minFrequency=minFrequency,
                    referenceFrequency=referenceFrequency,
                    nonLinear=nonLinear,
                    harmonics=harmonicsPerPeak,
                    size=numBins)
    pool = Pool()

    #compute hpcp for each frame and add the results to the pool
    for frame in frameGenerator:
        spectrum_mag = spectrum(windowing(frame))
        frequencies, magnitudes = spectralPeaks(spectrum_mag)
        if whitening:
            w_magnitudes = spectralWhitening(spectrum_mag,
                                            frequencies,
                                            magnitudes)
            hpcp_vector = hpcp(frequencies, w_magnitudes)
        else:
            hpcp_vector = hpcp(frequencies, magnitudes)
        pool.add('tonal.hpcp',hpcp_vector)

    return pool['tonal.hpcp'].T



def get_beat_synchronous_crema(y, sr):
    """
    Compute CREMA features and synchronize them to beats
    NOTE: This is the best feature for shingles
    
    NOTE ALSO: I recommend creating a conda envrionment with python 3.6
    and 
    pip install crema
    in that environment
    
    https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html
    
    conda create -n mir python=3.6
    conda activate mir
    pip install jupyter
    pip install crema
    pip install librosa
    jupyter notebook
    

    Parameters
    ----------
    y: ndarray(N)
        Audio samples
    sr: int
        Audio sample rate
    
    Returns
    -------
    ndarray(12, n_beats)
        Beat-synchronous median aggregated CREMA frames
    """
    from crema.models.chord import ChordModel
    from librosa.util import sync
    from librosa.onset import onset_strength
    from librosa.beat import beat_track
    model = ChordModel()
    C = model.outputs(y=y, sr=sr)['chord_pitch'].T

    SUPERFLUX_SIZE = 5
    hop_length = 512
    oenv = onset_strength(y=y, sr=sr, hop_length=hop_length, max_size=SUPERFLUX_SIZE)
    _, beats = beat_track(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    beats = np.array(np.round(beats/4), dtype=int)
    beats[beats >= oenv.size] = oenv.size-1

    return sync(C, beats, aggregate=np.median)

def get_beat_synchronous_hpcp(y, sr, hop_length):
    """
    Compute HPCP features and synchronize them to beats

    Parameters
    ----------
    y: ndarray(N)
        Audio samples
    sr: int
        Audio sample rate
    hop_length: int
        Hop length to use for initial HPCP features and beat tracking
    
    Returns
    -------
    ndarray(12, n_beats)
        Beat-synchronous median aggregated HPCP frames
    """
    from librosa.util import sync
    from librosa.onset import onset_strength
    from librosa.beat import beat_track
    C = get_hpcp(y, hop_length, sr)
    
    SUPERFLUX_SIZE = 5
    oenv = onset_strength(y=y, sr=sr, hop_length=hop_length, max_size=SUPERFLUX_SIZE)
    _, beats = beat_track(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    beats = np.array(np.round(beats), dtype=int)
    beats[beats >= oenv.size] = oenv.size-1

    return sync(C, beats, aggregate=np.median)


def get_beat_synchronous_chroma_cqt(y, sr, hop_length):
    """
    Compute librosa chroma-CQT features and synchronize them to beats

    Parameters
    ----------
    y: ndarray(N)
        Audio samples
    sr: int
        Audio sample rate
    hop_length: int
        Hop length to use for initial HPCP features and beat tracking
    
    Returns
    -------
    ndarray(12, n_beats)
        Beat-synchronous median aggregated HPCP frames
    """
    from librosa.util import sync
    from librosa.onset import onset_strength
    from librosa.beat import beat_track
    from librosa.feature import chroma_cqt
    C = chroma_cqt(y=y, sr=sr, hop_length=hop_length, bins_per_octave=12*3)
    
    SUPERFLUX_SIZE = 5
    oenv = onset_strength(y=y, sr=sr, hop_length=hop_length, max_size=SUPERFLUX_SIZE)
    _, beats = beat_track(onset_envelope=oenv, sr=sr, hop_length=hop_length)
    beats = np.array(np.round(beats), dtype=int)
    beats[beats >= oenv.size] = oenv.size-1

    return sync(C, beats, aggregate=np.median)