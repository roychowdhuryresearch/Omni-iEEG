import numpy as np
import mne
import re

def ends_with_number(channel_name):
    # Regular expression to check if the channel name ends with -{number}
    return bool(re.search(r'-\d+$', channel_name))

def read_raw(raw_path, resample=2000, drop_duplicates=False):
    raw = mne.io.read_raw_edf(raw_path, verbose= False, preload=True)
    # filter 60 Hz noise
    raw = raw.notch_filter(60, n_jobs=-1, notch_widths=2, verbose=False)
    raw_channels = raw.info['ch_names']
    data, channels = [], []
    if resample != None and resample != raw.info['sfreq']:
        raw = raw.copy().resample(resample, n_jobs=-1,verbose=False)

    for raw_ch in raw_channels:
        # if there is -{number} except -0, we skip it
        # check if -{number} is in the raw_ch
        # if ends_with_number(raw_ch) and "-0" not in raw_ch:
        #     continue
        ch_data = raw.get_data(raw_ch) * 1E6
        if drop_duplicates and "-0" in raw_ch:
            raw_ch = raw_ch.replace("-0", "")
        data.append(ch_data)
        channels.append(raw_ch)
    
    data = np.squeeze(np.array(data))
    # if resample != raw.info['sfreq']:
    #     # resample the data to the resample rate # resample = 1000 Hz, raw.info['sfreq'] = 2000 Hz
    #     data = mne.filter.resample(data, down=raw.info["sfreq"]/resample, npad="auto", n_jobs=-1)
    return data, channels

def concate_edf(fns, resample = 1000):
    data, channel = [], []
    for fn in fns:
        d, c = read_raw(fn, resample = resample)
        data.append(d), channel.append(c)
    data = np.concatenate(data)
    channel = np.concatenate(channel)
    return data, channel