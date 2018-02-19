import os
import logging
import logging.handlers
import ConfigParser
from optparse import OptionParser
import json
import sys

import numpy as np
import scipy.ndimage.measurements
import pylab as plt


def read_config(config_path):
    def as_dict(config):
        d = dict(config._sections)
        for k in d:
            d[k] = dict(config._defaults, **d[k])
            d[k].pop('__name__', None)
        return d
    
    config = ConfigParser.RawConfigParser()
    config.optionxform = str
    config.read(config_path)
    res = as_dict(config)
    return res

def read_params(config):
    logging.info('Input tomo_log: {}'.format(config))
    mask_file = config['mask_image']
    if not os.path.exists(mask_file):
        logging.error('Mask image not exists: {}'.format(mask_file))
        raise IOError('Mask image not exists: {}'.format(mask_file))
    else:
        logging.info('Mask image found: {}'.format(mask_file))
        
    data_file = config['data_image']
    if not os.path.exists(mask_file):
        logging.error('Data image not exists: {}'.format(data_file))
        raise IOError('Data image not exists: {}'.format(data_file))
    else:
        logging.info('Data image found: {}'.format(data_file))
    
    tomolog_file = config['tomo_log']
    if not os.path.exists(mask_file):
        log.error('Tomo log not exists: {}'.format(tomolog_file))
        raise IOError('Tomo log not exists: {}'.format(tomolog_file))
    else:
        logging.info('Tomo log found: {}'.format(tomolog_file))
                      
    zeros_mask = plt.imread(mask_file).astype('float32')
    if len(zeros_mask.shape) == 3:
        zeros_mask = zeros_mask[...,0]
    elif not len(zeros_mask.shape) == 2:
        logging.error('Wrong zeros mask dimensions number. Requied 2 or 3, given {}'.format(len(zeros_mask.shape)))
        raise ValueError('Wrong zeros mask dimensions number. Requied 2 or 3, given {}'.format(len(zeros_mask.shape))) 
    logging.info('Mask shape: {}'.format(zeros_mask.shape))
    
    data_image = plt.imread(data_file).astype('float32')
    if len(data_image.shape) == 3:
        data_image =data_image[...,0]
    elif not len(data_image.shape) == 2:
        logging.error('Wrong data image dimensions number. Requied 2 or 3, given {}'.format(len(zeros_mask.shape)))
        raise ValueError('Wrong data image dimensions number. Requied 2 or 3, given {}'.format(len(zeros_mask.shape))) 
    
    logging.info('Data shape: {}'.format(data_image.shape))
    
    config = read_config(tomolog_file)
    logging.info('Config: {}'.format(config))
    d_min = config['Reconstruction']['Minimum for CS to Image Conversion']
    d_min = float(d_min)
    d_max = config['Reconstruction']['Maximum for CS to Image Conversion']
    d_max = float(d_max)
    data = data_image /(data_image.max()-data_image.min())*(d_max-d_min)+d_min
    
    return data, zeros_mask, config

def calculate_background(data, zeros_mask):   
    labeled_mask, num_features = scipy.ndimage.measurements.label(zeros_mask)
    logging.info('Found regions: {}'.format(num_features-1))
    sigma = []
    for nf in range(num_features):
        if nf == 0 :
            continue
        
        data_constant = data[labeled_mask==nf]
        s = np.std(data_constant)
        sigma.append(s)
        
    logging.info('STD for regions: {}'.format(sigma))
    std = np.mean(sigma)
    logging.info('Mean STD for regions: {}'.format(std))
    mean_value = data.mean()
    logging.info('Mean reconstructed value for all data: {}'.format(mean_value))
    res = std/mean_value
    logging.info('Normalized STD: {}'.format(res))
    return  res

def main(config):
    data, zeros_mask, config = read_params(config)
    calculate_background(data, zeros_mask)

if __name__ == "__main__":
    
    LOG_FILENAME = 'astra_rec.out'

    my_logger = logging.getLogger('')
    my_logger.setLevel(logging.DEBUG)
    handler = logging.handlers.RotatingFileHandler(
        LOG_FILENAME,  maxBytes=1e5, backupCount=5)
    formatter = logging.Formatter('%(asctime)-15s %(levelname)-8s %(message)s')
    handler.setFormatter(formatter)

    my_logger.addHandler(handler)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    my_logger.addHandler(console)

    
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="config", help="JSON config file")
    (options, args) = parser.parse_args()
    if options.config is None:
        print 'Please define config file. Use: python background_artifacts.py -h'
        sys.exit(-1)
    
    logging.info('Loading config file: {}'.format(options.config))
    
    with open(options.config) as f:
        config = json.load(f)
    
    logging.info('Loaded config: {}'.format(config))
    main(config)
        