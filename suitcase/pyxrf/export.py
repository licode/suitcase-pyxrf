import os
import numpy as np
import h5py
import pandas as pd
import logging
logger = logging.getLogger()


def _make_hdf_srx(fpath, hdr, config_data,
                  num_det=3, create_each_det=False,
                  num_end_lines_excluded=None,
                  spectrum_len = 4096):
    """
    Save the data from databroker to hdf file for SRX beamline.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    hdr : header
        hdr = db[-1]
    config_data : dict
        dictionary to map general name like xrf_detector to name used
        in databroker, like xs_settings_ch1, xs_settings_ch2, these names
        might change due to configuration at beamline.
    num_det: int, optional
        number of fluorescence detectors
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines
    spectrum_len : int, optional
        standard spectrum length
    """

    start_doc = hdr['start']
    plan_n = start_doc.get('plan_name')
    if 'fly' not in plan_n: # not fly scan, step scan instead
        datashape = start_doc['shape']   # vertical first then horizontal
        fly_type = None

        snake_scan = start_doc.get('snaking')
        if snake_scan[1]:
            fly_type = 'pyramid'

        if hdr.stop is not None:
            data = hdr.table(fill=True, convert_times=False)
        else: # scan is not finished
            total_len = get_total_scan_point(hdr) - 2
            evs, _ = zip(*zip(hdr.events(hdr, fill=True), range(total_len)))
            namelist = (config_data['xrf_detector'] +
                        hdr.start.motors +config_data['scaler_list'])
            dictv = {v:[] for v in namelist}
            for e in evs:
                for k,v in dictv.items():
                    dictv[k].append(e.data[k])
            data = pd.DataFrame(dictv, index=np.arange(1, total_len+1)) # need to start with 1

        xrf_detector_names = config_data['xrf_detector']
        #express3 detector name changes in databroker
        if xrf_detector_names[0] not in data.keys():
            xrf_detector_names = ['xs_channel'+str(i) for i in range(1,4)]
        logger.info('Saving data to hdf file.')
        data_new = map_data2D(data, datashape,
                              det_list=xrf_detector_names,
                              pos_list=hdr.start.motors,
                              scaler_list=config_data['scaler_list'],
                              fly_type=fly_type, spectrum_len=spectrum_len)
        write_db_to_hdf(fpath, data_new, num_det=num_det,
                        create_each_det=create_each_det)
        if 'xs2' in hdr.start.detectors: # second dector
            logger.info('Saving data to hdf file for second xspress3 detector.')
            tmp = fpath.split('.')
            fpath1 = '.'.join([tmp[0]+'_1', tmp[1]])
            data_new = map_data2D(data, datashape,
                                  det_list=config_data['xrf_detector2'],
                                  pos_list=hdr.start.motors,
                                  scaler_list=config_data['scaler_list'],
                                  fly_type=fly_type, spectrum_len=spectrum_len)
            write_db_to_hdf(fpath1, data_new, num_det=num_det,
                            create_each_det=create_each_det)
    else:
        # srx fly scan
        # Added by AMK to allow flying of single element on xs2
        if 'E_tomo' in start_doc['scaninfo']['type']:
            num_det = 1

        scaler_list = ['i0', 'time']
        xpos_name = 'enc1'
        ypos_name = 'hf_stage_y'
        vertical_fast = False  # assuming fast on x as default
        if num_end_lines_excluded is None:
            datashape = [start_doc['shape'][1], start_doc['shape'][0]]   # vertical first then horizontal, assuming fast scan on x
        else:
            datashape = [start_doc['shape'][1]-num_end_lines_excluded, start_doc['shape'][0]]
        if 'fast_axis' in hdr.start['scaninfo']:
            if hdr.start['scaninfo']['fast_axis'] == 'VER':  # fast scan along vertical, y is fast scan, x is slow
                xpos_name = 'enc1'
                ypos_name = 'hf_stage_x'
                vertical_fast = True
                #datashape = [start_doc['shape'][0], start_doc['shape'][1]]   # fast vertical scan put shape[0] as vertical direction

        new_shape = datashape + [spectrum_len]
        total_points = datashape[0]*datashape[1]

        new_data = {}
        data = {}
        e = hdr.events(fill=True, stream_name='stream0')

        new_data['scaler_names'] = scaler_list
        scaler_tmp = np.zeros([datashape[0], datashape[1], len(scaler_list)])
        if vertical_fast:  # data shape only has impact on scalar data
            scaler_tmp = np.zeros([datashape[1], datashape[0], len(scaler_list)])
        for v in scaler_list+[xpos_name]:
            data[v] = np.zeros([datashape[0], datashape[1]])

        if create_each_det is False:
            new_data['det_sum'] = np.zeros(new_shape)
            print('det sum')
        else:
            for i in range(num_det):
                new_data['det'+str(i+1)] = np.zeros(new_shape)

        for m,v in enumerate(e):
            if m < datashape[0]:
                for n in scaler_list+[xpos_name]:
                    min_len = min(v.data[n].size, datashape[1])
                    data[n][m, :min_len] = v.data[n][:min_len]
                    if min_len < datashape[1]:  # position data or i0 has shorter length than fluor data
                        len_diff = datashape[1] - min_len
                        interp_list = (v.data[n][-1]-v.data[n][-3])/2*np.arange(1,len_diff+1) + v.data[n][-1]
                        data[n][m, min_len:datashape[1]] = interp_list
                if create_each_det is False:
                    for i in range(num_det):
                        new_data['det_sum'][m,:v.data['fluor'].shape[0],:] += v.data['fluor'][:,i,:]
                else:
                    for i in range(num_det):  # in case the data length in each line is different
                        new_data['det'+str(i+1)][m,:v.data['fluor'].shape[0],:] = v.data['fluor'][:,i,:]

        if vertical_fast: # need to transpose the data, as we scan y first
            if create_each_det is False:
                new_data['det_sum'] = np.transpose(new_data['det_sum'], axes=(1,0,2))
            else:
                for i in range(num_det):
                    new_data['det'+str(i+1)] = np.transpose(new_data['det'+str(i+1)], axes=(1,0,2))

        if vertical_fast is False:
            for i,v in enumerate(scaler_list):
                scaler_tmp[:, :, i] = data[v]
        else:
            for i,v in enumerate(scaler_list):
                scaler_tmp[:, :, i] = data[v].T
        new_data['scaler_data'] = scaler_tmp
        x_pos = data[xpos_name]

        # get y position data
        data1 = hdr.table(fill=True, stream_name='primary')
        if num_end_lines_excluded is not None:
            data1 = data1[:datashape[0]]
        if ypos_name not in data1.keys():
            ypos_name = 'hf_stage_z'        #vertical along z
        y_pos0 = np.hstack(data1[ypos_name])
        if len(y_pos0) >= x_pos.shape[0]:  # y position is more than actual x pos, scan not finished?
            y_pos = y_pos0[:x_pos.shape[0]]
            x_tmp = np.ones(x_pos.shape[1])
            xv, yv = np.meshgrid(x_tmp, y_pos)
            # need to change shape to sth like [2, 100, 100]
            data_tmp = np.zeros([2, x_pos.shape[0], x_pos.shape[1]])
            data_tmp[0,:,:] = x_pos
            data_tmp[1,:,:] = yv
            new_data['pos_data'] = data_tmp
            new_data['pos_names'] = ['x_pos', 'y_pos']
            if vertical_fast: # need to transpose the data, as we scan y first
                data_tmp = np.zeros([2, x_pos.shape[1], x_pos.shape[0]]) # fast scan on y has impact for scalar data
                data_tmp[1,:,:] = x_pos.T
                data_tmp[0,:,:] = yv.T
                new_data['pos_data'] = data_tmp
        else:
            logger.warning('x,y positions are not saved.')
        # output to file
        logger.info('Saving data to hdf file.')
        write_db_to_hdf(fpath, new_data, num_det=num_det,
                        create_each_det=create_each_det)


def map_data2D(data, datashape, det_list, pos_list, scaler_list,
               fly_type=None, subscan_dims=None, spectrum_len=4096):
    """
    Data is obained from databroker. Transfer items from data to a dictionay of
    numpy array, which has 2D shape same as scanning area.

    This function can handle stopped/aborted scans. Raster scan (snake scan) is
    also considered.

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        data from data broker
    datashape : tuple or list
        shape of two D image
    det_list : list, tuple
        list of detector channels
    pos_list : list, tuple
        list of pos pv
    scaler_list : list, tuple
        list of scaler pv
    fly_type : string or optional
        raster scan (snake scan) or normal
    subscan_dims : 1D array or optional
        used at HXN, 2D of a large area is split into small area scans
    spectrum_len : int, optional
        standard spectrum length

    Returns
    -------
    dict of numpy array
    """
    data_output = {}
    sum_data = None
    new_v_shape = datashape[0]  # updated if scan is not completed

    for n in range(len(det_list)):
        c_name = det_list[n]
        if c_name in data:
            detname = 'det'+str(n+1)
            logger.info('read data from %s' % c_name)
            channel_data = data[c_name]

            # new veritcal shape is defined to ignore zeros points caused by stopped/aborted scans
            new_v_shape = len(channel_data) // datashape[1]
            new_data = np.vstack(channel_data)
            new_data = new_data[:new_v_shape*datashape[1], :]
            new_data = new_data.reshape([new_v_shape, datashape[1],
                                         len(channel_data[1])])
            if new_data.shape[2] != spectrum_len:
                # merlin detector has spectrum len 2048
                # make all the spectrum len to 4096, to avoid unpredicted error in fitting part
                new_tmp = np.zeros([new_data.shape[0], new_data.shape[1], spectrum_len])
                new_tmp[:,:,:new_data.shape[2]] = new_data
                new_data = new_tmp
            if fly_type in ('pyramid',):
                new_data = flip_data(new_data, subscan_dims=subscan_dims)
            data_output[detname] = new_data
            if sum_data is None:
                sum_data = new_data
            else:
                sum_data += new_data

    data_output['det_sum'] = sum_data

    # scanning position data
    pos_names, pos_data = get_name_value_from_db(pos_list, data,
                                                 datashape)
    for i in range(len(pos_names)):
        if 'x' in pos_names[i]:
            pos_names[i] = 'x_pos'
        elif 'y' in pos_names[i]:
            pos_names[i] = 'y_pos'
    if fly_type in ('pyramid',):
        for i in range(pos_data.shape[2]):
            # flip position the same as data flip on det counts
            pos_data[:, :, i] = flip_data(pos_data[:, :, i], subscan_dims=subscan_dims)
    for i, v in enumerate(pos_names):
        data_output[v] = pos_data[:, :, i]

    # scaler data
    scaler_names, scaler_data = get_name_value_from_db(scaler_list, data,
                                                       datashape)
    if fly_type in ('pyramid',):
        scaler_data = flip_data(scaler_data, subscan_dims=subscan_dims)
    for i, v in enumerate(scaler_names):
        data_output[v] = scaler_data[:, :, i]
    return data_output


def write_db_to_hdf(fpath, data, num_det=3, create_each_det=True):
    """
    Data is a dictionary of numpy array. Save the data to hdf file.

    Parameters
    ----------
    fpath: str
        path to save hdf file
    data : dict
        fluorescence data with scaler value and positions
    num_det : int
        number of detector
    create_each_det : Bool, optional
        if number of point is too large, only sum data is saved in h5 file
    """
    interpath = 'xrfmap'
    sum_data = None

    with h5py.File(fpath, 'a') as f:
        if create_each_det:
            for n in range(num_det):
                detname = 'det' + str(n+1)
                new_data = data[detname]

                if sum_data is None:
                    sum_data = new_data
                else:
                    sum_data += new_data

                dataGrp = f.create_group(interpath+'/'+detname)
                ds_data = dataGrp.create_dataset('counts', data=new_data, compression='gzip')
                ds_data.attrs['comments'] = 'Experimental data from channel ' + str(n)
        else:
            sum_data = data['det_sum']

        # summed data
        if sum_data is not None:
            dataGrp = f.create_group(interpath+'/detsum')
            ds_data = dataGrp.create_dataset('counts', data=sum_data, compression='gzip')
            ds_data.attrs['comments'] = 'Experimental data from channel sum'

        # add positions
        if 'pos_names' in data:
            dataGrp = f.create_group(interpath+'/positions')
            pos_names = data['pos_names']
            pos_data = data['pos_data']
            dataGrp.create_dataset('name', data=helper_encode_list(pos_names))
            dataGrp.create_dataset('pos', data=pos_data)

        # scaler data
        if 'scaler_data' in data:
            dataGrp = f.create_group(interpath+'/scalers')
            scaler_names = data['scaler_names']
            scaler_data = data['scaler_data']
            dataGrp.create_dataset('name', data=helper_encode_list(scaler_names))
            dataGrp.create_dataset('val', data=scaler_data)


def helper_encode_list(data, data_type='utf-8'):
    return [d.encode(data_type) for d in data]


def get_name_value_from_db(name_list, data, datashape):
    """
    Get data from db, and parse it to the required format and output.
    """
    pos_names = []
    pos_data = np.zeros([datashape[0], datashape[1], len(name_list)])
    for i, v in enumerate(name_list):
        posv = np.zeros(datashape[0]*datashape[1])  # keep shape unchanged, so stopped/aborted run can be handled.
        data[v] = np.asarray(data[v])  # in case data might be list
        posv[:data[v].shape[0]] = np.asarray(data[v])
        pos_data[:, :, i] = posv.reshape([datashape[0], datashape[1]])
        pos_names.append(str(v))
    return pos_names, pos_data


def flip_data(input_data, subscan_dims=None):
    """
    Flip 2D or 3D array. The flip happens on the second index of shape.
    .. warning :: This function mutates the input values.

    Parameters
    ----------
    input_data : 2D or 3D array.

    Returns
    -------
    flipped data
    """
    new_data = np.asarray(input_data)
    data_shape = input_data.shape
    if len(data_shape) == 2:
        if subscan_dims is None:
            new_data[1::2, :] = new_data[1::2, ::-1]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :] = new_data[start:end:2, ::-1]
                i += ny

    if len(data_shape) == 3:
        if subscan_dims is None:
            new_data[1::2, :, :] = new_data[1::2, ::-1, :]
        else:
            i = 0
            for nx, ny in subscan_dims:
                start = i + 1
                end = i + ny
                new_data[start:end:2, :, :] = new_data[start:end:2, ::-1, :]
                i += ny
    return new_data


