import os
import numpy as np
import logging
import pandas as pd
logger = logging.getLogger()


def _make_hdf_srx(fpath, hdr, config_data,
                  create_each_det=False,
                  save_scalar=True, num_end_lines_excluded=None):
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
    create_each_det: bool, optional
        Do not create data for each detector is data size is too large,
        if set as false. This will slow down the speed of creating hdf file
        with large data size. srx beamline only.
    save_scalar : bool, optional
        choose to save scaler data or not for srx beamline, test purpose only.
    num_end_lines_excluded : int, optional
        remove the last few bad lines
    """
    spectrum_len = 4096
    start_doc = hdr['start']
    plan_n = start_doc.get('plan_name')
    if 'fly' not in plan_n: # not fly scan, step scan instead
        datashape = start_doc['shape']   # vertical first then horizontal
        fly_type = None

        snake_scan = start_doc.get('snaking')
        if snake_scan[1] == True:
            fly_type = 'pyramid'

        try:
            data = hdr.table(fill=True, convert_times=False)
        except IndexError: # scan is not finished
            total_len = get_total_scan_point(hdr) - 2
            evs, _ = zip(*zip(hdr.events(hdr, fill=True), range(total_len)))
            namelist = (config_data['xrf_detector'] +
                        hdr.start.motors +config_data['scaler_list'])
            dictv = {v:[] for v in namelist}
            for e in evs:
                for k,v in six.iteritems(dictv):
                    dictv[k].append(e.data[k])
            data = pd.DataFrame(dictv, index=np.arange(1, total_len+1)) # need to start with 1

        xrf_detector_names = config_data['xrf_detector']
        #express3 detector name changes in databroker
        if xrf_detector_names[0] not in data.keys():
            xrf_detector_names = ['xs_channel'+str(i) for i in range(1,4)]
        logger.info('Saving data to hdf file.')
        write_db_to_hdf(fpath, data,
                        datashape,
                        det_list=xrf_detector_names,
                        #roi_dict=roi_dict,
                        pos_list=hdr.start.motors,
                        scaler_list=config_data['scaler_list'],
                        fly_type=fly_type,
                        base_val=config_data['base_value'])  #base value shift for ic
        if 'xs2' in hdr.start.detectors: # second dector 
            logger.info('Saving data to hdf file for second xspress3 detector.')
            tmp = fpath.split('.')
            fpath1 = '.'.join([tmp[0]+'_1', tmp[1]])
            write_db_to_hdf(fpath1, data,
                            datashape,
                            det_list=config_data['xrf_detector2'],
                            #roi_dict=roi_dict,
                            pos_list=hdr.start.motors,
                            scaler_list=config_data['scaler_list'],
                            fly_type=fly_type,
                            base_val=config_data['base_value'])  #base value shift for ic
    else:
        # srx fly scan
        # Added by AMK to allow flying of single element on xs2
        if 'E_tomo' in start_doc['scaninfo']['type']:
            num_det = 1
        else:
            num_det = 3
        if save_scalar is True:
            scaler_list = ['i0', 'time']
            xpos_name = 'enc1'
            ypos_name = 'hf_stage_y'
        vertical_fast = False  # assuming fast on x as default
        if num_end_lines_excluded is None:
            datashape = [start_doc['shape'][1], start_doc['shape'][0]]   # vertical first then horizontal, assuming fast scan on x
        else:
            datashape = [start_doc['shape'][1]-num_end_lines_excluded, start_doc['shape'][0]]
        if 'fast_axis' in hdr.start.scaninfo:
            if hdr.start.scaninfo['fast_axis'] == 'VER':  # fast scan along vertical, y is fast scan, x is slow
                xpos_name = 'enc1'
                ypos_name = 'hf_stage_x'
                vertical_fast = True
                #datashape = [start_doc['shape'][0], start_doc['shape'][1]]   # fast vertical scan put shape[0] as vertical direction

        new_shape = datashape + [spectrum_len]
        total_points = datashape[0]*datashape[1]

        new_data = {}
        data = {}
        e = db.get_events(hdr, fill=True, stream_name='stream0')

        if save_scalar is True:
            new_data['scaler_names'] = scaler_list
            scaler_tmp = np.zeros([datashape[0], datashape[1], len(scaler_list)])
            if vertical_fast is True:  # data shape only has impact on scalar data
                scaler_tmp = np.zeros([datashape[1], datashape[0], len(scaler_list)])
            for v in scaler_list+[xpos_name]:
                data[v] = np.zeros([datashape[0], datashape[1]])

        if create_each_det is False:
            new_data['det_sum'] = np.zeros(new_shape)
        else:
            for i in range(num_det):
                new_data['det'+str(i+1)] = np.zeros(new_shape)

        for m,v in enumerate(e):
            if m < datashape[0]:
                if save_scalar is True:
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

        if vertical_fast is True: # need to transpose the data, as we scan y first
            if create_each_det is False:
                new_data['det_sum'] = np.transpose(new_data['det_sum'], axes=(1,0,2))
            else:
                for i in range(num_det):
                    new_data['det'+str(i+1)] = np.transpose(new_data['det'+str(i+1)], axes=(1,0,2))

        if save_scalar is True:
            if vertical_fast is False:
                for i,v in enumerate(scaler_list):
                    scaler_tmp[:, :, i] = data[v]
            else:
                for i,v in enumerate(scaler_list):
                    scaler_tmp[:, :, i] = data[v].T
            new_data['scaler_data'] = scaler_tmp
            x_pos = data[xpos_name]

        # get y position data
        if save_scalar is True:
            data1 = hdr.table(hdr, fill=True, stream_name='primary')
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
                if vertical_fast is True: # need to transpose the data, as we scan y first
                    data_tmp = np.zeros([2, x_pos.shape[1], x_pos.shape[0]]) # fast scan on y has impact for scalar data
                    data_tmp[1,:,:] = x_pos.T
                    data_tmp[0,:,:] = yv.T
                    new_data['pos_data'] = data_tmp
            else:
                logger.warning('x,y positions are not saved.')
        # output to file
        logger.info('Saving data to hdf file.')
        if create_each_det is False:
            create_each_det = False
        else:
            create_each_det = True
        write_db_to_hdf_base(fpath, new_data, num_det=num_det,
                             create_each_det=create_each_det)
