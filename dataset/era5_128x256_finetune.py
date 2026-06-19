from torch.utils.data import Dataset
# from tqdm import tqdm
import numpy as np
import io
import time
# import xarray as xr
import json
import pandas as pd
import os
import gc
from multiprocessing import Pool
from multiprocessing import shared_memory
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import copy
import queue
import torch
from petrel_client.client import Client

Years = {
    'train': ['1979-01-01 00:00:00', '2015-12-31 23:00:00'],
    'valid': ['2018-01-01 00:00:00', '2018-12-31 23:00:00'],
    'test': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
    'all': ['1979-01-01 00:00:00', '2020-12-31 23:00:00'],
    'train1': ['2015-01-01 00:00:00', '2015-12-31 23:00:00'],
    'train3': ['2013-01-01 00:00:00', '2015-12-31 23:00:00'],
    'train5': ['2011-01-01 00:00:00', '2015-12-31 23:00:00'],
    'train10': ['2006-01-01 00:00:00', '2015-12-31 23:00:00'],
    'train20': ['2006-01-01 00:00:00', '2015-12-31 23:00:00'],
}

# Years = {
#     'train': ['2015-01-01 00:00:00', '2015-12-31 23:00:00'],
#     'valid': ['2017-01-01 00:00:00', '2017-05-31 23:00:00'],
#     'test': ['2016-01-01 00:00:00', '2017-12-31 23:00:00'],
#     'all': ['2010-01-01 00:00:00', '2020-12-31 23:00:00']
# }

multi_level_vnames = [
    "z", "t", "q", "r", "u", "v", "vo", "pv",
]
single_level_vnames = [
    "t2m", "u10", "v10", "tcc", "tp", "tisr","msl",
]
long2shortname_dict = {"geopotential": "z", "temperature": "t", "specific_humidity": "q", "relative_humidity": "r", "u_component_of_wind": "u", "v_component_of_wind": "v", "vorticity": "vo", "potential_vorticity": "pv", \
    "2m_temperature": "t2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10", "total_cloud_cover": "tcc", "total_precipitation": "tp", "toa_incident_solar_radiation": "tisr"}
constants = [
    "lsm", "slt", "orography"
]

height_level = [1, 2, 3, 5, 7, 10, 20, 30, 50, 70, 100, 125, 150, 175, 200, 225, 250, 300, 350, 400, 450, \
    500, 550, 600, 650, 700, 750, 775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
# height_level = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

multi_level_dict_param = {"z":height_level, "t": height_level, "q": height_level, "r": height_level}

class CustomError(Exception):
    def __init__(self, job_pid, arr):
        self.job_pid = job_pid
        self.arr = arr
        super().__init__(f'Error in job {self.job_pid} with array {self.arr}')
        
def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

class era5_128x256_finetune(Dataset):
    def __init__(self, data_dir='huawei_100p:s3://ai4earth/era5_np128x256', split='train', file_stride=1,train_stride=1,**kwargs) -> None:
        super().__init__()
        # print("init begin")
        self.length = kwargs.get('length', 1)
        self.file_stride = kwargs.get('file_stride', 1) # time stride of data 
        self.file_stride = file_stride
        self.sample_stride = kwargs.get('sample_stride', 1)
        self.output_meanstd = kwargs.get("output_meanstd", False)
        self.use_diff_pos = kwargs.get("use_diff_pos", False)
        self.use_temporal_meanstd = kwargs.get("use_temporal_meanstd", False)
        # self.rm_equator = kwargs.get("rm_equator", True)
        Years_dict = kwargs.get('years', Years)

        self.pred_length = kwargs.get("pred_length", 0)
        self.inference_stride = kwargs.get("inference_stride", 2)
        self.train_stride = train_stride
        self.use_gt = kwargs.get("use_gt", True)   # Use gorund or not
        self.data_save_dir = kwargs.get("data_save_dir", None)
        self.mean_std_dir = kwargs.get("mean_std_dir", os.path.join(os.path.dirname(__file__)))

        self.save_single_level_names = kwargs.get("save_single_level_names", [])
        self.save_multi_level_names = kwargs.get("save_multi_level_names", [])

        self.client = Client(conf_path="~/petreloss.conf")

        vnames_type = kwargs.get("vnames", {})
        self.constants_types = vnames_type.get('constants', [])
        self.single_level_vnames = vnames_type.get('single_level_vnames', ["msl","u10", "v10","t2m",])
        self.multi_level_vnames = vnames_type.get('multi_level_vnames', ['z','q', 'u', 'v', 't'])
        self.height_level_list = vnames_type.get('hight_level_list', [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])
        self.height_level_indexes = [height_level.index(j) for j in self.height_level_list]

        self.split = split
        self.data_dir = data_dir
        years = Years_dict[split]
        self.init_file_list(years)

        # print(constants_index)
        if len(self.constants_types) > 0:
            self.constants_data = self.get_constants_data(self.constants_types)
        else:
            self.constants_data = None


        # 实例化归一化函数、获取均值和方差
        if self.use_temporal_meanstd:
            self.normalization = self.normalization2
            self._get_t_meanstd()
            self.get_meanstd = self.get_t_meanstd
        else:
            self.normalization = self.normalization1 
            self._get_meanstd() # load mean and std
            self.get_meanstd = self.get_meanstd1

        # dim of all variables
        self.data_element_num = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list)
        dim = len(self.single_level_vnames) + len(self.multi_level_vnames) * len(self.height_level_list) 


        self.index_dict1 = {}
        self.index_dict2 = {}
        i = 0
        for vname in self.single_level_vnames:
            self.index_dict1[(vname, 0)] = i
            i += 1
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_dict1[(vname, height)] = i
                i += 1

        self.index_queue = multiprocessing.Queue()
        self.unit_data_queue = multiprocessing.Queue()

        self.index_queue.cancel_join_thread() 
        self.unit_data_queue.cancel_join_thread()

        self.compound_data_queue = []
        self.sharedmemory_list = []
        self.compound_data_queue_dict = {}
        self.sharedmemory_dict = {}

        # 复合数据队列的数量
        self.compound_data_queue_num = 4

        self.lock = multiprocessing.Lock()
        self.a = np.zeros((dim, 128, 256), dtype=np.float32)
        # if self.rm_equator:
        #     self.a = np.zeros((dim, 720, 1440), dtype=np.float32)
        # else:
        #     self.a = np.zeros((dim, 721, 1440), dtype=np.float32)


        for _ in range(self.compound_data_queue_num):
            self.compound_data_queue.append(multiprocessing.Queue())
            shm = shared_memory.SharedMemory(create=True, size=self.a.nbytes)
            shm.unlink()
            self.sharedmemory_list.append(shm)

        self.arr = multiprocessing.Array('i', range(self.compound_data_queue_num))

        self._workers = []

        for _ in range(20):
            w = multiprocessing.Process(
                target=self.load_data_process)
            w.daemon = True
            # NB: Process.start() actually take some time as it needs to
            #     start a process and pass the arguments over via a pipe.
            #     Therefore, we only add a worker to self._workers list after
            #     it started, so that we do not call .join() if program dies
            #     before it starts, and __del__ tries to join but will get:
            #     AssertionError: can only join a started process.
            w.start()
            self._workers.append(w)
        w = multiprocessing.Process(target=self.data_compound_process)
        w.daemon = True
        w.start()
        self._workers.append(w)

        self._owner_pid = os.getpid()
        self._closed = False

    def close(self):
        if getattr(self, "_closed", True):
            return
        self._closed = True
        if os.getpid() != getattr(self, "_owner_pid", None):
            return

        for worker in getattr(self, "_workers", []):
            if worker.is_alive():
                worker.terminate()
        for worker in getattr(self, "_workers", []):
            worker.join(timeout=1)

        for q in [getattr(self, "index_queue", None), getattr(self, "unit_data_queue", None)]:
            if q is not None:
                q.close()
                q.cancel_join_thread()
        for q in getattr(self, "compound_data_queue", []):
            q.close()
            q.cancel_join_thread()

        for shm in getattr(self, "sharedmemory_list", []):
            try:
                shm.close()
            except FileNotFoundError:
                pass

    def __del__(self):
        self.close()


    def init_file_list(self, years):
        # get all file lists
        time_sequence = pd.date_range(years[0],years[1],freq=str(self.file_stride)+'h') #pd.date_range(start='2019-1-09',periods=24,freq='H')
        self.file_list= [os.path.join(str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]
        self.single_file_list= [os.path.join('single/'+str(time_stamp.year), str(time_stamp.to_datetime64()).split('.')[0]).replace('T', '/')
                      for time_stamp in time_sequence]


    def _get_meanstd(self):
        with open(os.path.join(self.mean_std_dir, 'mean_std.json'),mode='r') as f:
            multi_level_mean_std = json.load(f)
        with open(os.path.join(self.mean_std_dir, 'mean_std_single.json'),mode='r') as f:
            single_level_mean_std = json.load(f)
        self.mean_std = {}
        multi_level_mean_std['mean'].update(single_level_mean_std['mean'])
        multi_level_mean_std['std'].update(single_level_mean_std['std'])
        self.mean_std['mean'] = multi_level_mean_std['mean']
        self.mean_std['std'] = multi_level_mean_std['std']
        for vname in self.single_level_vnames:
            self.mean_std['mean'][vname] = np.array([self.mean_std['mean'][vname]])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array([self.mean_std['std'][vname]])[::-1][:,np.newaxis,np.newaxis]
        for vname in self.multi_level_vnames:
            self.mean_std['mean'][vname] = np.array(self.mean_std['mean'][vname])[::-1][:,np.newaxis,np.newaxis]
            self.mean_std['std'][vname] = np.array(self.mean_std['std'][vname])[::-1][:,np.newaxis,np.newaxis]


    def _get_t_meanstd(self):
        # Not use 
        self.mean_std = {"mean": {}, "std": {}}
        for vname in self.single_level_vnames:
            url = f"{self.data_dir}/mean_std/mean_{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                mean_data = np.load(f)

            
            url = f"{self.data_dir}/mean_std/pow2_mean_{vname}.npy"
            with io.BytesIO(self.client.get(url)) as f:
                meanpow2_data = np.load(f)
            self.mean_std['mean'][vname] = mean_data[np.newaxis, :, :]
            self.mean_std['std'][vname] = ((meanpow2_data - mean_data ** 2) ** 0.5)[np.newaxis, :, :]

        for vname in self.multi_level_vnames:
            mean_list = []
            stdpow2_mean_list = []
            for height in self.height_level_list:
                url = f"{self.data_dir}/mean_std/mean_{vname}_{height}.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    unit_data = np.load(f)
                mean_list.append(unit_data)
                
                url = f"{self.data_dir}/mean_std/pow2_mean_{vname}_{height}.npy"
                with io.BytesIO(self.client.get(url)) as f:
                    unit_data = np.load(f)
                stdpow2_mean_list.append(unit_data)   

            mean_data = np.stack(mean_list, axis=0)
            stdpow2_mean_data = np.stack(stdpow2_mean_list, axis=0)
            self.mean_std['mean'][vname] = mean_data
            self.mean_std['std'][vname] = (stdpow2_mean_data - mean_data ** 2) ** 0.5




    def get_noise_weight(self):
        # Not use in initialization
        diff_pow2_mean_list = []
        for vname in self.single_level_vnames:

            f = 'tmp'
            unit_data = np.load(f)
            # url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}.npy"
            # with io.BytesIO(self.client.get(url)) as f:
            #     unit_data = np.load(f)
            diff_pow2_mean_list.append(unit_data)
        
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:

                f = 'tmp'
                unit_data = np.load(f)
                # url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}_{height}.npy"
                # with io.BytesIO(self.client.get(url)) as f:
                #     unit_data = np.load(f)
                diff_pow2_mean_list.append(unit_data)   
        
        diff_pow2_mean = np.stack(diff_pow2_mean_list, axis=0)
        del diff_pow2_mean_list
        return diff_pow2_mean.reshape(diff_pow2_mean.shape[0], -1).mean(axis=-1)[:,np.newaxis,np.newaxis]**0.5



    def get_diffmeanstd(self):
        # Not use in initialization
        diff_mean_list = []
        diff_pow2_mean_list = []
        for vname in self.single_level_vnames:
            f = 'tmp'
            unit_data = np.load(f)
            # url = f"{self.data_dir}/diff_mean_std/diff_mean_{vname}.npy"
            # with io.BytesIO(self.client.get(url)) as f:
            #     unit_data = np.load(f)
            diff_mean_list.append(unit_data)

            f = 'tmp'
            unit_data = np.load(f)
            # url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}.npy"
            # with io.BytesIO(self.client.get(url)) as f:
            #     unit_data = np.load(f)
            diff_pow2_mean_list.append(unit_data)
        
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                f = 'tmp'
                unit_data = np.load(f)
                # url = f"{self.data_dir}/diff_mean_std/diff_mean_{vname}_{height}.npy"
                # with io.BytesIO(self.client.get(url)) as f:
                #     unit_data = np.load(f)
                diff_mean_list.append(unit_data)

                f = 'tmp'
                unit_data = np.load(f)
                # url = f"{self.data_dir}/diff_mean_std/diff_pow2_mean_{vname}_{height}.npy"
                # with io.BytesIO(self.client.get(url)) as f:
                #     unit_data = np.load(f)
                diff_pow2_mean_list.append(unit_data)   
        
        diff_mean = np.stack(diff_mean_list, axis=0)
        diff_pow2_mean = np.stack(diff_pow2_mean_list, axis=0)
        del diff_mean_list
        del diff_pow2_mean_list
        if self.use_diff_pos:
            diff_std = diff_pow2_mean - diff_mean ** 2
            return diff_mean, diff_std**0.5
        else:
            diff_std = diff_pow2_mean.reshape(diff_pow2_mean.shape[0], -1).mean(axis=-1) - diff_mean.reshape(diff_mean.shape[0], -1).mean(axis=-1)**2
            return diff_mean.reshape(diff_mean.shape[0], -1).mean(axis=-1)[:, np.newaxis, np.newaxis], diff_std[:,np.newaxis,np.newaxis]**0.5
          


    def get_constants_data(self, constants_types):
        # not use
        # file = os.path.join("constant", "z_lsm_slt.nc")
        # url = f"era5:s3://era5_nc/{file}"
        # array_lst = []
        # with io.BytesIO(self.client.get(url)) as f:
        #     nc_data = xr.open_dataset(f)
        #     for vname in constants_types:
        #         D = nc_data.data_vars[vname].data
        #         D = cv2.resize(D, (256, 128), interpolation=cv2.INTER_LINEAR)
        #         D = standardization(D)
        #         array_lst.append(D[np.newaxis, :, :])
        #     data = np.concatenate(array_lst, axis=0)
        #     array = data
        # del array_lst
        # return array
        pass

    def data_compound_process(self):
        recorder_dict = {}
        while True:
            job_pid, idx, vname, height = self.unit_data_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            if (job_pid, idx) in recorder_dict:
                recorder_dict[(job_pid, idx)][(vname, height)] = 1
            else:
                recorder_dict[(job_pid, idx)] = {(vname, height): 1}
            if len(recorder_dict[(job_pid, idx)]) == self.data_element_num:
                del recorder_dict[(job_pid, idx)]
                self.compound_data_queue_dict[job_pid].put((idx))

    def put_index(self, idx):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)
  
            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        for vname in self.single_level_vnames:
            self.index_queue.put((job_pid, idx, vname, 0))
        for vname in self.multi_level_vnames:
            for height in self.height_level_list:
                self.index_queue.put((job_pid, idx, vname, height))

        # self.index_queue.put((job_pid, idx, self.single_level_vnames))
        # self.index_queue.put((job_pid, idx, self.multi_level_vnames))


    def queue_wait_data(self):
        job_pid = os.getpid()
        return_data = {}
        idx = self.compound_data_queue_dict[job_pid].get()
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        if self.constants_data is not None:
            return_data[idx] = np.concatenate((self.constants_data, b), axis=0)
        else:
            return_data[idx] = copy.deepcopy(b)
        return return_data


    def get_data(self, idxes):
        job_pid = os.getpid()
        if job_pid not in self.compound_data_queue_dict:
            try:
                self.lock.acquire()
                for i in range(self.compound_data_queue_num):
                    if i == self.arr[i]:
                        self.arr[i] = job_pid
                        self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                        self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                        break
                if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                    print("error", job_pid, self.arr)

                
            except Exception as err:
                raise err
            finally:
                self.lock.release()

        try:
            idx = self.compound_data_queue_dict[job_pid].get(False)
            raise ValueError
        except queue.Empty:
            pass
        except Exception as err:
            raise err
        
        b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
        return_data = {}
        for idx in idxes:
            for vname in self.single_level_vnames:
                self.index_queue.put((job_pid, idx, vname, 0))
            for vname in self.multi_level_vnames:
                for height in self.height_level_list:
                    self.index_queue.put((job_pid, idx, vname, height))
            idx = self.compound_data_queue_dict[job_pid].get()
            if self.constants_data is not None:
                return_data[idx] = np.concatenate((self.constants_data, b), axis=0)
            else:
                return_data[idx] = copy.deepcopy(b)
            
        return return_data

    def load_data_process(self):
        while True:
            job_pid, idx, vname, height = self.index_queue.get()
            if job_pid not in self.compound_data_queue_dict:
                try:
                    self.lock.acquire()
                    for i in range(self.compound_data_queue_num):
                        if job_pid == self.arr[i]:
                            self.compound_data_queue_dict[job_pid] = self.compound_data_queue[i]
                            self.sharedmemory_dict[job_pid] = self.sharedmemory_list[i]
                            break
                    if (i == self.compound_data_queue_num - 1) and job_pid != self.arr[i]:
                        print("error", job_pid, self.arr)
                except Exception as err:
                    raise err
                finally:
                    self.lock.release()
            
            if vname in self.single_level_vnames:
                file = self.single_file_list[idx]
                url = f"{self.data_dir}/{file}-{vname}.npy"
            elif vname in self.multi_level_vnames:
                file = self.file_list[idx]
                url = os.path.join(self.data_dir,
                                   f"{file}-{vname}-{height}.0.npy")
            b = np.ndarray(self.a.shape, dtype=self.a.dtype, buffer=self.sharedmemory_dict[job_pid].buf)
            unit_data = None

            with io.BytesIO(self.client.get(url)) as f:
                try:
                    unit_data = np.load(f)
                except Exception as err:
                    raise ValueError(f"{url}")
            unit_data = self.normalization(vname, height, unit_data)

            b[self.index_dict1[(vname, height)], :] = unit_data[:]
            self.unit_data_queue.put((job_pid, idx, vname, height))

    def normalization1(self, vname, height, data):
        if vname in self.single_level_vnames:
            index = 0
        else:
            index = height_level.index(height)
        data -=  np.array(self.mean_std['mean'][vname][index], dtype=np.float32)
        data /= np.array(self.mean_std['std'][vname][index], dtype=np.float32)
        return data
    
    def normalization2(self, vname, height, data):
        if vname in self.single_level_vnames:
            index = 0
        else:
            index = self.height_level_list.index(height)
        data -=  np.array(self.mean_std['mean'][vname][index], dtype=np.float32)
        data /= np.array(self.mean_std['std'][vname][index],dtype=np.float32)
        return data

    def get_maxidx(self):
        return len(self.file_list) - (self.length-1) * self.sample_stride - 1

    def __len__(self):
        # if self.split != "valid":
        #     data_len = (len(self.file_list) - (self.length - 1) * self.sample_stride) // (self.train_stride // self.sample_stride // self.file_stride)
        # elif self.use_gt:
        #     data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
        #     data_len -= self.pred_length * self.sample_stride + 1
        #     data_len = (data_len + self.inference_stride // self.sample_stride // self.file_stride - 1) // (self.inference_stride // self.sample_stride // self.file_stride)
        # else:
        #     data_len = len(self.file_list) - (self.length - 1) * self.sample_stride
        #     data_len = (data_len + self.inference_stride // self.sample_stride // self.file_stride - 1) // (self.inference_stride // self.sample_stride // self.file_stride)
        data_len = (len(self.file_list) - (self.length - 1) * self.sample_stride) // (self.train_stride // self.sample_stride // self.file_stride)

        return data_len


    def get_meanstd1(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname][self.height_level_indexes])
            return_data_std.append(self.mean_std['std'][vname][self.height_level_indexes])

        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)[:, 0, 0]), torch.from_numpy(np.concatenate(return_data_std, axis=0)[:, 0, 0])

    def get_t_meanstd(self):
        return_data_mean = []
        return_data_std = []
        
        for vname in self.single_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])
     
        for vname in self.multi_level_vnames:
            return_data_mean.append(self.mean_std['mean'][vname])
            return_data_std.append(self.mean_std['std'][vname])

        return torch.from_numpy(np.concatenate(return_data_mean, axis=0)), torch.from_numpy(np.concatenate(return_data_std, axis=0))




    def get_clim_daily(self):
        # not use in initialization
        f = 'tmp'
        array = np.load(f)
        url = f"{self.data_dir}/time_means_daily.npy"
        # with io.BytesIO(self.client.get(url)) as f:
        #     array = np.load(f)                                  #(8760, 110, 32, 64)
        # array = torch.from_numpy(array)
        time_index = list(range(0, 8760, self.file_stride))
        array = (array - self.data_mean.unsqueeze(-1).unsqueeze(-1)) / self.data_std.unsqueeze(-1).unsqueeze(-1)
        array = array[time_index].transpose(0, 1)[self.total_data_index].transpose(0,1)
        return array


    # def __getitem__(self, index):
    #     index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
    #     array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
    #     array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
    #     del array_dict
    #     tar_idx = np.array([index + self.sample_stride * (self.length - 1)])
    #     return array_seq, tar_idx
    
    def __getitem__(self, index):
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        # if self.split == "valid":
        #     index = index * (self.inference_stride // self.sample_stride // self.file_stride)
        # else:
        index = index * (self.train_stride // self.sample_stride // self.file_stride)
        
        array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
        if self.length < 3:
            array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
        else:
            array_seq = [array_dict[index], array_dict[index + self.sample_stride], array_dict[index + (self.length-1) * self.sample_stride]]
        del array_dict
        tar_idx = np.array([index + self.sample_stride * (self.length - 1)])
        # return array_seq, tar_idx
        return array_seq[0]


    def getitem(self, index):
        index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        array_dict = self.get_data([index + i * self.sample_stride for i in range(self.length)])
        array_seq = [array_dict[index + i * self.sample_stride] for i in range(self.length)]
        del array_dict
        tar_idx = np.array([index + self.sample_stride * (self.length - 1)])
        return array_seq, tar_idx

    def get_target(self, index):
        # index = min(index, len(self.file_list) - (self.length-1) * self.sample_stride - 1)
        data = self.get_data([index])
        return data[index]


# if __name__ == "__main__":
#     data_set = era5_128x256_finetune(split='train')

#     for i in range(20):
#         data_set.__getitem__(i)
    

    # print("complete")