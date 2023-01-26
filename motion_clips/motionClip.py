import numpy as np
from pandas import read_csv, DataFrame
import numpy
import math
import os
import datetime

from torch.utils.data.datapipes.utils.decoder import mathandler


def quat_to_rotation_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
        >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
        >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
        True
        >>> M = quaternion_matrix([1, 0, 0, 0])
        >>> numpy.allclose(M, numpy.identity(4))
        True
        >>> M = quaternion_matrix([0, 1, 0, 0])
        >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
        True
        from source :
        https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/examples/pybullet/gym/pybullet_envs/deep_mimic/mocap/transformation.py#L1043
    """
    eps = numpy.finfo(float).eps * 4.0
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < eps:
        return numpy.identity(3)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    # return numpy.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
    #                   [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
    #                   [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
    #                   [0.0, 0.0, 0.0, 1.0]])
    # Returns the normal rotation matrix by remove last row and column.
    return numpy.array([[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0]],
                       [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0]],
                       [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2]]])

def raisim_quat_to_rotation_matrix(q):
    R = np.zeros((3, 3), dtype=np.float64)
    R[0, 0] = q[0] * q[0] + q[1] * q[1] - q[2] * q[2] - q[3] * q[3]
    R[1, 0] = 2 * q[0] * q[3] + 2 * q[1] * q[2]
    R[2, 0] = 2 * q[1] * q[3] - 2 * q[0] * q[2]

    R[0, 1] = 2 * q[1] * q[2] - 2 * q[0] * q[3]
    R[1, 1] = q[0] * q[0] - q[1] * q[1] + q[2] * q[2] - q[3] * q[3]
    R[2, 1] = 2 * q[0] * q[1] + 2 * q[2] * q[3]

    R[0, 2] = 2 * q[0] * q[2] + 2 * q[1] * q[3]
    R[1, 2] = 2 * q[2] * q[3] - 2 * q[0] * q[1]
    R[2, 2] = q[0] * q[0] - q[1] * q[1] - q[2] * q[2] + q[3] * q[3]
    return R


def raisim_rotmat_to_quat(R):
    q = [0, 0, 0, 0]
    tr = R[0] + R[4] + R[8]
    if tr > 0.0:
        S = math.sqrt(tr + 1.0) * 2.0
        q[0] = 0.25 * S
        q[1] = (R[5] - R[7]) / S
        q[2] = (R[6] - R[2]) / S
        q[3] = (R[1] - R[3]) / S
    elif (R[0] > R[4]) and (R[0] > R[8]):
        S = math.sqrt(1.0 + R[0] - R[4] - R[8]) * 2.0
        q[0] = (R[5] - R[7]) / S
        q[1] = 0.25 * S
        q[2] = (R[3] + R[1]) / S
        q[3] = (R[6] + R[2]) / S
    elif R[4] > R[8]:
        S = math.sqrt(1.0 + R[4] - R[0] - R[8]) * 2.0
        q[0] = (R[6] - R[2]) / S
        q[1] = (R[3] + R[1]) / S
        q[2] = 0.25 * S
        q[3] = (R[7] + R[5]) / S
    else:
        S = math.sqrt(1.0 + R[8] - R[0] - R[4]) * 2.0
        q[0] = (R[1] - R[3]) / S
        q[1] = (R[6] + R[2]) / S
        q[2] = (R[7] + R[5]) / S
        q[3] = 0.25 * S
    return q

class MotionClipParser:
    '''
    Class to store and process motion clips.
    '''
    def __init__(self, data_folder=None):
        self.data_folder = data_folder  # where the test datat is saved
        self.motion_clip = {"Interp_Motion_Data": None,
                            "Ref_COM": None,
                            "Ref_EE": None,
                            "Ref_Orn": None,
                            "Ref_Vel": None
                            }
        self.folder_prefixes = {"Interp_Motion_Data": "Interp_Motion_Data/interp_motion_data_a1_",
                    "Ref_COM": "Ref_COM/reference_COM_A1_",
                    "Ref_EE": "Ref_EE_pos/reference_ee_pos_A1_",
                    "Ref_Orn": "Ref_Orn/reference_orientation_A1_",
                    "Ref_Vel": "Ref_Vel/reference_velocities_A1_"}
        self.motion_clip_additional_data = {"Rotation_matrix": None}  # additional data from the motion clips
        self.reference_data_obs = None  # contains the reference data present in the observations, contains  the
        self.motion_clip_sim_frames = None
        self.motion_clip_folder = None
        # data for all the frames

    def get_single_motion_clip(self, folder, interp_file_name):
        # Motion clip info
        mocapid = interp_file_name.split("interp_motion_data_a1_")[-1]
        print(f"Loading mocap: folder:{folder}/mocapid:{mocapid}.csv")
        # Reads the motion clip
        keys = self.folder_prefixes.keys()
        for data_type in keys:
            file_name = f"{folder}/{self.folder_prefixes[data_type]}{mocapid}.csv"
            print(f"MOCAP PARSER: reading {file_name} ... ")
            df = read_csv(file_name).to_numpy()
            if data_type == "Interp_Motion_Data":
                self.motion_clip["Interp_Motion_Data_raisim"] = df
                # need to do some re-ordering for the columns as the quaternion w component, the half cosine is
                # in different places.
                df = read_csv(file_name)
                # cols = df.columns.tolist()
                # print(cols)
                reordering = [0, 1, 2, 4, 5, 6, 3]+list(range(7, 19))
                # cols = cols[reordering]
                # cols = cols[:3] + cols[4:7] + cols[3] + cols[-12:]
                # df = df[cols]
                df = df.iloc[:, reordering]
                df = df.to_numpy()
            self.motion_clip[data_type] = df
        self.motion_clip_sim_frames = len(self.motion_clip["Interp_Motion_Data"])
        # Additional mc info
        self.adapt_data_for_policy()
        # For the policy
        self.get_reference_data_in_obs()
        # Saving datat
        cluster_name = folder.split('/')[-1]
        self.save_current_motion_clip(cluster_name=cluster_name, motion_clip_name=mocapid)

    def save_current_motion_clip(self, cluster_name, motion_clip_name):
        # save the data in a folder
        self.motion_clip_folder = f"{self.data_folder}/{cluster_name}-{motion_clip_name}"
        # os.system(f"mkdir -r {self.motion_clip_folder}")
        os.makedirs(self.motion_clip_folder)
        df = DataFrame(self.motion_clip_additional_data["Rotation_matrix"])
        print(df)
        df.to_csv(f"{self.motion_clip_folder}/{'Rotation_matrix'}.csv")
        df = DataFrame(self.reference_data_obs)
        df.to_csv(f"{self.motion_clip_folder}/{'reference_data_in_obs'}.csv")
        df = DataFrame(self.motion_clip['Interp_Motion_Data'])
        df.to_csv(f"{self.motion_clip_folder}/{'gc'}.csv")

    def adapt_data_for_policy(self):
        '''
        Adapts observations, creates the data that is needed for the policy in order to]
        save time in inference time.
        :return:
        '''
        if self.motion_clip["Interp_Motion_Data"] is None:
            print("No motion clip data.")
        else:
            # Generate rotation matrices.
            print(f"Number of simulation frames in motion clip: {self.motion_clip_sim_frames}")
            self.motion_clip_additional_data["Rotation_matrix"] = []
            for frame in range(self.motion_clip_sim_frames):
                gc = self.motion_clip["Interp_Motion_Data_raisim"][frame]
                # Quaternion to rotation matrix conversion
                quat = gc[3:3+4]
                # rotmat = self.quat_to_rotation_matrix(quat)
                rotmat = raisim_quat_to_rotation_matrix(quat)
                flat_rotmat = rotmat.flatten()
                self.motion_clip_additional_data["Rotation_matrix"].append(flat_rotmat)

    def get_reference_data_in_obs(self):
        if self.motion_clip["Interp_Motion_Data"] is None:
            print("No motion clip data.")
        else:
            # Gets reference data to be read by policy in order.
            self.reference_data_obs = []
            for frame in range(self.motion_clip_sim_frames):
                gc = self.motion_clip["Interp_Motion_Data"][frame]
                com = list(gc[:3])
                rotmat = list(self.motion_clip_additional_data["Rotation_matrix"][frame])
                jp = list(gc[-12:])
                ref_obs = numpy.array(com + rotmat + jp)
                self.reference_data_obs.append(ref_obs)

    def __str__(self):
        out = ''
        for data_type in self.motion_clip.keys():
            out += f"{data_type}>>"+str(self.motion_clip[data_type])+"\n"
        return out

    def test(self):
        self.data_folder = f'/tmp/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-mp_parser_test'
        self.get_single_motion_clip(folder="/home/arnaud/raisim_v110/raisim_ws/raisimLib/raisimGymTorch/raisimGymTorch/env/envs/walking_fwd/09-28-shapes",
                                interp_file_name="interp_motion_data_a1_mocapid_circle_33secs_sp_3_293_angle_-326")
        print("Rotation matrix:\n")
        print(self.motion_clip_additional_data["Rotation_matrix"][1000])
        print("Policy obs:\n")
        print(self.reference_data_obs[1000])
        print(np.array(self.reference_data_obs).shape)
        print(self)


if __name__ == "__main__":
    print(f'Test of {__file__}')
    mp = MotionClipParser(data_folder=f'~/tmp/{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}-mp_parser_test')
    mp.test()
