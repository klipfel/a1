from pandas import read_csv


class MotionClipParser:
    '''
    Class to store and process motion clips.
    '''
    def __init__(self):
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

    def get_single_motion_clip(self, folder, interp_file_name):
        mocapid = interp_file_name.split("interp_motion_data_a1_")[-1]
        print(f"Loading mocap: folder:{folder}/mocapid:{mocapid}.csv")
        # Reads the motion clip
        keys = self.folder_prefixes.keys()
        for data_type in keys:
            file_name = f"{folder}/{self.folder_prefixes[data_type]}{mocapid}.csv"
            print(f"MOCAP PARSER: reading {file_name} ... ")
            self.motion_clip[data_type] = read_csv(file_name).to_numpy()

    def __str__(self):
        out = ''
        for data_type in self.motion_clip.keys():
            out += f"{data_type}>>"+str(self.motion_clip[data_type])+"\n"
        return out

if __name__ == "__main__":
    print(f'Test of {__file__}')
    mp = MotionClipParser()
    mp.get_single_motion_clip(folder="/home/arnaud/raisim_v110/raisim_ws/raisimLib/raisimGymTorch/raisimGymTorch/env/envs/walking_fwd/05-31-walk_cluster_no_slip",
                                interp_file_name="interp_motion_data_a1_mocapid_startup_forward_profile_v_0_307_sp_0_817_sp1_0_829_corrected")
    print(mp)
