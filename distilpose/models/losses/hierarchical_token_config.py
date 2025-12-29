"""
Hierarchical Token Structure for COCO 17 Keypoints
===================================================

3-Level Hierarchy:
- Coarse (KT_C): 6 body parts
- Mid (KT_M): 11 joint groups  
- Fine (KT_F): 17 individual keypoints

COCO Keypoint Order:
0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear
5: left_shoulder, 6: right_shoulder
7: left_elbow, 8: right_elbow
9: left_wrist, 10: right_wrist
11: left_hip, 12: right_hip
13: left_knee, 14: right_knee
15: left_ankle, 16: right_ankle
"""

import torch
import torch.nn as nn

# ============ Coarse Level (6 parts) ============
COARSE_PARTS = {
    'head': [0, 1, 2, 3, 4],           # nose, eyes, ears
    'torso': [5, 6, 11, 12],            # shoulders, hips
    'left_arm': [5, 7, 9],              # shoulder, elbow, wrist
    'right_arm': [6, 8, 10],            # shoulder, elbow, wrist
    'left_leg': [11, 13, 15],           # hip, knee, ankle
    'right_leg': [12, 14, 16],          # hip, knee, ankle
}

COARSE_NAMES = ['head', 'torso', 'left_arm', 'right_arm', 'left_leg', 'right_leg']
NUM_COARSE = 6

# ============ Mid Level (11 groups) ============
MID_GROUPS = {
    'face': [0, 1, 2, 3, 4],            # head details
    'upper_body': [5, 6],                # shoulders
    'left_upper_arm': [5, 7],            # shoulder-elbow
    'left_lower_arm': [7, 9],            # elbow-wrist
    'right_upper_arm': [6, 8],           # shoulder-elbow
    'right_lower_arm': [8, 10],          # elbow-wrist
    'hip': [11, 12],                     # hip joints
    'left_upper_leg': [11, 13],          # hip-knee
    'left_lower_leg': [13, 15],          # knee-ankle
    'right_upper_leg': [12, 14],         # hip-knee
    'right_lower_leg': [14, 16],         # knee-ankle
}

MID_NAMES = [
    'face', 'upper_body',
    'left_upper_arm', 'left_lower_arm',
    'right_upper_arm', 'right_lower_arm',
    'hip',
    'left_upper_leg', 'left_lower_leg',
    'right_upper_leg', 'right_lower_leg'
]
NUM_MID = 11

# ============ Fine Level (17 keypoints) ============
FINE_KEYPOINTS = list(range(17))
NUM_FINE = 17

# ============ Parent-Child Mapping ============
# Coarse -> Mid mapping
COARSE_TO_MID = {
    'head': ['face'],
    'torso': ['upper_body', 'hip'],
    'left_arm': ['left_upper_arm', 'left_lower_arm'],
    'right_arm': ['right_upper_arm', 'right_lower_arm'],
    'left_leg': ['left_upper_leg', 'left_lower_leg'],
    'right_leg': ['right_upper_leg', 'right_lower_leg'],
}

# Mid -> Fine mapping (already in MID_GROUPS values)

# ============ Aggregation Weights ============
def get_aggregation_matrix_mid_to_coarse():
    """
    Create aggregation matrix from Mid (11) to Coarse (6).
    
    Returns:
        torch.Tensor [NUM_COARSE, NUM_MID]: Aggregation weights
    """
    agg_matrix = torch.zeros(NUM_COARSE, NUM_MID)
    
    for c_idx, c_name in enumerate(COARSE_NAMES):
        mid_list = COARSE_TO_MID[c_name]
        for m_name in mid_list:
            m_idx = MID_NAMES.index(m_name)
            agg_matrix[c_idx, m_idx] = 1.0 / len(mid_list)  # Average
    
    return agg_matrix

def get_aggregation_matrix_fine_to_mid():
    """
    Create aggregation matrix from Fine (17) to Mid (11).
    
    Returns:
        torch.Tensor [NUM_MID, NUM_FINE]: Aggregation weights
    """
    agg_matrix = torch.zeros(NUM_MID, NUM_FINE)
    
    for m_idx, m_name in enumerate(MID_NAMES):
        fine_list = MID_GROUPS[m_name]
        for f_idx in fine_list:
            agg_matrix[m_idx, f_idx] = 1.0 / len(fine_list)  # Average
    
    return agg_matrix

def get_aggregation_matrix_fine_to_coarse():
    """
    Create aggregation matrix from Fine (17) to Coarse (6).
    
    Returns:
        torch.Tensor [NUM_COARSE, NUM_FINE]: Aggregation weights
    """
    agg_matrix = torch.zeros(NUM_COARSE, NUM_FINE)
    
    for c_idx, c_name in enumerate(COARSE_NAMES):
        fine_list = COARSE_PARTS[c_name]
        for f_idx in fine_list:
            agg_matrix[c_idx, f_idx] = 1.0 / len(fine_list)  # Average
    
    return agg_matrix


# ============ Token Indices Helper ============
class HierarchicalTokenIndices:
    """Helper class to manage hierarchical token indices in concatenated token tensor."""
    
    def __init__(self):
        self.num_coarse = NUM_COARSE
        self.num_mid = NUM_MID
        self.num_fine = NUM_FINE
        self.total_kpt_tokens = NUM_COARSE + NUM_MID + NUM_FINE  # 6+11+17=34
        
    def get_coarse_indices(self):
        """Returns indices [0:6]"""
        return slice(0, self.num_coarse)
    
    def get_mid_indices(self):
        """Returns indices [6:17]"""
        return slice(self.num_coarse, self.num_coarse + self.num_mid)
    
    def get_fine_indices(self):
        """Returns indices [17:34]"""
        return slice(self.num_coarse + self.num_mid, 
                     self.num_coarse + self.num_mid + self.num_fine)
    
    def split_tokens(self, kpt_tokens):
        """
        Split concatenated keypoint tokens into 3 levels.
        
        Args:
            kpt_tokens: [B, 34, D]
        
        Returns:
            kt_c: [B, 6, D]
            kt_m: [B, 11, D]
            kt_f: [B, 17, D]
        """
        kt_c = kpt_tokens[:, self.get_coarse_indices()]
        kt_m = kpt_tokens[:, self.get_mid_indices()]
        kt_f = kpt_tokens[:, self.get_fine_indices()]
        return kt_c, kt_m, kt_f


# ============ Test ============
if __name__ == '__main__':
    print("Hierarchical Token Configuration")
    print("="*50)
    print(f"Coarse level: {NUM_COARSE} parts")
    for name, joints in COARSE_PARTS.items():
        print(f"  {name}: {joints}")
    
    print(f"\nMid level: {NUM_MID} groups")
    for name, joints in MID_GROUPS.items():
        print(f"  {name}: {joints}")
    
    print(f"\nFine level: {NUM_FINE} keypoints")
    print(f"  Individual joints: {FINE_KEYPOINTS}")
    
    print(f"\nTotal keypoint tokens: {NUM_COARSE + NUM_MID + NUM_FINE} (6+11+17=34)")
    
    # Test aggregation matrices
    agg_f2m = get_aggregation_matrix_fine_to_mid()
    agg_m2c = get_aggregation_matrix_mid_to_coarse()
    agg_f2c = get_aggregation_matrix_fine_to_coarse()
    
    print(f"\nAggregation matrices created:")
    print(f"  Fine→Mid: {agg_f2m.shape}")
    print(f"  Mid→Coarse: {agg_m2c.shape}")
    print(f"  Fine→Coarse: {agg_f2c.shape}")

