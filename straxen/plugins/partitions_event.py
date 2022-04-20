import numpy as np

def rotate(x_arr, y_arr, theta):
    new_x = np.cos(theta)*x_arr+np.sin(theta)*y_arr
    new_y = -np.sin(theta)*x_arr+np.cos(theta)*y_arr
    return np.array([new_x,new_y])

def A_and_B_selection_event(xs,ys,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):

    new_x,new_y = rotate(xs,ys,-np.pi/6)
    
    cond = new_x < wire_out
    cond&= new_x > -wire_out
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def C_and_D_selection_event(xs,ys,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):

    cond = ~A_and_B_selection_event(xs,ys,dist_from_center_if_parallel,r_in_out, wire_out)
    
    return cond