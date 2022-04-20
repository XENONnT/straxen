import numpy as np

def rotate(x_arr, y_arr, theta):
    new_x = np.cos(theta)*x_arr+np.sin(theta)*y_arr
    new_y = -np.sin(theta)*x_arr+np.cos(theta)*y_arr
    return np.array([new_x,new_y])

def A1_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x > -dist_from_center_if_parallel
    cond&= new_x < dist_from_center_if_parallel
    cond&= new_y >= 30
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def A2_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x > -dist_from_center_if_parallel
    cond&= new_x < dist_from_center_if_parallel
    cond&= new_y >= 0
    cond&= new_y < 30
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def A3_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x > -dist_from_center_if_parallel
    cond&= new_x < dist_from_center_if_parallel
    cond&= new_y >= -30
    cond&= new_y < 0
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def A4_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x > -dist_from_center_if_parallel
    cond&= new_x < dist_from_center_if_parallel
    cond&= new_y < -30
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def B1_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x <= -dist_from_center_if_parallel
    cond&= new_x**2+new_y**2 < r_in_out**2
    cond&= new_x >= - wire_out
    
    return cond

def B2_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x >= dist_from_center_if_parallel
    cond&= new_x**2+new_y**2 < r_in_out**2
    cond&= new_x <= wire_out
    
    return cond

def C1_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 < r_in_out**2
    cond&= new_x < -wire_out
    cond&= new_y > 0
    
    return cond

def C2_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 < r_in_out**2
    cond&= new_x < -wire_out
    cond&= new_y <= 0
    
    return cond

def C3_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 < r_in_out**2
    cond&= new_x > wire_out
    cond&= new_y > 0
    
    return cond

def C4_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 < r_in_out**2
    cond&= new_x > wire_out
    cond&= new_y <= 0
    
    return cond

def D1_selection(se_pop,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 > r_in_out**2
    cond&= new_y > new_x
    cond&= new_y < -new_x
    
    return cond

def D2_selection(se_pop,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 > r_in_out**2
    cond&= new_y < new_x
    cond&= new_y < -new_x
    
    return cond

def D3_selection(se_pop,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 > r_in_out**2
    cond&= new_y < new_x
    cond&= new_y > -new_x
    
    return cond

def D4_selection(se_pop,r_in_out = 60):
    
    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x**2+new_y**2 > r_in_out**2
    cond&= new_y > new_x
    cond&= new_y > -new_x
    
    return cond

def hot_spot_selection(df):
    
    cond = (df['x_mlp']-8)**2+(df['y_mlp']+15)**2 < 10**2
    
    return cond

def null_spot_selection(df):
    
    cond = (df['x_mlp']+17)**2+(df['y_mlp'])**2 < 10**2
    
    return cond

def A2_pure_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    cond = A2_selection(se_pop,dist_from_center_if_parallel,r_in_out)
    cond&= ~hot_spot_selection(se_pop)
    
    return cond

def A3_pure_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60):
    
    cond = A3_selection(se_pop,dist_from_center_if_parallel,r_in_out)
    cond&= ~hot_spot_selection(se_pop)
    
    return cond

def B2_pure_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):
    
    cond = B2_selection(se_pop,dist_from_center_if_parallel,r_in_out, wire_out)
    cond&= ~hot_spot_selection(se_pop)
    
    return cond
    
    return cond

def A_and_B_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):

    new_x,new_y = rotate(se_pop['x_mlp'],se_pop['y_mlp'],-np.pi/6)
    
    cond = new_x < wire_out
    cond&= new_x > -wire_out
    cond&= new_x**2+new_y**2 < r_in_out**2
    
    return cond

def C_and_D_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):

    cond = ~A_and_B_selection(se_pop,dist_from_center_if_parallel,r_in_out, wire_out)
    
    return cond

def A_and_B_pure_selection(se_pop,dist_from_center_if_parallel = 13.1,r_in_out = 60, wire_out = 28):

    cond = A_and_B_selection(se_pop,dist_from_center_if_parallel,r_in_out, wire_out)
    cond&= ~A3_selection(se_pop,dist_from_center_if_parallel,r_in_out)
    cond&= ~B2_selection(se_pop,dist_from_center_if_parallel,r_in_out, wire_out)
    
    return cond