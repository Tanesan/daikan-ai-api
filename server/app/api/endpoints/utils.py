

def super_resolution_estimate_execution_time(num_cpu, width, height):
    c_avg = 0.0029972695031465026
    c_map = {
        2: 0.0017438850501822973,
        4: 0.00147418822638114,
        32: 0.0029972695031465026
    }
    
    c_factor = c_map.get(num_cpu, c_avg)
    
    return (c_factor * width * height / num_cpu) * 1.5
