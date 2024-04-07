# Script for generating subnets
import random

if __name__ == '__main__':
    width_choice = [1.0, 0.75, 0.5]
    depth_choice = [0.67, 1.0]

    width_total = 22
    depth_total = 8

    width_repeat = 10
    depth_repeat = 5

    width_list = []
    depth_list = []

    # 150 random width 
    # Considering width = 1.0
    for num in range(20, -1, -5):    # 20, 15, 10, 5, 0
        for _ in range(width_repeat):
            width = [1.0 for _ in range(num)]
            append = [random.choice(width_choice) for _ in range(width_total - num)] 
            width = width + append 
            random.shuffle(width) 
            width.append(1.0)
            width_list.append(width)

    # Considering width = 0.5
    for num in range(20, -1, -5):
        for _ in range(width_repeat):
            width = [0.5 for _ in range(num)]
            append = [random.choice(width_choice) for _ in range(width_total - num)] 
            width = width + append 
            random.shuffle(width) 
            width.append(1.0)
            width_list.append(width)

    # Totally random
    for num in range(50):
        width = [random.choice(width_choice) for _ in range(width_total)]
        width.append(1.0)
        width_list.append(width)

    # 40 random depth
    # Considering depth = 1.0
    for num in range(7, -1, -2):
        for _ in range(depth_repeat):
            depth = [1.0 for _ in range(num)]
            append = [random.choice(depth_choice) for _ in range(depth_total - num)]
            depth = depth + append
            depth_list.append(depth)

    # Considering depth = 2/3
    for num in range(7, -1, -2):
        for _ in range(depth_repeat):
            depth = [0.67 for _ in range(num)]
            append = [random.choice(depth_choice) for _ in range(depth_total - num)]
            depth = depth + append
            depth_list.append(depth)


    for width in width_list:
        for _ in range(4):
            # sample depth
            depth = random.choice(depth_list)
            
            print(*width)
            print(*depth)
