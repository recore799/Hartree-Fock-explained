def generate_shells_bfs(max_l):
    from collections import deque
    shells = [(0, 0, 0)]
    queue = deque(shells)
    
    while queue:
        current = queue.popleft()
        current_sum = sum(current)
        
        if current_sum >= max_l:
            continue
            
        for i in range(3):
            new_shell = list(current)
            new_shell[i] += 1
            new_shell = tuple(new_shell)
            
            if new_shell not in shells:
                shells.append(new_shell)
                queue.append(new_shell)
    
    
    return sorted(shells, key=lambda x: (sum(x), x))


b = generate_shells_bfs(4)

print(b)
