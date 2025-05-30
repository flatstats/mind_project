import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches

class NeuralNode:
    def __init__(self, x, y, init_energy=0.1, id_counter=0):
        self.position = np.array([x, y])
        self.energy = init_energy
        self.echo = 0.0
        self.curvature = 0.0
        self.id = id_counter
        self.role = 'undecided'
        self.age = 0
        self.parent_id = None
        self.division_threshold = 0.7
        self.migration_vector = np.array([0.0, 0.0])
        self.connectivity = 0.0

    def calculate_connectivity(self, all_nodes, max_range=2.0):
        """Checking to see if they connect based on distance and echo resonance"""
        connectivity = 0.0
        for other_node in all_nodes:
            if other_node.id != self.id:
                distance = np.linalg.norm(self.position - other_node.position)
                if distance < max_range:
                    # Some distance decay with echo
                    distance_factor = np.exp(-distance**2 / 1.0)
                    echo_resonance = np.abs(self.echo - other_node.echo)
                    resonance_factor = np.exp(-echo_resonance)
                    connectivity += distance_factor * resonance_factor
        
        self.connectivity = connectivity
        return connectivity

    def update(self, local_field, local_curvature, all_curvatures, lumen_center, 
               decay=0.01, growth=0.05, step=0):
        self.age += 1
        self.energy += growth * local_field
        self.echo += local_field * local_curvature
        self.curvature = local_curvature
        self.energy = max(0.0, self.energy - decay)
        
        # To help with proliferation zones... needs work
        dist_to_lumen = np.linalg.norm(self.position - lumen_center)
        
        # Letting roles be decided emergently like in biological systems
        if len(all_curvatures) > 0:
            high_curv_thresh = np.percentile(all_curvatures, 80)
            low_curv_thresh = np.percentile(all_curvatures, 40)
            
            if dist_to_lumen < 2.0 and self.energy > 0.4:
                self.role = 'proliferative'
                self.division_threshold *= 0.8  # easier to divide
                self.energy *= 1.01

            # Neurons are likely high energy and isolated 
            elif self.energy > 0.6 and local_curvature > high_curv_thresh:
                self.role = 'neuron_candidate'
                self.energy *= 1.02  # slight growth advantage
                
                # After maturation
                if self.age > 5 and dist_to_lumen > 0.1:
                    lumen_direction = self.position - lumen_center
                    migration_strength = 0.05 * min(1.0, self.age / 10.0)
                    self.migration_vector = (lumen_direction / np.linalg.norm(lumen_direction)) * migration_strength
                    
            # Glial cells are more like moderate energy and high connectivity
            elif (self.energy > 0.3 and local_curvature < low_curv_thresh 
                  and local_field > 0.4):
                self.role = 'glia_candidate'
                self.energy *= 0.98  # stable but slower growth
                

                
        # Apply migration
        if np.linalg.norm(self.migration_vector) > 0:
            self.position += self.migration_vector * 0.5
            self.migration_vector *= 0.9  # decay migration over time
        
        # Role-specific updates
        if self.role == 'stabilizer':
            self.energy *= 0.99
        elif self.role == 'explorer':
            self.energy *= 1.01
        elif self.role == 'messenger':
            self.energy += 0.03 * np.sin(step / 10.0)

    def is_ready_to_divide(self, echo_thresh=0.5):
        return self.energy > self.division_threshold and self.echo > echo_thresh

    def divide(self, id_counter):
        offset = np.random.randn(2) * 0.3  # smaller offset for neural cells
        daughter1 = NeuralNode(*(self.position + offset), 
                              init_energy=self.energy * 0.5, 
                              id_counter=id_counter)
        daughter2 = NeuralNode(*(self.position - offset), 
                              init_energy=self.energy * 0.5, 
                              id_counter=id_counter+1)
        daughter1.parent_id = self.id
        daughter2.parent_id = self.id
        return daughter1, daughter2

def create_morphogen_gradients(field_size):
    """Create anterior-posterior and dorsal-ventral gradients"""
    # Anterior-posterior gradient like Hox genes
    ap_gradient = np.zeros((field_size, field_size))
    for i in range(field_size):
        ap_gradient[i, :] = np.linspace(0.1, 0.3, field_size)  # anterior = high
    
    # Dorsal-ventral gradient like Sonic hedgehog from ventral
    dv_gradient = np.zeros((field_size, field_size))
    for j in range(field_size):
        # Ventral should have high concentration
        dv_gradient[:, j] = 0.2 * np.exp(-((j - field_size*0.2)**2) / (field_size/3)**2)
    
    return ap_gradient, dv_gradient

def run_neural_development_simulation():
    steps = 150
    dt = 0.1
    field_size = 100
    x = np.linspace(-10, 10, field_size)
    y = np.linspace(-10, 10, field_size)
    X, Y = np.meshgrid(x, y)
    
    # An organizing center
    source_center = np.array([0, 0])
    source_radius = 4.0
    source_intensity = 1.0
    R = np.sqrt((X - source_center[0])**2 + (Y - source_center[1])**2)
    source_field = source_intensity * np.exp(-((R / source_radius)**2))
    
    # Dynamic pulse
    pulse_origin = np.array([-8, -8])
    pulse_direction = np.array([1, 1]) / np.sqrt(2)
    pulse_speed = 2.0
    
    # Neural tube development 
    fusion_trigger = 30
    fusion_pressure = 0.02
    lumen_formation_delay = 10
    
    neural_field = np.zeros_like(source_field)
    ap_gradient, dv_gradient = create_morphogen_gradients(field_size)
    
    asymmetry_field = np.zeros_like(source_field)
    asym_center = np.array([5, -5])
    asym_radius = 7.0
    asym_strength = 0.1
    for i in range(field_size):
        for j in range(field_size):
            dist = np.linalg.norm(np.array([X[i, j], Y[i, j]]) - asym_center)
            asymmetry_field[i, j] = asym_strength * np.exp(-((dist / asym_radius) ** 2))
    
    nodes = []
    node_coords = set()
    id_counter = 0
    feedback_threshold = 0.2
    
    # Limits for different cell types
    MAX_STABILIZERS = 100
    MAX_EXPLORERS = 80
    MAX_NEURONS = 150
    MAX_GLIA = 100
    
    mid_x_idx = field_size // 2
    lumen_center = np.array([field_size // 2, field_size // 2])
    
    # Storage for visualization
    step_data = []
    
    print("Starting neural development simulation...")
    
    for t in range(steps):
        pulse_pos = pulse_origin + pulse_direction * pulse_speed * t * dt
        pulse_field = np.zeros_like(source_field)
        
        for i in range(field_size):
            for j in range(field_size):
                pos = np.array([X[i, j], Y[i, j]])
                rel = pos - pulse_pos
                dist = np.linalg.norm(rel)
                if dist < 3.0:
                    decay = np.exp(-dist**2 / 1.5)
                    pulse_field[i, j] = decay
        
        # Neural tube fusion 
        if t > fusion_trigger:
            fusion_intensity = min(1.0, (t - fusion_trigger) / 20.0)
            for i in range(field_size):
                neural_field[i, max(0, mid_x_idx - 2)] += fusion_pressure * fusion_intensity
                neural_field[i, min(field_size-1, mid_x_idx + 2)] += fusion_pressure * fusion_intensity
        
        # Central cavity
        if t > fusion_trigger + lumen_formation_delay:
            lumen_strength = 0.04 * min(1.0, (t - fusion_trigger - lumen_formation_delay) / 15.0)
            for i in range(field_size):
                for j in range(field_size):
                    dist_x = abs(i - field_size // 2)
                    dist_y = abs(j - field_size // 2) * 0.7
                    dist = np.sqrt(dist_x**2 + dist_y**2)
                    neural_field[i, j] -= lumen_strength * np.exp(-dist**2 / 12.0)
        
        total_field = source_field + pulse_field + asymmetry_field + ap_gradient + dv_gradient
        
        grad_x, grad_y = np.gradient(total_field)
        curvature = np.sqrt(grad_x**2 + grad_y**2)
        feedback_signal = total_field * curvature
        attractor_zone = feedback_signal > feedback_threshold
        
        neural_field[attractor_zone] += 0.08
        
        # Add some stabilization
        neural_field += 0.005 * np.exp(-curvature * 2)
        
        # Seed new nodes
        for i, j in np.argwhere(attractor_zone):
            coord = (X[i, j], Y[i, j])
            if coord not in node_coords:
                node_coords.add(coord)
                node = NeuralNode(*coord, id_counter=id_counter)
                nodes.append(node)
                id_counter += 1
        
        # Getting curvature for roles
        all_curvatures = []
        for node in nodes:
            ix = np.abs(x - node.position[0]).argmin()
            iy = np.abs(y - node.position[1]).argmin()
            if 0 <= ix < field_size and 0 <= iy < field_size:
                all_curvatures.append(curvature[iy, ix])
        
        updated_nodes = []
        for node in nodes:
            ix = np.abs(x - node.position[0]).argmin()
            iy = np.abs(y - node.position[1]).argmin()
            
            if 0 <= ix < field_size and 0 <= iy < field_size:
                local_field = total_field[iy, ix]
                local_curv = curvature[iy, ix]
                node.update(local_field, local_curv, all_curvatures, lumen_center, step=t)
                
                if node.is_ready_to_divide():
                    d1, d2 = node.divide(id_counter)
                    id_counter += 2
                    updated_nodes.extend([d1, d2])
                else:
                    updated_nodes.append(node)
            else:
                updated_nodes.append(node)  
        
        # Population control and role assignment
        ranked_nodes = sorted(updated_nodes, key=lambda z: z.energy, reverse=True)
        
        neurons = [n for n in ranked_nodes if n.role == 'neuron_candidate'][:MAX_NEURONS]
        glia = [n for n in ranked_nodes if n.role == 'glia_candidate'][:MAX_GLIA]
        proliferative = [n for n in ranked_nodes if n.role == 'proliferative']
        
        # Remaining nodes get traditional roles
        remaining = [n for n in ranked_nodes if n.role not in ['neuron_candidate', 'glia_candidate', 'proliferative']]
        stabilizers = remaining[:MAX_STABILIZERS]
        explorers = remaining[MAX_STABILIZERS:MAX_STABILIZERS + MAX_EXPLORERS]
        
        for node in stabilizers:
            node.role = 'stabilizer'
        for node in explorers:
            node.role = 'explorer'
        
        nodes = neurons + glia + proliferative + stabilizers + explorers
        
        if t % 10 == 0:  
            step_data.append({
                'step': t,
                'neural_field': neural_field.copy(),
                'curvature': curvature.copy(),
                'nodes': [(n.position.copy(), n.role, n.energy) for n in nodes],
                'total_field': total_field.copy()
            })
        
        if t % 20 == 0:
            neuron_count = len([n for n in nodes if n.role == 'neuron_candidate'])
            glia_count = len([n for n in nodes if n.role == 'glia_candidate'])
            prolif_count = len([n for n in nodes if n.role == 'proliferative'])
            print(f"Step {t}: Total={len(nodes)}, Neurons={neuron_count}, Glia={glia_count}, Proliferative={prolif_count}")
    
    return step_data, nodes, neural_field, curvature, X, Y

def visualize_neural_development(step_data, final_nodes, final_neural_field, final_curvature, X, Y):

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Curvature field
    im1 = axes[0, 0].contourf(X, Y, final_curvature, levels=50, cmap='viridis')
    axes[0, 0].set_title('Final Curvature Field')
    axes[0, 0].set_aspect('equal')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Neural field
    im2 = axes[0, 1].contourf(X, Y, final_neural_field, levels=50, cmap='plasma')
    axes[0, 1].set_title('Neural Field Development')
    axes[0, 1].set_aspect('equal')
    plt.colorbar(im2, ax=axes[0, 1])
    
    role_colors = {
        'neuron_candidate': 'red',
        'glia_candidate': 'blue', 
        'proliferative': 'green',
        'stabilizer': 'cyan',
        'explorer': 'magenta',
        'undecided': 'gray'
    }
    
    for node in final_nodes:
        color = role_colors.get(node.role, 'gray')
        size = 15 if node.role in ['neuron_candidate', 'glia_candidate'] else 8
        axes[0, 2].scatter(node.position[0], node.position[1], 
                          c=color, s=size, alpha=0.7)
    
    axes[0, 2].set_title('Cell Type Distribution')
    axes[0, 2].set_aspect('equal')
    axes[0, 2].set_xlim(-10, 10)
    axes[0, 2].set_ylim(-10, 10)
    
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=color, markersize=8, label=role)
                      for role, color in role_colors.items() if role != 'undecided']
    axes[0, 2].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
    
    energies = [node.energy for node in final_nodes]
    axes[1, 0].hist(energies, bins=30, alpha=0.7, color='orange')
    axes[1, 0].set_title('Node Energy Distribution')
    axes[1, 0].set_xlabel('Energy Level')
    axes[1, 0].set_ylabel('Count')
    
    roles = {}
    for node in final_nodes:
        if node.role not in roles:
            roles[node.role] = []
        roles[node.role].append(node.age)
    
    role_names = list(roles.keys())
    age_data = [roles[role] for role in role_names]
    axes[1, 1].boxplot(age_data, labels=role_names)
    axes[1, 1].set_title('Age Distribution by Cell Type')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    step_numbers = [data['step'] for data in step_data]
    neuron_counts = []
    glia_counts = []
    total_counts = []
    
    for data in step_data:
        neurons = len([node for pos, role, energy in data['nodes'] if role == 'neuron_candidate'])
        glia = len([node for pos, role, energy in data['nodes'] if role == 'glia_candidate'])
        total = len(data['nodes'])
        
        neuron_counts.append(neurons)
        glia_counts.append(glia)
        total_counts.append(total)
    
    axes[1, 2].plot(step_numbers, neuron_counts, 'r-', label='Neurons', linewidth=2)
    axes[1, 2].plot(step_numbers, glia_counts, 'b-', label='Glia', linewidth=2)
    axes[1, 2].plot(step_numbers, total_counts, 'k--', label='Total', linewidth=2)
    axes[1, 2].set_title('Population Dynamics')
    axes[1, 2].set_xlabel('Simulation Step')
    axes[1, 2].set_ylabel('Cell Count')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    if len(step_data) >= 3:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        time_points = [0, len(step_data)//2, -1]
        titles = ['Early Development', 'Mid Development', 'Late Development']
        
        for idx, (time_idx, title) in enumerate(zip(time_points, titles)):
            data = step_data[time_idx]
            
            im = axes[0, idx].contourf(X, Y, data['neural_field'], levels=30, cmap='plasma')
            axes[0, idx].set_title(f'{title}\nNeural Field (Step {data["step"]})')
            axes[0, idx].set_aspect('equal')
            
            for pos, role, energy in data['nodes']:
                color = role_colors.get(role, 'gray')
                size = 12 if role in ['neuron_candidate', 'glia_candidate'] else 6
                axes[0, idx].scatter(pos[0], pos[1], c=color, s=size, alpha=0.8)
            
            axes[1, idx].contourf(X, Y, data['curvature'], levels=30, cmap='viridis')
            axes[1, idx].set_title(f'Curvature Field (Step {data["step"]})')
            axes[1, idx].set_aspect('equal')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Running neural development simulation...")
    step_data, final_nodes, final_neural_field, final_curvature, X, Y = run_neural_development_simulation()
    
    print(f"\nSimulation complete!")
    print(f"Final population: {len(final_nodes)} nodes")
    
    role_counts = {}
    for node in final_nodes:
        role_counts[node.role] = role_counts.get(node.role, 0) + 1
    
    print("Final cell type distribution:")
    for role, count in role_counts.items():
        print(f"  {role}: {count}")

    visualize_neural_development(step_data, final_nodes, final_neural_field, final_curvature, X, Y)