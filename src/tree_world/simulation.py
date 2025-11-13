import random
import math
import sys
import torch
from typing import List, Tuple

from tree_world.embeddings import embed_text_sentence_transformers
from tree_world.live_viz_matplotlib import LiveVizMPL



class TreeWorldConfig:
    dim: int = 2

    # Agent
    max_health: int = 1000
    move_cost: int = 2
    rest_cost: int = 1
    eat_distance: float = 10.0
    fruit_health_increase: float = 25.0

    # Model 
    model_type: str = "AgentModel"
    simple_tem: bool = True

    # Sensory inputs
    sensory_embedding_dim: int = 1024
    sensory_embedding_model: str = "BAAI/bge-large-en-v1.5"
    
    location_dim: int = 2
    embed_dim: int = 32
    dropout: float = 0.1
    num_guesses: int = 5

    # Sensor
    sensor_type: str = "SimpleSensor"
    max_sense_distance: float = 100.0
    heading_tolerance: float = 0.9

    # Trees
    num_trees: int = 25
    tree_spacing: float = 100
    regrow_every: int = 1000
    max_fruit: int = 5
    poisonous_probability: float = 0.3
    poison_fruits: List[str] = ["manchineel", "nightshade", "strychnine fruit", "desert rose"]
    edible_fruits: List[str] = ["apple", "banana", "cherry", "date", "elderberry", "fig", "mango", "nectarine", "orange", "papaya", "pear", "plum"]
    can_see_fruit_distance: float = 100.0

    tree_ids = [
        "Joey", "Chandler", "Ross", "Rachel", "Monica", "Phoebe", "Gunther", "Bob", "George", "Kramer",
        "Stewie", "Brian", "Peter", "Lois", "Meg", "Chris", "Buster", "Bill", "Suzie", "Stanton",
        "Bruce", "Clinton", "Barack", "Donald", "Joe", "Kamala", "Earnest", "Thomas", "Andrew", 
        "Hillary", "Sarah", "Elizabeth", "Abigail", "Mary", "Ann", "Jane", "Lydia", "Hannah", "Ralph",
        "Richard", "Jimmy", "Fred", "Barney", "Homer", "Bart", "Lisa", "Maggie", "Ned", 
        "Maude", "Sam", "Todd", "Tommy", "Ulysses", "Victor", "Winston", "Xavier", "Albert",
        "Ben", "Carl", "Dave", "Eric", "Frank", "Hal", "Ian", "Jack", "Larry", "Michael", "Nate", 
        "Oliver", "Paul", "Quincy", "Terry", "Ruprecht", "Sergei", "Tobias", "Monty", "Nigel", "Oscar",
        "Peter", "Quentin", "Rupert", "Samson", "Toby", "Yogi",
        "Ziggy", "Alice", "Charlie", "Diana", "Eve", "Isaac",
        "Karen", "Larry", "Mia", "Olivia", "Paul",  "Uma", "Wendy", "Yvonne", "Zachary", "Ada", "Bertha", "Clara",
        "Diana", "Eleanor", "Florence", "Grace", "Irene", "Julia", "Katherine", "Lillian", "Mary",
        "Nora", "Ophelia", "Patricia", "Queenie", "Rebecca", "Theresa", "Ursula", "Victoria",
        "Xandra", "Zoe", "Agatha", "Bella", "Chloe", "Diana", "Ella", "Fiona", "Greta",
        "Iris", "Jasmine",
    ]


class Tree:
    """
    A tree is a point in space, with a given embedding.
    It has a location in the world.
    It has a fruit, which can be harvested by the agent.
    It can regrow fruit over time.
    """
    def __init__(self, tree_id: str, name: str, embedding: torch.Tensor, location: torch.Tensor, 
                       max_fruit: int, is_poisonous: bool, regrow_every: int):
        self.tree_id = tree_id
        self.name = name
        self.embedding = embedding
        self.location = location
        self.max_fruit = max_fruit
        self.fruit = max_fruit
        self.is_poisonous = is_poisonous
        self.regrow_every = regrow_every
        self.regrow_count = 0

    def step(self):
        self.regrow_count = self.regrow_count + 1
        if self.regrow_count >= self.regrow_every:
            self.regrow_fruit()
            self.regrow_count = 0

    def regrow_fruit(self):
        self.fruit = self.max_fruit

    def harvest_fruit(self):
        harvested_fruit = min(self.fruit, 1)
        self.fruit = max(self.fruit - 1, 0)
        return harvested_fruit


class Sensor:
    def sense(self, world: 'TreeWorld', position: torch.Tensor, heading: torch.Tensor):
        raise NotImplementedError("Subclasses must implement this method")
    
    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        if config.sensor_type == "SimpleSensor":
            return SimpleSensor.from_config(config)
        elif config.sensor_type == "DirectionalSensor":
            return DirectionalSensor.from_config(config)
        else:
            raise ValueError(f"Unknown sensor type: {config.sensor_type}")


class SimpleSensor(Sensor):
    """
    Return a distance-weighted average of the tree embeddings.

    This is a diffusion model; it models the tree locations as a Gaussian distribution, and the sensor is a Gaussian kernel.
    """

    def sense(self, world: 'TreeWorld', position: torch.Tensor, heading: torch.Tensor):
        tree_locations = world.get_tree_locations()
       
        # in a diffusion model, we have 1/2 du/dt = \sum_i d^2u/dx_i^2 = \Delta u (\Delta = Laplacian)
        # This is for u(x, t) = c t^{-d/2} \exp(-1/(2t) \|x\|^2) where d is the dimension of the space.
        # The "variance" of a diffusion is t^{1/2}; t is a virtual time parameter.
        # we want to choose t so that at config.max_sense_distance, the kernel is 0.01.
        # Now, u(0, t) = c t^{d/2}, and we want u(0, t) = 1 and u(config.max_sense_distance, t) = a = small number.
        # This reduces to u(x, t) = \exp(-1/(2t) \|x\|^2) with \exp(-r^2 / 2t) = a ==> t = -r^2 / 2 \log(a).
        # for r = config.max_sense_distance
        a = 0.1
        virtual_time = world.config.max_sense_distance**2 / 2 / math.log(1/a)
        distances = torch.norm(tree_locations - position[None, :], dim=1)
        kernel = torch.exp(-0.5 * (distances).pow(2) / virtual_time)
        
        tree_embeddings = world.get_tree_embeddings()
        embedding = torch.mm(kernel[None, :], tree_embeddings).squeeze()

        closest_index = torch.argmin(distances)
        closest_distance = distances[closest_index]
        closest_tree = world.trees[closest_index]

        return closest_distance, embedding, closest_tree

    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        return cls()


class DirectionalSensor(Sensor):
    """
    Given a position and a heading, return the distance and embedding of the tree closest to the heading.

    The heading is a unit vector in the direction of the tree.

    The position is a point in space.

    For a tree at location `tree_location`, the tree heading is `(tree_location - position) / ||tree_location - position||`.

    We can compare the dot product of the heading and the tree heading; we allow a tolerance for the dot product; 
    the sensor will return the tree embedding and distance if the dot product is within the tolerance.

    If multiple trees are within the tolerance, the sensor will return the distance and embedding of the tree closest to the position.

    """
    def __init__(self, heading_tolerance: float=0.9, max_distance: float=10.0):
        self.heading_tolerance = heading_tolerance
        self.max_distance = max_distance

    def sense(self, world: 'TreeWorld', position: torch.Tensor, heading: torch.Tensor):
        tree_locations = world.get_tree_locations()
        tree_headings = (tree_locations - position[None, :])
        tree_headings = tree_headings / (torch.norm(tree_headings, dim=1) + 1e-6)[:, None]
        dot_products = torch.mm(tree_headings, heading[:, None]).squeeze()

        close_trees = dot_products > self.heading_tolerance
        if not close_trees.any():
            # print(f"No tree found along heading {heading.numpy().tolist()}, returning None")
            return None, None

        indices = torch.arange(len(world.trees))[close_trees]
        distances = torch.norm(tree_locations[indices] - position[None, :], dim=1)
        closest_index = torch.argmin(distances)

        tree_index = indices[closest_index]
        distance = distances[closest_index]

        
        if distance > self.max_distance:
            # print(f"Tree found at distance {distance} along heading {heading.numpy().tolist()} but is too far, returning None")
            return None, None, None

        # TODO: return the closest tree if you're not looking at one?

        # print(f"Tree ({world.trees[tree_index].name}) found at distance {distance} and index {tree_index}")
        return distance, world.tree_embeddings[tree_index], world.trees[tree_index]

    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        return cls(config.heading_tolerance, config.max_sense_distance)


class AgentModel:
    def __init__(self, sensory_embedding_dim: int, sensory_embedding_model: str, dim: int=2, can_see_fruit_distance: float=10.0, max_distance: float=100.0):
        self.sensory_embedding_dim = sensory_embedding_dim
        self.sensory_embedding_model = sensory_embedding_model
        self.dim = dim

        self.can_see_fruit_distance = can_see_fruit_distance
        self.max_distance = max_distance

        self.setup_tree_embeddings()

        self.last_heading_delta = None

    def reset(self):
        self.last_heading_delta = None

    def setup_tree_embeddings(self):
        num_fruit_strs = ["no fruit", "one fruit", "two fruit", "several fruit", "many fruit"]
        fruit_embeddings = embed_text_sentence_transformers(num_fruit_strs, self.sensory_embedding_model)
        self.num_fruit_embeddings = {num_fruit_str: fruit_embedding for num_fruit_str, fruit_embedding in zip(num_fruit_strs, fruit_embeddings)}

    def get_tree_embedding(self, distance: float, embedding: torch.Tensor, num_fruit: int):
        if distance is None:
            return torch.zeros(self.sensory_embedding_dim)
        elif distance > self.can_see_fruit_distance:
            return embedding
        else: 
            if num_fruit == 0:
                num_fruit_str = "no fruit"
            elif num_fruit == 1:
                num_fruit_str = "one fruit"
            elif num_fruit == 2:
                num_fruit_str = "two fruit"
            elif 3 <= num_fruit <= 5:
                num_fruit_str = "several fruit"
            else:
                num_fruit_str = "many fruit"

            fruit_embedding = self.num_fruit_embeddings[num_fruit_str]
            return embedding + fruit_embedding

    def get_orthogonal_direction(self, heading: torch.Tensor):
        orthogonal_direction = torch.randn(self.dim)
        orthogonal_direction = orthogonal_direction - torch.dot(orthogonal_direction, heading) * heading
        orthogonal_direction = orthogonal_direction / torch.norm(orthogonal_direction)
        if self.last_heading_delta is not None:
            dp = torch.dot(orthogonal_direction, self.last_heading_delta)
            if dp < 0:
                # keep the orthogonal direction in the same direction
                orthogonal_direction = -orthogonal_direction
        self.last_heading_delta = orthogonal_direction
        return orthogonal_direction

    def get_action(self, distance: float, embedding: torch.Tensor, heading: torch.Tensor, health: float,
                   agent_location: torch.Tensor=None, obj_location: torch.Tensor=None):
        if distance is None:
            print("No tree found, randomizing heading and position delta")
            # pick an orthogonal direction to the heading
            orthogonal_direction = self.get_orthogonal_direction(heading)
            new_heading = heading + 0.1 * orthogonal_direction
            new_heading = new_heading / torch.norm(new_heading)
            position_delta = torch.randn(self.dim).abs()
        else:
            print("Tree found, moving along heading")
            new_heading = heading
            position_delta = heading

        return position_delta, new_heading
    
    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        return cls(config.sensory_embedding_dim, config.sensory_embedding_model, config.dim, config.can_see_fruit_distance, config.max_sense_distance)


class Agent:
    """
    An agent is a point in space, with a given embedding.
    It has a location in the world.
    It has a health, which can be restored by eating edible fruit or depleted by eating poisonous fruit, 
    moving around the world, and by time going by.

    It has a sensor that can detect the distance to a tree and the embedding of the tree in a fixed direction.
    """
    def __init__(self, model: AgentModel, sensor: Sensor, max_health: int, dim: int=2, move_cost: int=2, rest_cost: int=1,
                 eat_distance: float=1.0, fruit_health_increase: float=10.0):
        self.location = torch.zeros(dim)
        self.heading = torch.randn(dim)
        self.heading = self.heading / torch.norm(self.heading)
        self.sensor = sensor
        self.model = model
        self.dim = dim
        self.max_health = max_health
        self.health = max_health

        self.move_cost = move_cost
        self.rest_cost = rest_cost

        self.max_distance = 100.0
        self.eat_distance = eat_distance
        self.fruit_health_increase = fruit_health_increase
        self.fruit_eaten = 0
        self.poisonous_fruit_eaten = 0

        self.total_movement = 00

    def reset(self):
        self.health = self.max_health
        self.fruit_eaten = 0
        self.poisonous_fruit_eaten = 0
        self.model.reset()
        self.total_movement = 0.0


    def step(self, world: 'TreeWorld'):
        distance, embedding, tree = self.sensor.sense(world, self.location, self.heading)
        num_fruit = tree.fruit if tree is not None else 0
        embedding = self.model.get_tree_embedding(distance, embedding, num_fruit)


        if distance is not None and distance < self.eat_distance:
            # print(f"Eating fruit from tree {tree.name} with fruit amount {num_fruit}")
            self.fruit_eaten += 1
            if tree.is_poisonous:
                self.poisonous_fruit_eaten += 1

            # NOTE: this requires the agent to face the tree when eating fruit
            self.eat_fruit(1, tree.is_poisonous)
            tree.harvest_fruit()
        
        position_delta, self.heading = self.model.get_action(distance, embedding, self.heading, self.health / self.max_health,
                                                             self.location, tree.location if tree is not None else None)

        if position_delta is None:
            self.rest()
        else:
            self.move(position_delta)

    def move(self, direction: torch.Tensor):
        self.location = self.location + direction
        self.health = self.health - self.move_cost
        self.total_movement = self.total_movement + torch.norm(direction)

    def rest(self):
        self.health = self.health - self.rest_cost

    def eat_fruit(self, fruit: int, is_poisonous: bool):
        if is_poisonous:
            self.health = self.health - fruit * self.fruit_health_increase
        else:
            self.health = self.health + fruit * self.fruit_health_increase

        if self.health > self.max_health:
            self.health = self.max_health

    @classmethod
    def from_config(cls, config: TreeWorldConfig):
        if config.model_type == "AgentModel":
            model = AgentModel.from_config(config)
        elif config.model_type == "StateBasedAgent":
            from tree_world.agents import StateBasedAgent
            model = StateBasedAgent.from_config(config)
        elif config.model_type == "StateBasedAgentWithDriveEmbedding":
            from tree_world.agents import StateBasedAgentWithDriveEmbedding
            model = StateBasedAgentWithDriveEmbedding.from_config(config)
        elif config.model_type == "TEMAgent":
            from tree_world.agents import TEMAgent
            model = TEMAgent.from_config(config)
        elif config.model_type == "HomeostaticAgent":
            from tree_world.agents import HomeostaticAgent
            model = HomeostaticAgent.from_config(config)
        elif config.model_type == "PathTracingTEMAgent":
            from tree_world.agents import PathTracingTEMAgent
            model = PathTracingTEMAgent.from_config(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")

        sensor = Sensor.from_config(config)
        agent = cls(model, sensor, config.max_health, config.dim, config.move_cost, config.rest_cost, config.eat_distance, 
                    config.fruit_health_increase)
        agent.location = torch.randn(config.dim) * config.tree_spacing * config.dim
        agent.heading = torch.randn(config.dim) / torch.norm(torch.randn(config.dim))
        return agent


class TreeWorld:
    """
    A tree world is an open space of a given dimension, with trees placed at given locations.
    The trees have an embedding, intended to be derived from an embedding model.
    Each tree has fruit, which can be harvested by the agent. Once the fruit run out, it can no longer be harvested.
    Trees regrow fruit over time, but slowly enough that the agent will die if it just sits and waits for the fruit to regrow.

    Some fruit are poisonous, and will kill the agent if eaten.
    Some fruit are edible, and will provide the agent with energy if eaten.
    The agent can move around the world, and harvest fruit from trees.
    The agent dies if its health reaches 0.
    The agent's health is restored by eating edible fruit.
    The agent's health is depleted by eating poisonous fruit, moving around the world, and by time going by.

    The agent has a sensor that can detect the distance to a tree and the embedding of the tree.
    """

    def __init__(self, trees: List[Tree], agent: Agent, config: TreeWorldConfig):
        self.trees = trees
        self.agent = agent
        self.tree_locations = torch.stack([tree.location for tree in trees])
        self.config = config
        self.record_positions = []
        self.record_healths = []

        self.last_target_location_estimated = None

        self.tree_names, self.tree_embeddings_type, self.tree_name_embeddings, self.tree_id_embeddings, self.tree_id_embeddings_dict = (
            self.make_name_embeddings()
        )

    def get_tree_locations(self):
        return self.tree_locations

    def make_name_embeddings(self):
        tree_types = self.config.poison_fruits + self.config.edible_fruits
        tree_embeddings = embed_text_sentence_transformers(tree_types, self.config.sensory_embedding_model)
        tree_id_embeddings = embed_text_sentence_transformers(self.config.tree_ids, self.config.sensory_embedding_model)
        return (
            tree_types, 
            tree_embeddings, 
            {tree_type: tree_embedding for tree_type, tree_embedding in zip(tree_types, tree_embeddings)},
            tree_id_embeddings,
            {tree_id: tree_id_embedding for tree_id, tree_id_embedding in zip(self.config.tree_ids, tree_id_embeddings)}
        )

    def get_tree_ids_from_embeddings(self, embeddings: torch.Tensor):
        # matches will be (batch_size, num_trees, embed_dim) @ (embed_dim, num_tree_ids) = (batch_size, num_trees, num_tree_ids)
        squeeze = False
        if embeddings.ndim == 2:
            squeeze = True
            embeddings = embeddings[None, :, :]
        matches = torch.matmul(embeddings, self.tree_id_embeddings.transpose(0, 1).unsqueeze(0))

        # id_idxs will be (batch_size, num_trees)
        id_idxs = matches.argmax(dim=-1)

        all_tree_ids = []
        for batch in range(embeddings.shape[0]):
            tree_ids = []
            for tree in range(embeddings.shape[1]):
                id_idx = id_idxs[batch, tree]
                tree_id = self.config.tree_ids[id_idx]
                tree_ids.append(tree_id)

            all_tree_ids.append(tree_ids)

        if squeeze:
            return all_tree_ids[0]
        else:
            return all_tree_ids

    def get_tree_embeddings(self):
        return torch.stack([tree.embedding for tree in self.trees])

    def step(self):
        self.agent.step(self)
        for tree in self.trees:
            tree.step()

        if False: # hasattr(self.agent.model, "target_location_estimated") and self.agent.model.target_changed:
            target_location_estimated = self.agent.model.target_location_estimated
            if target_location_estimated is not None:
                if self.last_target_location_estimated is not None:

                    if (target_location_estimated == self.last_target_location_estimated).all():
                        return
                    
                # print(f"Target location estimated changed from {self.last_target_location_estimated} to {target_location_estimated}")
                
                self.last_target_location_estimated = target_location_estimated

                target_sensory = self.agent.model.target_sensory
                target_name = None
                if target_sensory is not None:
                   affinity = torch.matmul(self.tree_embeddings_type, target_sensory[:, None]).squeeze(-1)
                   target_name = self.tree_names[torch.argmax(affinity).item()]

                min_distance = float("inf")
                min_distance_to_sensory = float("inf")
                min_sensory_distance = float("inf")
                closest_tree = None
                closest_to_sensory = None
                for tree in self.trees:
                    distance = torch.norm(tree.location - target_location_estimated)
                    if distance < min_distance:
                        min_distance = distance
                        closest_tree = tree
                    
                    if target_sensory is not None:
                        sensory_distance = torch.norm(tree.embedding - target_sensory)
                        if sensory_distance < min_sensory_distance:
                            min_sensory_distance = sensory_distance
                        if target_name is not None and tree.name == target_name and distance < min_distance_to_sensory:
                            min_distance_to_sensory = torch.norm(tree.location - target_location_estimated)
                            closest_to_sensory = tree

                if closest_tree is not None:
                    print(f"Closest tree to target location estimated: {closest_tree.name} at distance {min_distance.item()}; is it for food? {self.agent.model.target_location_is_for_food}")

                if closest_to_sensory is not None:
                    print(f"Closest tree to target matching sensory: {closest_to_sensory.name} at distance {min_distance_to_sensory.item()}")
                elif target_sensory is not None:
                    print(f"No tree found to target matching sensory: closest sensory distance {min_sensory_distance.item()}")

    def reset(self):
        self.agent.location = torch.zeros(self.agent.dim) # torch.randn(self.agent.dim) * self.config.tree_spacing * self.agent.dim
        self.agent.heading = torch.ones(self.agent.dim) / torch.norm(torch.ones(self.agent.dim)) # torch.randn(self.agent.dim) / torch.norm(torch.randn(self.agent.dim))
        self.agent.reset()
        for tree in self.trees:
            tree.regrow_fruit()
    
    def run(self, num_steps: int, record=False, allow_death=True, live_viz=None):
        self.reset()
        if record:
            self.record_positions = [self.agent.location.numpy().tolist()]
            self.record_healths = [self.agent.health]

        for i in range(num_steps):
            self.step()

            if live_viz is not None:
                live_viz.update(i)

            if record:
                self.record_positions.append(self.agent.location.numpy().tolist())
                self.record_healths.append(self.agent.health)

            if self.agent.health <= 0 and allow_death:
                return False

            if i % 100 == 0 and i > 0 and hasattr(self.agent.model, "train"):
                self.agent.model.train()
        return True

    def randomize(self):
        # recreate the world with new trees, without recreating the agent
        config = self.config

        tree_ids = []
        tree_types = []
        poison_trees = []
        for i in range(config.num_trees):
            is_poisonous = random.random() < config.poisonous_probability
            tree_id = random.choice(config.tree_ids)
            tree_ids.append(tree_id)
            name = random.choice(config.poison_fruits) if is_poisonous else random.choice(config.edible_fruits)
            tree_types.append(name)
            poison_trees.append(is_poisonous)
        
        tree_locations = torch.randn(config.num_trees, config.dim) * config.tree_spacing * config.dim
        tree_embeddings = (
            embed_text_sentence_transformers(tree_types, config.sensory_embedding_model)
            + embed_text_sentence_transformers(tree_ids, config.sensory_embedding_model)
        )
        assert tree_embeddings.shape == (config.num_trees, config.sensory_embedding_dim)

        trees = []
        for i in range(config.num_trees):
            tree = Tree(tree_ids[i], tree_types[i], tree_embeddings[i], tree_locations[i], config.max_fruit, poison_trees[i], config.regrow_every)
            trees.append(tree)

        self.trees = trees
        self.tree_locations = torch.stack([tree.location for tree in trees])
        self.tree_names, self.tree_embeddings_type, self.tree_name_embeddings, self.tree_id_embeddings, self.tree_id_embeddings_dict = (
            self.make_name_embeddings()
        )

        self.reset()

        dists = torch.cdist(tree_embeddings, tree_embeddings)
        # print(f"dists: {dists.numpy()}")

    @classmethod
    def random_from_config(cls, config: TreeWorldConfig):
        tree_ids = []
        tree_types = []
        poison_trees = []
        for i in range(config.num_trees):
            is_poisonous = random.random() < config.poisonous_probability
            tree_id = random.choice(config.tree_ids)
            tree_ids.append(tree_id)
            name = random.choice(config.poison_fruits) if is_poisonous else random.choice(config.edible_fruits)
            tree_types.append(name)
            poison_trees.append(is_poisonous)
        
        tree_locations = torch.randn(config.num_trees, config.dim) * config.tree_spacing * config.dim
        tree_embeddings = (
            embed_text_sentence_transformers(tree_types, config.sensory_embedding_model)
            + embed_text_sentence_transformers(tree_ids, config.sensory_embedding_model)
        )
        assert tree_embeddings.shape == (config.num_trees, config.sensory_embedding_dim)

        trees = []
        for i in range(config.num_trees):
            tree = Tree(tree_ids[i], tree_types[i], tree_embeddings[i], tree_locations[i], config.max_fruit, poison_trees[i], config.regrow_every)
            trees.append(tree)

        agent = Agent.from_config(config)
        return cls(trees, agent, config)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        steps = int(sys.argv[1])
    else:
        steps = 1000

    config = TreeWorldConfig()
    config.model_type = "PathTracingTEMAgent"  # "HomeostaticAgent"   # "StateBasedAgentWithDriveEmbedding"

    world = TreeWorld.random_from_config(config)
    viz = LiveVizMPL(world)

    runs = 100 if hasattr(world.agent.model, "train") else 1
    print(f"Running {runs} runs")
    print("--------------------------------")
    for i in range(runs):
        world.randomize()
        viz.reset()

        s = min([steps, (i + 1) * 1000])
        print(f"Running tree world {i} for {s} steps...")
        world.run(s, record=(i == runs - 1), allow_death=False, live_viz=viz)
        print()
        print("Tree world run complete.")

        print(f"Agent health: {world.agent.health}")
        print(f"Agent fruit eaten: {world.agent.fruit_eaten}")
        print(f"Agent poisonous fruit eaten: {world.agent.poisonous_fruit_eaten}")
        print(f"Agent total movement: {world.agent.total_movement}")
        print(f"Agent final location: {torch.norm(world.agent.location).item()}")

        # capture the memory
        memory = world.agent.model.tem_model.memory
        torch.save([memory, memory.memory_locations, memory.memory_location_sds, memory.memory_senses], f"memory_example.pt")

        # if hasattr(world.agent.model, "train"):
        #     print("Training agent model")
        #     world.agent.model.train()
        #     print("Agent model trained")

        print("--------------------------------")

    from tree_world.visualize import visualize_treeworld_run
    visualize_treeworld_run(
        world.tree_locations.numpy().tolist(),
        [tree.name for tree in world.trees],
        [tree.is_poisonous for tree in world.trees],
        world.record_positions,
        world.record_healths,
        config.max_health,
        title="TreeWorld run",
        save_path="tree_world_run.png",
    )
