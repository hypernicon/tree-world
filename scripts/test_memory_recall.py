import torch

from tree_world.simulation import TreeWorld, TreeWorldConfig, SimpleSensor
from tree_world.models.memory import BidirectionalMemory
from tree_world.models.drives import train_drive_classifier


if __name__ == "__main__":

    # create a world and memory
    print("Creating world and memory...")
    config = TreeWorldConfig()
    world = TreeWorld.random_from_config(config)
    config.embed_dim = 1024
    memory = BidirectionalMemory(location_dim=config.dim, sensory_dim=config.sensory_embedding_dim, embed_dim=config.embed_dim)

    # print out the tree locations
    for tree in world.trees:
        print(tree.tree_id, tree.name, tree.location.detach().cpu().numpy().tolist())
    
    # fill the memory with sensor data
    print("Filling memory with sensor data...")
    sensor = SimpleSensor.from_config(config)
    points = torch.linspace(-500, 500, 100)
    grid = torch.cartesian_prod(points, points)

    trees = []
    for point in grid:
        distance, embedding, tree = sensor.sense(world, point, None)
        memory.write(point[None, :], torch.ones(config.dim)[None, :], embedding[None, :].clone())
        trees.append(tree)

    for tree in world.trees:
        memory.write(tree.location[None, :], torch.ones(config.dim)[None, :], tree.embedding[None, :].clone())
        trees.append(tree)

    # make a drive classifier
    print("Training drive classifier...")
    drive_classifier, drive_keys = train_drive_classifier(config, with_ids=True)

    # can we find the trees in the memory?
    print("Searching for trees in the memory...")
    for tree in world.trees:
        read = memory.read(tree.location[None, :], torch.ones(config.dim)[None, :]).squeeze()
        similarity = (tree.embedding.clone() * read).sum()
        print(f"For {tree.tree_id} of type {tree.name}, similarity score at position: {similarity.item():.2f}")
        name_embedding = world.tree_id_embeddings_dict[tree.tree_id].clone()
        name_similarity = (name_embedding * read).sum()
        print(f"For {tree.tree_id} of type {tree.name}, similarity score at name: {name_similarity.item():.2f}")


        top_locations, top_location_sds, top_senses, found, num_found = memory.search(
            name_embedding[None, :].clone(), num_results=5, threshold=0.0, diversity_steps=0
        )
        print(f"For {tree.tree_id} of type {tree.name}, found {num_found[0].item()} trees in the memory")
        for i in range(num_found[0].item()):
            distance = torch.norm(top_locations[0, i] - tree.location)
            score = (name_embedding * top_senses[0, i]).sum()
            print(f"\tDistance to {tree.name}: {distance.item():.2f}, similarity score: {score.item():.2f}")

