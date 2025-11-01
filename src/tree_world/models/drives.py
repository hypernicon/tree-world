import torch

class DriveClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_drives: int, hidden_dim: int=128):
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, num_drives)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.softmax(x)
        return x


class DriveEmbeddingClassifier(torch.nn.Module):
    def __init__(self, input_dim: int, num_drives: int):
        super().__init__()
        self.input_dim = input_dim
        self.drive_embeddings = torch.nn.Embedding(num_drives, input_dim)
        self.softmax = torch.nn.Softmax(dim=1)
    
    def forward(self, x: torch.Tensor):
        y = torch.matmul(x, self.drive_embeddings.weight.transpose(0, 1))
        y = self.softmax(y)
        return y



def train_drive_classifier(config: "TreeWorldConfig"):
    from tree_world.embeddings import embed_text_sentence_transformers

    num_drives = 3
    input_dim = config.sensory_embedding_dim

    tree_embeddings = embed_text_sentence_transformers(config.poison_fruits + config.edible_fruits, config.sensory_embedding_model)
    tree_embeddings = tree_embeddings.clone()

    labels = [0] * len(config.poison_fruits) + [1] * len(config.edible_fruits)
    labels = torch.tensor(labels)

    states = ["no fruit", "one fruit", "two fruit", "several fruit", "many fruit"]
    state_values = [0, 0.25, 0.5, 0.75, 1.0]
    state_embeddings = embed_text_sentence_transformers(states, config.sensory_embedding_model)
    state_embeddings = state_embeddings.clone()

    new_embeddings = [tree_embedding for tree_embedding in tree_embeddings]
    new_targets = [[1, 0, 0] for _ in range(len(config.poison_fruits))]
    new_targets += [[0, 1, 0] for _ in range(len(config.edible_fruits))]
    for i, state in enumerate(states):
        state_embedding = state_embeddings[i]
        for j,fruit in enumerate(config.poison_fruits):
            new_embedding = state_embedding + tree_embeddings[j]
            new_target = [state_values[i], 0, 1 - state_values[i]]
            new_targets.append(new_target)
            new_embeddings.append(new_embedding)
        
        base = len(config.poison_fruits)
        for j,fruit in enumerate(config.edible_fruits):
            new_embedding = state_embedding + tree_embeddings[base + j]
            new_target = [0, state_values[i], 1 - state_values[i]]
            new_targets.append(new_target)
            new_embeddings.append(new_embedding)
    
    new_embeddings = torch.stack(new_embeddings)
    new_targets = torch.tensor(new_targets)

    num_drives = 3
    model = DriveEmbeddingClassifier(input_dim, num_drives)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for i in range(1000):
        optimizer.zero_grad()
        outputs = model(new_embeddings)
        # literal cross entropy loss between outputs and new_targets = -sum(new_targets * log(outputs))
        loss = -torch.sum(new_targets * torch.log(outputs), dim=1).mean()
        loss.backward()
        optimizer.step()

    mse = torch.nn.functional.mse_loss(outputs, new_targets)

    accuracy = 0.0
    cnt = 0
    for i, tree_embedding in enumerate(tree_embeddings):
        if i < len(config.poison_fruits):
            target_idx = 0
        else:
            target_idx = 1
        accuracy += (torch.abs(outputs[i, target_idx] - 1) < 0.05).float()
        cnt += 1

    print(f"Drive Embedding Classifier Accuracy (with fruit amount): {accuracy*100 / cnt:.2f}%")

    floor_idx = cnt
    for i, state in enumerate(states):
        base_idx = i * (len(config.poison_fruits) + len(config.edible_fruits))
        for j, fruit in enumerate(config.poison_fruits):
            # print(f"{fruit} with {state}: {outputs[base_idx + j, 0]:.2f} vs. {state_values[i]:.2f}")
            accuracy += (torch.abs(outputs[floor_idx + base_idx + j, 0] - state_values[i]) < 0.05).float()
            cnt += 1
            
        base = len(config.poison_fruits)
        for j, fruit in enumerate(config.edible_fruits):
            # print(f"{fruit} with {state}: {outputs[base_idx + base + j, 1]:.2f} vs. {state_values[i]:.2f}")
            accuracy += (torch.abs(outputs[floor_idx + base_idx + base + j, 1] - state_values[i]) < 0.05).float()
            cnt += 1

    print(f"Drive Embedding Classifier Loss (with fruit amount): {loss.item()} MSE: {mse.item()} Accuracy: {accuracy*100 / cnt:.2f}%")

    return model, {"poison": 0, "edible": 1, "neutral": 2}



if __name__ == "__main__":

    from tree_world.embeddings import embed_text_sentence_transformers
    from tree_world.simulation import TreeWorldConfig

    config = TreeWorldConfig()

    num_drives = 2
    input_dim = config.sensory_embedding_dim

    tree_embeddings = embed_text_sentence_transformers(config.poison_fruits + config.edible_fruits, config.sensory_embedding_model)
    print(tree_embeddings.shape)

    labels = [0] * len(config.poison_fruits) + [1] * len(config.edible_fruits)
    labels = torch.tensor(labels)

    model = DriveClassifier(input_dim, num_drives)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    tree_embeddings = tree_embeddings.clone()

    # model.to("mps")
    # tree_embeddings = tree_embeddings.to("mps")
    # labels = labels.to("mps")

    for i in range(1000):
        optimizer.zero_grad()
        outputs = model(tree_embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
    print(f"Drive Classifier Loss: {loss.item()} Accuracy: {accuracy:.2f}%")

    embedding_model = DriveEmbeddingClassifier(input_dim, num_drives)
    embedding_optimizer = torch.optim.Adam(embedding_model.parameters(), lr=0.01)
    embedding_criterion = torch.nn.CrossEntropyLoss()

    for i in range(1000):
        embedding_optimizer.zero_grad()
        outputs = embedding_model(tree_embeddings)
        loss = embedding_criterion(outputs, labels)
        loss.backward()
        embedding_optimizer.step()

    accuracy = (outputs.argmax(dim=1) == labels).float().mean() * 100
    print(f"Drive Embedding Classifier Loss: {loss.item()} Accuracy: {accuracy:.2f}%")



    # now check for the presence of fruits
    model = train_drive_classifier(config)



