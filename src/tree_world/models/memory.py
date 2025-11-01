import torch


def geodesic_distance(location1: torch.Tensor, location2: torch.Tensor):
    """
    Calculate the geodesic distance between two points on the unit sphere.
    """
    return torch.acos((location1*location2).sum(dim=-1)) / torch.pi


def normalize_location(location: torch.Tensor, temperature: float=0.25):
    """ 
    Convert a point in free R^d to the simplex and thence to the positive orthant of the unit sphere.
    """
    first_part = torch.exp(location / temperature)
    second_part = 1 + first_part.sum(dim=-1, keepdim=True)
    return torch.sqrt(torch.cat([first_part / second_part, 1 / second_part], dim=-1))


def unnormalize_location(location: torch.Tensor, temperature: float=0.25):
    """ Convert a point on the unit sphere back to free R^d.
    """
    return temperature * (2 * (torch.log(location[..., :-1]) - torch.log(location[..., -1])[..., None]))


def make_key_grid(embed_dim: int, points_per_dim: int, epsilon: float=1e-6):
    if embed_dim > 8:
        raise ValueError("Embedding dimension too high for grid search. Use a lower dimension or a different key generation method.")

    # first, space out a line with the desired increment between points
    points = torch.linspace(-1 + epsilon, 1 - epsilon, points_per_dim)

    # unfold to infinity, with concentration at the origin (half the points will be between -1 and 1)
    points = points / (1 - points.abs())

    # then, create a grid of points by taking the cartesian product of the points
    grid = torch.cartesian_product([points] * embed_dim)
    return grid, grid.shape[0]


class BidrectionalMemory(torch.nn.Module):
    """
    This is a memory cache to store pairs of (key, value) embeddings.

    :param key_dim: The dimension of the key embeddings.
    :param value_dim: The dimension of the value embeddings.
    :param memory_size: The maximum length of the memory cache.
    :param dropout: The dropout rate.

    """
    def __init__(self, query_dim: int, value_dim: int, embed_dim: int, memory_size: int, dropout: float=0.1, batch_size: int=1,
                 matching_keys_target: int=8, matching_keys_tolerance: float=0.5, threshold: float=0.5):
        super().__init__()
        self.query_dim = query_dim
        self.value_dim = value_dim 
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.dropout = dropout
        self.memory_padding_index = 0

        # the memory has a set of position sensors that are randomly distributed on the unit sphere
        # these will not change over time TODO -- should they be optimized to minimize retrieval faults?
        # note these keys are on the first orthant of the unit sphere
        key_base = normalize_location(torch.randn(memory_size, embed_dim))

        self.memory_keys = torch.nn.Buffer(key_base)


        key_distances = torch.arccos(key_base @ key_base.transpose(0, 1))  # (memory_size, memory_size), between 0 and 1
        self.memory_keys_distances = torch.nn.Buffer(key_distances)
        self.memory_values = torch.nn.Buffer(torch.zeros(batch_size, memory_size, embed_dim))
        self.memory_values_spherical = torch.nn.Buffer(normalize_location(self.memory_values))

        self.query_proj = torch.nn.Linear(query_dim, embed_dim, bias=False)
        self.key_in_proj = torch.nn.Linear(query_dim, embed_dim, bias=False)
        self.key_out_proj = torch.nn.Linear(embed_dim, query_dim, bias=False)
        self.write_proj = torch.nn.Linear(value_dim, embed_dim, bias=False)
        self.read_proj = torch.nn.Linear(embed_dim, value_dim, bias=False)
        self.align_read_write_projections()

        self.factor = embed_dim ** -0.5

        self.score_exponent = 5.0
        self.threshold = threshold

        self.avg_keys_written = 0.0
        self.avg_keys_ema_factor = 0.9
        self.calibrate(keys_written_target=matching_keys_target, tolerance=matching_keys_tolerance)

    def align_read_write_projections(self):
        """
        Align the read projection to the write projection by minimizing the mean squared error between the two.

        Initialize to the pseudo-inverse of the write projection.
        """
        if self.value_dim == self.embed_dim:
            self.write_proj.weight.data = torch.eye(self.embed_dim)
            self.read_proj.weight.data = torch.eye(self.embed_dim)
        
        else:

            pseudo_inverse = torch.linalg.pinv(self.write_proj.weight.data)
            self.read_proj.weight.data = pseudo_inverse

        if self.query_dim == self.embed_dim:
            self.key_in_proj.weight.data = torch.eye(self.embed_dim)
            self.key_out_proj.weight.data = torch.eye(self.embed_dim)
        
        else:
            pseudo_inverse = torch.linalg.pinv(self.key_in_proj.weight.data)
            self.key_out_proj.weight.data = pseudo_inverse

    def calibrate(self, keys_written_target: float=8, tolerance: float=0.5):
        """
        Calibrate the score exponent to achieve the desired number of keys written.
        """
        score_exponent = 1.0

        random_keys = normalize_location(self.key_in_proj(torch.randn(min([self.memory_size, 1000]), 1, self.query_dim)))
        random_scores = self.score(random_keys, self.memory_keys[None, ...], exponent=score_exponent)
        keys_written = (random_scores >= self.threshold).sum(dim=-1).float().mean().item()

        # note: Binary search would be faster, but this is good enough for now
        while not (-tolerance  < (keys_written - keys_written_target) < tolerance):
            print(f"score exponent: {score_exponent}, keys written: {keys_written}, target: {keys_written_target}")
            if keys_written > keys_written_target:
                score_exponent *= 2.0
            else:
                score_exponent *= 0.99
            
            self.avg_keys_written = 0.0
            random_keys = normalize_location(self.key_in_proj(torch.randn(min([self.memory_size, 1000]), 1, self.query_dim)))
            random_scores = self.score(random_keys, self.memory_keys[None, ...], exponent=score_exponent)
            keys_written = (random_scores >= self.threshold).sum(dim=-1).float().mean().item()

        print(f"BidrectionalMemory calibrated score exponent: {score_exponent} with {keys_written} keys written")
        
        self.avg_keys_written = keys_written
        self.score_exponent = score_exponent

    def reset(self):
        self.memory_values.data.zero_()

    def score(self, queries: torch.Tensor, keys: torch.Tensor, suppress: torch.LongTensor=None, threshold: float=None, exponent: float=None):
        """
        Score the queries against the keys.

        Note queries and keys are on the unit sphere, but values are in free R^d (d=embed_dim, so this is down-projected).

        :param queries: The queries to read from the memory cache. Has shape (batch_size, num_queries, embed_dim).
        :param keys: The keys to read from the memory cache. Has shape (batch_size, num_keys, embed_dim).
        :param suppress: The indices of the keys to suppress. Has shape (batch_size, num_queries, max_suppressed), None by default. Zeroes can always be suppressed to pad the tensor
        :param threshold: The threshold for the match scores. Default is self.threshold. Results with scores below this threshold are ignored.
        :return: The scores. Has shape (batch_size, num_queries, num_keys).
        """
        if exponent is None:
            exponent = self.score_exponent

        if threshold is None:
            threshold = self.threshold

        # compute the alignment scores
        # due to normalization, this is a number in [0, 1]
        # scores = (batch_size, num_queries, num_keys)
        if keys.shape[0] == 1:
            keys = keys.repeat(queries.shape[0], 1, 1)

        scores = torch.bmm(queries, keys.transpose(1, 2)).pow(exponent)

        if suppress is not None:
            # suppress should be (batch_size, num_queries, max_suppressed)
            scores = scores.scatter(dim=-1, index=suppress, src=0)

        if threshold is not None and threshold > 0:
            max_score = scores.max(dim=-1).values  # (batch_size, num_queries)
            threshold = torch.where(max_score < threshold, 0.9 * max_score, threshold)[..., None]
            scores = scores.masked_fill(scores < threshold, 0)

        # active_indices = scores.squeeze().nonzero().squeeze()
        # print(f"scores {scores.shape}: {active_indices.detach().cpu().numpy().tolist()}")

        return scores

    def read(self, queries: torch.Tensor, suppress: torch.LongTensor=None, threshold: float=None, spherical_query: bool=False):
        """
        Read from the memory cache by keys. Allows suppression of certain keys.

        Note queries and keys are on the unit sphere, but values are in free R^d (d=embed_dim, so this is down-projected).

        :param queries: The queries to read from the memory cache. Has shape (batch_size, num_queries, query_dim).
        :param suppress: The indices of the keys to suppress. Has shape (batch_size, num_queries, max_suppressed), None by default. Zeroes can always be suppressed to pad the tensor
        :param threshold: The threshold for the match scores. Default is self.threshold. Results with scores below this threshold are ignored.
        :return: values, indices, number of results found, and match scores for each query.
        """
        if threshold is None:
            threshold = self.threshold

        if self.query_dim != self.embed_dim:
            queries = self.query_proj(queries)

        if spherical_query:
            queries_spherical = queries
        else:
            queries_spherical = normalize_location(queries)

        # scores has shape (batch_size, num_queries, num_keys)
        scores = self.score(queries_spherical, self.memory_keys[None, ...], suppress, threshold)
        weights = scores / scores.sum(dim=-1, keepdim=True)

        # pre_values has shape (batch_size, num_queries, value_dim)
        pre_values = torch.bmm(weights, self.memory_values)

        if self.value_dim != self.embed_dim:
            values = self.read_proj(pre_values)
        else:
            values = pre_values

        return values


    def search(self, values: torch.Tensor, num_results: int=1, suppress: torch.LongTensor=None, threshold: float=None,
               center_factor: float=0.01, surround_factor: float=0.1, diversity_steps: int=5, mix_factor: float=0.2,
               exponent: float=None):
        """
        Search the values for the most similar values to queries, returning the closest matching keys.

        :param values: The values to search the memory cache for. Has shape (batch_size, num_queries, value_dim).
        :param num_results: The number of results to return for each query, one by default.
        :param suppress: The indices of the keys to suppress. Has shape (batch_size, num_queries, max_suppressed), None by default.
        :param threshold: The threshold for the match scores. Default is self.threshold. Results with scores below this threshold are ignored.
        :return: values, indices, number of results found, and match scores for each query.
        """
        if threshold is None:
            threshold = self.threshold

        values_spherical = normalize_location(self.write_proj(values))

        # scores has shape (batch_size, num_queries, memory_size) and is between 0 and 1
        scores = self.score(values_spherical, self.memory_values_spherical, suppress, threshold)

        # now on-center off-surround competition to select a diverse match set
        # based on the distances between the keys
        # scores_center will be (batch_size, num_queries, memory_size, memory_size)
        # this method recursively convolves with a mexican hat kernel
        # TODO: we need 0 < center_factor < surround_factor <= 1; but what are good values?
        # TODO: we need to operate on a sparser result set to avoid the quadratic complexity in the memory size
        if diversity_steps > 0:
            center_kernel = torch.exp(-((self.memory_keys_distances / center_factor)**2)).unsqueeze(0).unsqueeze(1)
            surround_kernel = torch.exp(-((self.memory_keys_distances / surround_factor)**2)).unsqueeze(0).unsqueeze(1)

            for i in range(diversity_steps):
                scores = scores[..., None] * center_kernel - mix_factor * scores[..., None] * surround_kernel
                scores = scores.sum(dim=-1)
        
        # get the top k results for each query, result is (batch_size, num_queries, num_results)
        scores, indices = torch.topk(scores, k=num_results, dim=-1, largest=True, sorted=True)

        # pull the values from self.memory_keys using the indices; has shape (batch_size, num_queries, num_results, embed_dim)
        proposed_keys = torch.nn.functional.embedding(indices, self.memory_keys, padding_idx=self.memory_padding_index)

        proposed_keys = proposed_keys.view(self.batch_size, -1, self.embed_dim + 1)
        proposed_keys.requires_grad = True

        target_values = values[..., None, :].expand(-1, -1, num_results, -1).view(self.batch_size, -1, self.value_dim)

        for i in range(5):
            proposed_values = self.read(proposed_keys, threshold=threshold, spherical_query=True)
            error = torch.norm(proposed_values - target_values, dim=-1)
            print(f"{i} error {error.detach().cpu().numpy().tolist()}")
            grad = torch.autograd.grad(error.sum(), proposed_keys, create_graph=True)[0]
            proposed_keys = proposed_keys - 0.01 * grad

        proposed_keys = proposed_keys.view(self.batch_size, values.shape[1], num_results, self.embed_dim + 1)

        output_keys = self.key_out_proj(unnormalize_location(proposed_keys))

        found = (scores > threshold).sum(dim=-1)

        return output_keys, indices, found, scores


    @torch.no_grad()
    def write(self, keys: torch.Tensor, values: torch.Tensor, suppress: torch.LongTensor=None, threshold: float=None, epsilon: float=1e-6):
        """
        Write to the memory cache. 

        :param keys: The keys to write to the memory cache. Has shape (batch_size, num_keys, key_dim).
        :param values: The values to write to the memory cache. Has shape (batch_size, num_keys, value_dim).
        """
        if keys.shape[0] != values.shape[0]:
            raise ValueError("The number of keys and values must be the same.")
        
        if keys.shape[0] == 0:
            return 

        if threshold is None:
            threshold = self.threshold

        # scores & weights have shape (batch_size, num_keys, memory_size)
        keys_spherical = normalize_location(self.key_in_proj(keys))
        scores = self.score(keys_spherical, self.memory_keys[None, ...], suppress, threshold)
        weights = scores / scores.sum(dim=-1, keepdim=True)

        # prior_values & new_values have shape (batch_size, num_keys, embed_dim)
        prior_values = torch.bmm(weights, self.memory_values)
        new_values = self.write_proj(values)

        # now our updating deltas have shape (batch_size, memory_size, embed_dim)
        weighted_new_values = torch.bmm(weights.transpose(1, 2), new_values)

        # this factor preserves the prior values and has shape (batch_size, num_keys)
        factor = 1 - weights.pow(2).sum(dim=-1)

        # this implements an update rule v' = w v_new + (1 - w^T w) (v_new / v_prior) v
        # where v is self.memory_values, w is weights, v_new is new_values, v_prior is prior_values
        # this rule guarantess the a read of the new values will return the new values while preserving the prior values
        # and storing the new values predominantly in the keys that were most similar to the query point
        extra_new_values = factor[..., None] * new_values * (new_values / prior_values)

        # avoid division by zero in two different ways, in which case we just write the new values
        condition = (factor[..., None] > epsilon) and (prior_values > epsilon)
        new_values = torch.where(condition, weighted_new_values + extra_new_values, new_values)

        # TODO: make sparse updates, and use scatter to update the memory values

        max_weight = weights.max(dim=1).values  # (batch_size, memory_size)
        condition = (max_weight > epsilon)[..., None]
        print(f"condition shape {condition.shape}, writing to {condition.sum(dim=-2).squeeze().detach().cpu().numpy().tolist()} keys")
        new_values = torch.where(condition, new_values, self.memory_values)
        new_values_spherical = torch.where(condition, normalize_location(new_values), self.memory_values_spherical)

        keys_written = condition.sum(dim=-2).float().mean().item()
        self.avg_keys_written = self.avg_keys_ema_factor * self.avg_keys_written + (1 - self.avg_keys_ema_factor) * keys_written

        self.memory_values = new_values
        self.memory_values_spherical = new_values_spherical



if __name__ == "__main__":
    
    query_dim = 32
    value_dim = 32
    embed_dim = 32
    memory_size = 2048

    # test normalizations
    location = torch.randn(1000, embed_dim)
    location_spherical = normalize_location(location)
    spherical_dist = geodesic_distance(location_spherical[..., None, :], location_spherical[None, ..., :])
    relocation = unnormalize_location(location_spherical)
    error = torch.norm(relocation - location, dim=-1).mean().item()
    print(f"error on re-normalization: {error}")
    print(f"spherical distance: {spherical_dist.shape} -- {torch.nan_to_num(spherical_dist).mean().item()}")
    print(f"avg similarity: {(location_spherical @ location_spherical.transpose(0, 1)).mean().item()}")

    memory = BidrectionalMemory(query_dim, value_dim, embed_dim, memory_size)

    queries = torch.randn(16, query_dim)
    values = torch.randn(16, value_dim)

    print("queries: ", queries[0].cpu().numpy().tolist())
    print("values: ", values[0].cpu().numpy().tolist())

    initial = memory.read(queries[:1, None, :]).squeeze().detach()
    print("initial read: ", torch.norm(initial).item())

    memory.write(queries[:1, None, :], values[:1, None, :])
    value_read = memory.read(queries[:1, None, :]).squeeze()
    error = torch.norm(value_read - values[0]).item()
    print(f"0 error on read after write: {error}")

    for i, (query, value) in enumerate(zip(queries, values)):
        memory.write(query[None, None, :], value[None, None, :])
        value_read = memory.read(query[None, None, :]).squeeze()
        error = torch.norm(value_read - value).item()
        print(f"{i} error on read after write: {error}")

    # Now read back the values
    num_clashes = 0
    for i, (query, value) in enumerate(zip(queries, values)):
        value_read = memory.read(query[None, None, :]).squeeze()
        error = torch.norm(value_read - value).item()
        if error > 0.01:
            num_clashes += 1
        print(f"{i} error on read after subsequeunt writes: {error}")

    print(f"number of clashes: {num_clashes}")

    keys, indices, found, scores = memory.search(values[:1, None, :], diversity_steps=0, num_results=5)
    print("search: ", keys[0].squeeze().detach().cpu().numpy().tolist())
    print("indices: ", indices[0].squeeze().detach().cpu().numpy().tolist())
    print("found: ", found[0].squeeze().detach().cpu().numpy().tolist())
    print("scores: ", scores[0].squeeze().detach().cpu().numpy().tolist())

    print("avg keys written: ", memory.avg_keys_written)

    
